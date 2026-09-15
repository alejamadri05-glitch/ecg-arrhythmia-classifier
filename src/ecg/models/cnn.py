"""Red convolucional 1D (PyTorch): morfología del latido (convoluciones) + ritmo (features RR).

Reglas anti-fuga dentro de `CNNClassifier.fit`:
- El StandardScaler de las RR y los pesos de clase se calculan solo con el subconjunto de
  entrenamiento.
- El early stopping usa pacientes de validación *internos*, separados de los de entrenamiento.
  Nunca se usa el fold que después se evalúa.
"""

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import WeightedRandomSampler

from ecg import config
from ecg.evaluate import summary

MODEL_VERSION = "1.0.0"
RR_COLUMNS = {"all": [0, 1, 2, 3], "ratios": [2, 3], "none": []}


class ECGNet(nn.Module):
    def __init__(self, n_classes: int = 4, n_rr: int = 4):
        super().__init__()
        self.n_rr = n_rr
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )  # fmt: skip
        self.head = nn.Sequential(
            nn.Linear(128 + n_rr, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )  # fmt: skip

    def forward(self, x: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 252) latido normalizado; rr: (B, n_rr) features RR escaladas."""
        z = self.conv(x).squeeze(-1)
        if self.n_rr:
            z = torch.cat([z, rr], dim=1)
        return self.head(z)


@dataclass
class CNNConfig:
    rr: str = "all"  # "all" | "ratios" | "none"
    balance: str = "weights"  # "weights" (pérdida ponderada) | "sampler" | "none"
    augment: bool = False  # ruido gaussiano, escala de amplitud, desplazamiento temporal
    invert: bool = False  # inversión de polaridad aleatoria (requiere augment)
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 7
    inner_val_splits: int = 4  # 1/4 de los pacientes de entrenamiento para early stopping
    seed: int = config.SEED
    device: str = "auto"


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def loss_class_weights(y_idx: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """Pesos inversos a la frecuencia (media 1 entre clases presentes; 0 si la clase no está)."""
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, len(y_idx) / (n_classes * np.maximum(counts, 1)), 0.0)
    return w / w[counts > 0].mean()


def augment_batch(x: torch.Tensor, invert: bool = False, max_shift: int = 5) -> torch.Tensor:
    """Aumento de datos por latido: escala 0.8–1.2, desplazamiento ±5 muestras, ruido σ=0.05."""
    n, _, length = x.shape
    dev = x.device
    x = x * (0.8 + 0.4 * torch.rand(n, 1, 1, device=dev))
    shift = torch.randint(-max_shift, max_shift + 1, (n, 1), device=dev)
    idx = (torch.arange(length, device=dev).unsqueeze(0) - shift).clamp(0, length - 1)
    x = torch.gather(x, 2, idx.unsqueeze(1))
    x = x + 0.05 * torch.randn_like(x)
    if invert:
        sign = torch.where(torch.rand(n, 1, 1, device=dev) < 0.5, -1.0, 1.0)
        x = x * sign
    return x


class CNNClassifier:
    def __init__(self, cfg: CNNConfig | None = None):
        self.cfg = cfg or CNNConfig()
        self.classes = list(config.CLASSES)
        self.rr_cols = RR_COLUMNS[self.cfg.rr]

    # --- preparación ---------------------------------------------------------------------
    def _rr(self, F: np.ndarray) -> np.ndarray:
        if not self.rr_cols:
            return np.zeros((len(F), 0), dtype=np.float32)
        return self.scaler_.transform(F[:, self.rr_cols]).astype(np.float32)

    def _tensors(self, X: np.ndarray, F: np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
        return x.to(self.device_), torch.from_numpy(self._rr(F)).to(self.device_)

    def inner_split(self, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Separa pacientes de entrenamiento y de validación interna (estratificado por clase)."""
        sgkf = StratifiedGroupKFold(
            n_splits=self.cfg.inner_val_splits, shuffle=True, random_state=self.cfg.seed
        )
        return next(sgkf.split(np.zeros(len(y)), y, groups))

    # --- entrenamiento -------------------------------------------------------------------
    def fit(self, X, F, y, groups=None, epochs: int | None = None) -> "CNNClassifier":
        """Con `groups`: early stopping en pacientes internos. Con `epochs`: épocas fijas, sin
        validación (para reentrenar con todos los pacientes)."""
        cfg = self.cfg
        if (groups is None) == (epochs is None):
            raise ValueError("Pasar `groups` (early stopping) o `epochs` (épocas fijas), no ambos")
        seed_everything(cfg.seed)
        self.device_ = pick_device(cfg.device)
        lookup = {c: i for i, c in enumerate(self.classes)}
        y_idx = np.array([lookup[v] for v in y], dtype=np.int64)

        if groups is not None:
            tr, va = self.inner_split(y, groups)
        else:
            tr, va = np.arange(len(y)), np.array([], dtype=np.int64)
        self.train_idx_, self.val_idx_ = tr, va

        self.scaler_ = StandardScaler().fit(F[tr][:, self.rr_cols]) if self.rr_cols else None
        x_tr, rr_tr = self._tensors(X[tr], F[tr])
        y_tr = torch.from_numpy(y_idx[tr]).to(self.device_)

        self.model_ = ECGNet(len(self.classes), len(self.rr_cols)).to(self.device_)
        opt = torch.optim.Adam(self.model_.parameters(), lr=cfg.lr)
        weight = None
        if cfg.balance == "weights":
            weight = torch.tensor(loss_class_weights(y_idx[tr]), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=None if weight is None else weight.to(self.device_))
        if cfg.balance == "sampler":
            counts = np.bincount(y_idx[tr], minlength=len(self.classes))
            sample_w = torch.as_tensor(1.0 / counts[y_idx[tr]], dtype=torch.double)
            gen = torch.Generator().manual_seed(cfg.seed)
            sampler = WeightedRandomSampler(sample_w, len(tr), replacement=True, generator=gen)

        n_epochs = epochs if epochs is not None else cfg.max_epochs
        best_f1, best_state, wait = -1.0, None, 0
        self.history_ = []
        for epoch in range(1, n_epochs + 1):
            self.model_.train()
            if cfg.balance == "sampler":
                order = torch.as_tensor(list(sampler), device=self.device_)
            else:
                order = torch.randperm(len(tr), device=self.device_)
            total = 0.0
            for i in range(0, len(order), cfg.batch_size):
                b = order[i : i + cfg.batch_size]
                xb = augment_batch(x_tr[b], cfg.invert) if cfg.augment else x_tr[b]
                opt.zero_grad()
                loss = loss_fn(self.model_(xb, rr_tr[b]), y_tr[b])
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            record = {"epoch": epoch, "train_loss": total / len(order)}

            if len(va):
                val_f1 = summary(y[va], self.predict(X[va], F[va]))["macro_f1"]
                record["val_macro_f1"] = val_f1
                if val_f1 > best_f1:
                    best_f1, wait, self.best_epoch_ = val_f1, 0, epoch
                    best_state = copy.deepcopy(self.model_.state_dict())
                else:
                    wait += 1
            self.history_.append(record)
            if len(va) and wait >= cfg.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        else:
            self.best_epoch_ = n_epochs
        return self

    # --- inferencia ----------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, F: np.ndarray, batch: int = 4096) -> np.ndarray:
        self.model_.eval()
        x, rr = self._tensors(X, F)
        out = [torch.softmax(self.model_(x[i : i + batch], rr[i : i + batch]), dim=1).cpu()
               for i in range(0, len(x), batch)]  # fmt: skip
        return torch.cat(out).numpy() if out else np.zeros((0, len(self.classes)), np.float32)

    def predict(self, X: np.ndarray, F: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes)[self.predict_proba(X, F).argmax(axis=1)]

    # --- persistencia --------------------------------------------------------------------
    def save(self, path) -> None:
        torch.save(
            {
                "model_version": MODEL_VERSION,
                "config": asdict(self.cfg),
                "classes": self.classes,
                "best_epoch": self.best_epoch_,
                "state_dict": {k: v.cpu() for k, v in self.model_.state_dict().items()},
                "scaler_mean": None if self.scaler_ is None else self.scaler_.mean_.tolist(),
                "scaler_scale": None if self.scaler_ is None else self.scaler_.scale_.tolist(),
            },
            path,
        )

    @staticmethod
    def load(path, device: str = "cpu") -> "CNNClassifier":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        clf = CNNClassifier(CNNConfig(**{**ckpt["config"], "device": device}))
        clf.device_ = pick_device(device)
        clf.classes = ckpt["classes"]
        clf.best_epoch_ = ckpt["best_epoch"]
        clf.model_version = ckpt["model_version"]
        clf.scaler_ = None
        if ckpt["scaler_mean"] is not None:
            clf.scaler_ = StandardScaler()
            clf.scaler_.mean_ = np.asarray(ckpt["scaler_mean"])
            clf.scaler_.scale_ = np.asarray(ckpt["scaler_scale"])
            clf.scaler_.var_ = clf.scaler_.scale_**2
            clf.scaler_.n_features_in_ = len(clf.scaler_.mean_)
        clf.model_ = ECGNet(len(clf.classes), len(clf.rr_cols))
        clf.model_.load_state_dict(ckpt["state_dict"])
        clf.model_.to(clf.device_).eval()
        return clf
