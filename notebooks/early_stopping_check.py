"""Exploración de early stopping y estabilizadores para la CNN (Fase 4).

Entrena 25 épocas y mide en cada una la validación interna y el fold externo.
Uso: python notebooks/early_stopping_check.py <fold> <variante>
Variantes: base | lr3e-4 | aug | wd | ema | lr3e-4+aug+ema
Resultados de la corrida original: reports/cnn_early_stopping_check.txt
"""

import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ecg.evaluate import summary
from ecg.models.cnn import CNNClassifier, CNNConfig, ECGNet, augment_batch, loss_class_weights, seed_everything
from ecg.segment import load_split
from ecg.train import patient_folds
from sklearn.preprocessing import StandardScaler

fold = int(sys.argv[1])
variant = sys.argv[2]  # base | lr3e-4 | aug | wd
EPOCHS = 25

d = load_split("ds1")
X, F, y, g = d["X"], d["F"], d["y"], d["record"]
tr, te = patient_folds(g)[fold]
clf = CNNClassifier(CNNConfig())
itr, iva = clf.inner_split(y[tr], g[tr])
itr, iva = tr[itr], tr[iva]

seed_everything(42)
dev = torch.device("mps")
lookup = {c: i for i, c in enumerate("NSVF")}
yi = np.array([lookup[v] for v in y])
sc = StandardScaler().fit(F[itr])
T = lambda idx: (torch.from_numpy(X[idx]).unsqueeze(1).to(dev), torch.from_numpy(sc.transform(F[idx]).astype(np.float32)).to(dev), torch.from_numpy(yi[idx]).to(dev))
xtr, rtr, ytr = T(itr)
sets = {"inner": T(iva), "outer": T(te)}
model = ECGNet().to(dev)
lr = 3e-4 if "lr3e-4" in variant else 1e-3
wd = 1e-2 if "wd" in variant else 0.0
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True) if "ema" in variant else None
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) if wd else torch.optim.Adam(model.parameters(), lr=lr)
w = torch.tensor(loss_class_weights(yi[itr]), dtype=torch.float32, device=dev)
lossf = nn.CrossEntropyLoss(weight=w)
rows = []
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(itr), device=dev)
    for i in range(0, len(perm), 256):
        b = perm[i:i + 256]
        xb = augment_batch(xtr[b]) if "aug" in variant else xtr[b]
        opt.zero_grad(); lossf(model(xb, rtr[b]), ytr[b]).backward(); opt.step()
        if ema is not None: ema.update_parameters(model)
    net = ema.module if ema is not None else model
    net.eval()
    row = {"ep": ep}
    with torch.no_grad():
        for name, (xs, rs, ys) in sets.items():
            logits = torch.cat([net(xs[i:i + 4096], rs[i:i + 4096]) for i in range(0, len(xs), 4096)])
            row[f"{name}_loss"] = lossf(logits, ys).item()
            pred = np.array(list("NSVF"))[logits.argmax(1).cpu().numpy()]
            idx = iva if name == "inner" else te
            row[f"{name}_f1"] = summary(y[idx], pred)["macro_f1"]
    rows.append(row)
df = pd.DataFrame(rows).round(3)
print(f"fold {fold} {variant}")
late = df[df.ep >= 6]
print("outer_f1 épocas 6-25: media %.3f  sd %.3f  | inner_f1 sd %.3f | salto medio entre épocas %.3f"
      % (late.outer_f1.mean(), late.outer_f1.std(), late.inner_f1.std(), late.outer_f1.diff().abs().mean()))
print("corr outer_f1 vs inner_f1: %.2f | vs -inner_loss: %.2f" % (df.outer_f1.corr(df.inner_f1), df.outer_f1.corr(-df.inner_loss)))
print("epoch by inner_f1:", int(df.ep[df.inner_f1.idxmax()]), "-> outer", df.outer_f1[df.inner_f1.idxmax()],
      "| epoch by inner_loss:", int(df.ep[df.inner_loss.idxmin()]), "-> outer", df.outer_f1[df.inner_loss.idxmin()],
      "| max outer", df.outer_f1.max())
