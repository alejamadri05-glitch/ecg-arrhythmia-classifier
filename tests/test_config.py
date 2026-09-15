from ecg import config


def test_no_record_shared_between_ds1_and_ds2():
    """Prueba anti-fuga de datos: ningún paciente puede estar en entrenamiento y prueba."""
    assert set(config.DS1).isdisjoint(config.DS2)


def test_splits_have_22_unique_records_each():
    assert len(config.DS1) == len(set(config.DS1)) == 22
    assert len(config.DS2) == len(set(config.DS2)) == 22


def test_paced_records_excluded_from_both_splits():
    assert set(config.PACED).isdisjoint(config.DS1 + config.DS2)


def test_aami_mapping():
    expected = {
        "N": {"N", "L", "R", "e", "j"},
        "S": {"A", "a", "J", "S"},
        "V": {"V", "E"},
        "F": {"F"},
    }
    for cls, symbols in expected.items():
        assert {s for s, c in config.AAMI_MAP.items() if c == cls} == symbols
    assert set(config.AAMI_MAP.values()) == set(config.CLASSES)


def test_q_and_non_beat_symbols_are_not_classified():
    for sym in ["/", "f", "Q", "+", "~", "|"]:
        assert sym not in config.AAMI_MAP
    for sym in ["+", "~", "|"]:
        assert sym not in config.BEAT_SYMBOLS


def test_window_is_252_samples():
    assert config.WINDOW == 252
