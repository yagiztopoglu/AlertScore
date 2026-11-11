"""Değerlendirme metrikleri.

İki ayrı bakış açısı vardır ve KARIŞTIRILMAZ:
- Nokta bazlı (point-wise): her zaman noktası ayrı bir örnek (Precision, Recall, F1, PR-AUC).
- Olay bazlı (event-wise): her anomali penceresi bir olay (Event Recall, Detection Delay,
  Yanlış alarm = hiçbir anomali penceresine değmeyen alarm olayı).

Not: NAB anomali pencereleri geniştir (genelde saatler-günler) ve gerçek başlangıç anını
değil, anomalinin çevresini işaretler. Bu yüzden nokta bazlı Precision/Recall pencere
genişliğinden etkilenir; olay bazlı metrikler daha anlamlıdır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def point_labels(ts, windows) -> np.ndarray:
    """Zaman damgası pencerelerin içindeyse 1, değilse 0."""
    t = pd.DatetimeIndex(ts).to_numpy()
    lab = np.zeros(len(t), dtype=int)
    for a, b in windows:
        lab[(t >= np.datetime64(a)) & (t <= np.datetime64(b))] = 1
    return lab


def val_event_count(ts, windows, lo: int, hi: int) -> int:
    """[lo, hi) indeks aralığına (doğrulama bölümü) değen anomali penceresi sayısı."""
    ts = pd.DatetimeIndex(ts)
    if hi <= lo:
        return 0
    start, end = ts[lo], ts[hi - 1]
    return sum(1 for a, b in windows if a <= end and b >= start)


def alert_episodes(pred: np.ndarray, merge_gap: int = 3) -> list[tuple[int, int]]:
    """Ardışık (aralarında en fazla merge_gap adım boşluk olan) alarm noktalarını tek alarm olayına birleştirir."""
    idx = np.flatnonzero(pred)
    if idx.size == 0:
        return []
    eps, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 <= merge_gap:
            prev = i
        else:
            eps.append((int(start), int(prev)))
            start = prev = i
    eps.append((int(start), int(prev)))
    return eps


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def evaluate(ts, scores, threshold: float, windows, test_start: int, merge_gap: int = 3) -> dict:
    """Test bölümü (test_start'tan sona kadar) üzerinde tüm metrikleri hesaplar.

    threshold dışarıdan verilir (eğitim/doğrulama verisinden seçilmiş olmalı).
    Tanımsız değerler (ör. test bölümünde hiç anomali yoksa Recall) NaN döner.
    """
    ts = pd.DatetimeIndex(ts)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=1e12)
    y = point_labels(ts, windows)[test_start:]
    s = scores[test_start:]
    t = ts[test_start:]
    pred = s > threshold

    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    pr_auc = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")

    episodes = alert_episodes(pred, merge_gap)
    false_eps = [e for e in episodes if y[e[0]:e[1] + 1].sum() == 0]
    days = (t[-1] - t[0]).total_seconds() / 86400 if len(t) > 1 else float("nan")

    n_events, detected, delays = 0, 0, []
    for a, b in windows:
        m = np.flatnonzero((t >= a) & (t <= b))
        if m.size == 0:
            continue
        n_events += 1
        hits = np.flatnonzero(pred[m])
        if hits.size:
            detected += 1
            first = t[m[hits[0]]]
            delays.append((first - max(a, t[0])).total_seconds() / 60)

    return {
        "threshold": float(threshold),
        "test_points": int(len(t)),
        "test_days": float(days),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "pr_auc": pr_auc,
        "alert_episodes": len(episodes),
        "false_alarms": len(false_eps),
        "false_alarms_per_day": _ratio(len(false_eps), days) if days == days else float("nan"),
        # Normal noktaların yüzde kaçı alarm verdi? Alarm sürekli açık kalırsa "alarm olayı" sayısı
        # düşük görünür (tek uzun olay); bu metrik o durumu yakalar (özellikle veri kaymasında).
        "false_alarm_point_rate": _ratio(fp, fp + tn),
        "events": n_events,
        "events_detected": detected,
        "event_recall": _ratio(detected, n_events),
        "mean_delay_min": float(np.mean(delays)) if delays else float("nan"),
    }
