"""Deney motoru: model çalıştırma, benchmark ve dayanıklılık deneyleri.

Metodoloji (hepsi tek yerde, tüm modeller için aynı):
- Zamansal bölme: ilk %40 eğitim, sonraki %20 doğrulama (eşik seçimi), son %40 test.
  Rastgele karıştırma YOK; zaman serisinde gelecek geçmişe sızmamalı.
- Ön işleme (ölçekleme) yalnızca eğitim bölümünden öğrenilir.
- Eşik = doğrulama bölümü skorlarının `quantile` yüzdeliği. Etiket kullanılmaz, test verisi
  eşik seçiminde hiç görülmez. Tüm modeller aynı kuralı kullanır.
- Metrikler yalnızca test bölümünde hesaplanır.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Series
from .metrics import evaluate, point_labels, val_event_count
from .models import MODEL_NAMES, Config, make_model


@dataclass
class RunResult:
    model: str
    scores: np.ndarray
    threshold: float
    metrics: dict
    fit_s: float
    infer_ms_per_1k: float
    loss_history: list


def split_points(n: int, cfg: Config) -> tuple[int, int]:
    train_end = int(n * cfg.train_frac)
    val_end = int(n * (cfg.train_frac + cfg.val_frac))
    return train_end, val_end


def pick_threshold(scores: np.ndarray, ts, windows, cfg: Config, quantile: float | None = None) -> float:
    """Eşiği doğrulama bölümünün skorlarından seçer. Test bölümüne asla bakmaz.

    val_normal modunda doğrulama bölümündeki anomali pencereleri içindeki noktalar dışarıda bırakılır;
    böylece doğrulamaya düşen bir anomali eşiği şişirip modeli sağırlaştırmaz. Dışarıda bırakılınca
    50'den az nokta kalırsa val_all'a geri dönülür.
    """
    q = cfg.quantile if quantile is None else quantile
    train_end, val_end = split_points(len(scores), cfg)
    v = np.asarray(scores[train_end:val_end], dtype=float)
    if cfg.threshold_source == "val_normal":
        normal = point_labels(ts, windows)[train_end:val_end] == 0
        if normal.sum() >= 50:
            v = v[normal]
    return float(np.quantile(v, q))


def fit_models(values: np.ndarray, cfg: Config, models=None) -> dict:
    """Her modeli yalnızca eğitim bölümüyle eğitir. {isim: (model, eğitim_süresi_s)}."""
    train_end, _ = split_points(len(values), cfg)
    fitted = {}
    for name in models or MODEL_NAMES:
        det = make_model(name, cfg)
        t0 = time.perf_counter()
        det.fit(values[:train_end])
        fitted[name] = (det, time.perf_counter() - t0)
    return fitted


def evaluate_fitted(fitted: dict, values, ts, windows, cfg: Config, quantile: float | None = None) -> dict:
    """Eğitilmiş modelleri verilen değerler üzerinde skorlar, eşiği doğrulamadan seçer, test'te ölçer."""
    q = cfg.quantile if quantile is None else quantile
    train_end, val_end = split_points(len(values), cfg)
    out = {}
    for name, (det, fit_s) in fitted.items():
        t0 = time.perf_counter()
        scores = det.score(values)
        infer = time.perf_counter() - t0
        thr = pick_threshold(scores, ts, windows, cfg, q)
        m = evaluate(ts, scores, thr, windows, val_end, cfg.merge_gap)
        m["val_events"] = val_event_count(ts, windows, train_end, val_end)
        out[name] = RunResult(name, scores, thr, m, fit_s, infer * 1e6 / len(values),
                              list(getattr(det, "loss_history_", [])))
    return out


def run_models(series: Series, cfg: Config, models=None, values=None) -> dict:
    values = series.df["value"].to_numpy(float) if values is None else values
    fitted = fit_models(values, cfg, models)
    return evaluate_fitted(fitted, values, series.df["timestamp"], series.windows, cfg)


def result_row(r: RunResult, **extra) -> dict:
    return {**extra, "model": r.model, **r.metrics,
            "fit_s": r.fit_s, "inference_ms_per_1k_points": r.infer_ms_per_1k}


def benchmark(series_list, cfg: Config, models=None, progress=None) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(series_list):
        if progress:
            progress(i, len(series_list), s.name)
        for r in run_models(s, cfg, models).values():
            rows.append(result_row(r, series=s.name))
    return pd.DataFrame(rows)


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Model başına ortalama (NaN'lar atlanır). 'series_with_events': olay içeren test serisi sayısı."""
    num = ["precision", "recall", "f1", "pr_auc", "false_alarms_per_day", "false_alarm_point_rate",
           "event_recall", "mean_delay_min", "inference_ms_per_1k_points", "fit_s"]
    g = df.groupby("model", sort=False)
    out = g[num].mean()
    out.insert(0, "events_detected_total", g["events_detected"].sum())
    out.insert(0, "events_total", g["events"].sum())
    out.insert(0, "series_with_events", g["events"].apply(lambda x: int((x > 0).sum())))
    return out.reset_index()


# ---------------------------------------------------------------- belirsizlik: tohum tekrarı ve güven aralığı

def seed_benchmark(series_list, cfg: Config, models=None, seeds=(0, 1, 2), progress=None) -> pd.DataFrame:
    """Benchmark'ı farklı rastgele tohumlarla tekrarlar (Isolation Forest ve Autoencoder tohuma bağlıdır,
    Rolling Z-Score deterministiktir). Her satıra 'seed' sütunu eklenir."""
    from dataclasses import replace
    parts = []
    for sd in seeds:
        df = benchmark(series_list, replace(cfg, seed=int(sd)), models,
                       progress=(lambda i, n, name, sd=sd: progress(sd, i, n, name)) if progress else None)
        df.insert(0, "seed", int(sd))
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Her tohum için model başına özet (havuzlanmış olay yakalama + seri ortalamaları), sonra tohumlar arası
    ortalama ve standart sapma."""
    rows = []
    for (sd, m), g in df.groupby(["seed", "model"], sort=False):
        ev = g["events"].sum()
        rows.append({"seed": sd, "model": m,
                     "event_recall_pooled": g["events_detected"].sum() / ev if ev else float("nan"),
                     "f1": g["f1"].mean(), "pr_auc": g["pr_auc"].mean(),
                     "false_alarms_per_day": g["false_alarms_per_day"].mean(),
                     "false_alarm_point_rate": g["false_alarm_point_rate"].mean()})
    per = pd.DataFrame(rows)
    cols = [c for c in per.columns if c not in ("seed", "model")]
    g = per.groupby("model", sort=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    out.insert(0, "n_seeds", per.groupby("model", sort=False)["seed"].nunique())
    return out.reset_index()


def _pooled_stats(g: pd.DataFrame) -> dict:
    ev = g["events"].sum()
    return {"event_recall_pooled": g["events_detected"].sum() / ev if ev else float("nan"),
            "f1": g["f1"].mean(), "pr_auc": g["pr_auc"].mean(),
            "false_alarms_per_day": g["false_alarms_per_day"].mean()}


def bootstrap_ci(bench: pd.DataFrame, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Seri bazında bootstrap: seriler yerine koyarak yeniden örneklenir (seri bağımsız birim sayılır).
    Dönüş: (model başına %95 güven aralığı, model çiftleri için fark aralığı).
    Fark aralığı 0'ı içeriyorsa iki model arasındaki farkın bu veriyle ayırt edilemediği söylenir.
    Olay sayısı az olduğu için aralıklar geniştir; bu bilerek gösterilir."""
    rng = np.random.default_rng(seed)
    names = list(dict.fromkeys(bench["model"]))
    sers = list(dict.fromkeys(bench["series"]))
    by = {k: g.set_index("series") for k, g in bench.groupby("model", sort=False)}
    metrics = ["event_recall_pooled", "f1", "pr_auc", "false_alarms_per_day"]
    draws = {m: {k: [] for k in metrics} for m in names}
    for _ in range(n_boot):
        pick = rng.choice(sers, size=len(sers), replace=True)
        for m in names:
            st = _pooled_stats(by[m].loc[pick].reset_index())
            for k in metrics:
                draws[m][k].append(st[k])
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    rows = []
    for m in names:
        pt = _pooled_stats(by[m].reset_index())
        for k in metrics:
            d = np.array(draws[m][k], dtype=float)
            rows.append({"model": m, "metric": k, "value": pt[k],
                         "ci_low": float(np.nanpercentile(d, lo)), "ci_high": float(np.nanpercentile(d, hi))})
    ci = pd.DataFrame(rows)
    drows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for k in metrics:
                d = np.array(draws[a][k], dtype=float) - np.array(draws[b][k], dtype=float)
                pt = _pooled_stats(by[a].reset_index())[k] - _pooled_stats(by[b].reset_index())[k]
                l, h = float(np.nanpercentile(d, lo)), float(np.nanpercentile(d, hi))
                drows.append({"model_a": a, "model_b": b, "metric": k, "difference_a_minus_b": pt,
                              "ci_low": l, "ci_high": h, "distinguishable": bool(l > 0 or h < 0)})
    return ci, pd.DataFrame(drows)


# ---------------------------------------------------------------- dayanıklılık deneyleri

def noise_experiment(series: Series, cfg: Config, levels=(0.0, 0.1, 0.25, 0.5),
                     repeats: int = 3, models=None) -> pd.DataFrame:
    """Tüm seriye Gaussian gürültü ekler (std = seviye × eğitim std'si). Etiket pencereleri değişmez.

    Gürültü eğitim bölümüne de girdiği için modeller her seviyede yeniden eğitilir.
    """
    v0 = series.df["value"].to_numpy(float)
    train_end, _ = split_points(len(v0), cfg)
    sd = float(v0[:train_end].std())
    rows = []
    for level in levels:
        for rep in range(1 if level == 0 else repeats):
            rng = np.random.default_rng(cfg.seed + 1000 * rep + int(level * 1000))
            noisy = v0 + rng.normal(0.0, level * sd, len(v0))
            for r in run_models(series, cfg, models, values=noisy).values():
                rows.append(result_row(r, series=series.name, noise_level=level, repeat=rep))
    return pd.DataFrame(rows)


def drift_experiment(series: Series, cfg: Config, levels=(0.0, 0.5, 1.0, 1.5),
                     kind: str = "step", models=None) -> pd.DataFrame:
    """Test bölümünün bir noktasından itibaren normal çalışma seviyesini kaydırır (veri kayması).

    Etiketler zaman damgasıyla tanımlandığı için korunur: kayma sadece değerleri değiştirir,
    anomali pencereleri aynı kalır. Pencere DIŞINDAKİ her alarm yanlış alarm sayılır;
    yani kayma yüzünden çıkan alarmlar doğrudan yanlış alarm olarak ölçülür.
    Modeller temiz eğitim verisiyle bir kez eğitilir (kayma eğitimden sonra başlar).
    """
    ts = series.df["timestamp"]
    v0 = series.df["value"].to_numpy(float)
    n = len(v0)
    train_end, val_end = split_points(n, cfg)
    sd = float(v0[:train_end].std())
    drift_start = val_end + int(0.25 * (n - val_end))
    fitted = fit_models(v0, cfg, models)
    rows = []
    for level in levels:
        v = v0.copy()
        if kind == "ramp":
            v[drift_start:] += level * sd * np.linspace(0.0, 1.0, n - drift_start)
        else:
            v[drift_start:] += level * sd
        for r in evaluate_fitted(fitted, v, ts, series.windows, cfg).values():
            rows.append(result_row(r, series=series.name, drift_level=level, drift_kind=kind,
                                   drift_start=str(ts.iloc[drift_start])))
    return pd.DataFrame(rows)


def _free_onsets(labels: np.ndarray, start: int, n: int, duration: int, margin: int, count: int) -> list[int]:
    """Test bölümünde, gerçek anomali pencerelerine değmeyen eşit aralıklı başlangıç noktaları bulur."""
    onsets = []
    for frac in np.linspace(0.15, 0.85, count):
        cand = start + int(frac * (n - start))
        for _ in range(n):
            lo, hi = max(0, cand - margin), min(n, cand + duration + margin)
            if cand + duration + margin < n and labels[lo:hi].sum() == 0 and \
                    all(abs(cand - o) > duration + 2 * margin for o in onsets):
                onsets.append(cand)
                break
            cand += max(1, duration // 2)
            if cand + duration + margin >= n:
                break
    return onsets


def delay_experiment(series: Series, cfg: Config, magnitudes=(2, 3, 5, 8), duration: int = 24,
                     n_inject: int = 3, models=None) -> pd.DataFrame:
    """Bilinen başlangıç anına sahip yapay anomaliler enjekte eder ve tespit gecikmesini ölçer.

    Test bölümünde gerçek anomali pencerelerinden uzak yerlere, `duration` adım süren bir seviye
    sıçraması (büyüklük × eğitim std'si) eklenir. Başlangıç anı kesin bilindiği için gecikme
    gerçek anlamda ölçülür (NAB pencerelerinde bu mümkün değildir).
    """
    ts = series.df["timestamp"]
    v0 = series.df["value"].to_numpy(float)
    n = len(v0)
    train_end, val_end = split_points(n, cfg)
    sd = float(v0[:train_end].std())
    labels = point_labels(ts, series.windows)
    onsets = _free_onsets(labels, val_end, n, duration, margin=cfg.window, count=n_inject)
    if not onsets:
        return pd.DataFrame()
    fitted = fit_models(v0, cfg, models)
    rows = []
    for mag in magnitudes:
        v = v0.copy()
        for o in onsets:
            v[o:o + duration] += mag * sd
        for name, (det, fit_s) in fitted.items():
            scores = det.score(v)
            thr = pick_threshold(scores, ts, series.windows, cfg)
            pred = scores > thr
            delays_min, delays_steps = [], []
            for o in onsets:
                hit = np.flatnonzero(pred[o:o + duration])
                if hit.size:
                    delays_steps.append(int(hit[0]))
                    delays_min.append((ts.iloc[o + hit[0]] - ts.iloc[o]).total_seconds() / 60)
            rows.append({
                "series": series.name, "model": name, "magnitude_sd": mag, "duration_steps": duration,
                "injected": len(onsets), "detected": len(delays_steps),
                "detection_rate": len(delays_steps) / len(onsets),
                "mean_delay_steps": float(np.mean(delays_steps)) if delays_steps else float("nan"),
                "mean_delay_min": float(np.mean(delays_min)) if delays_min else float("nan"),
            })
    return pd.DataFrame(rows)


def threshold_sensitivity(results: dict, series: Series, cfg: Config,
                          quantiles=(0.90, 0.95, 0.98, 0.99, 0.995, 0.999)) -> pd.DataFrame:
    """Aynı skorlar üzerinde eşik yüzdeliğini değiştirir; Precision/Recall/Yanlış alarm nasıl değişiyor?"""
    ts = series.df["timestamp"]
    n = len(ts)
    train_end, val_end = split_points(n, cfg)
    rows = []
    for name, r in results.items():
        for q in quantiles:
            thr = pick_threshold(r.scores, ts, series.windows, cfg, q)
            m = evaluate(ts, r.scores, thr, series.windows, val_end, cfg.merge_gap)
            m["val_events"] = val_event_count(ts, series.windows, train_end, val_end)
            rows.append({"series": series.name, "model": name, "quantile": q, **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- kayıt

def save_results(df: pd.DataFrame, name: str, cfg: Config, results_dir: Path) -> Path:
    """Sonuçları CSV'ye yazar; her satıra çalıştırma zamanı ve ayarlar eklenir (tekrarlanabilirlik)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["run_at"] = datetime.now().isoformat(timespec="seconds")
    out["config"] = json.dumps(asdict(cfg))
    path = results_dir / f"{name}.csv"
    out.to_csv(path, index=False)
    return path
