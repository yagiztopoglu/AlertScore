"""Tüm deneyleri çalıştırıp sonuçları results/ klasörüne CSV olarak yazar.

Kullanım (proje klasöründen, sanal ortam aktifken):
    python scripts/run_experiments.py                 # 3 model, tüm NAB serileri
    python scripts/run_experiments.py --quick         # daha az tekrar/epoch (hızlı deneme)
    python scripts/run_experiments.py --models "Rolling Z-Score,Isolation Forest"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from alertscore.data import SKAB_DEFAULT_SENSOR, SKAB_SENSORS, list_nab_series, list_skab_files, load_nab, load_skab  # noqa: E402
from alertscore.experiments import (  # noqa: E402
    benchmark, delay_experiment, drift_experiment, noise_experiment, run_models, seed_benchmark,
    save_results, summarize_benchmark, threshold_sensitivity,
)
from alertscore.models import MODEL_NAMES, SKAB_CONFIG, Config  # noqa: E402
from alertscore.report import build_xlsx  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_NAMES), help="virgülle ayrılmış model adları")
    ap.add_argument("--quick", action="store_true", help="hızlı mod: 1 tekrar, 10 epoch")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--threshold-source", choices=["val_all", "val_normal"], default="val_all",
                    help="val_all: doğrulamadaki tüm noktalar; val_normal: doğrulamada yalnız anomali penceresi dışı noktalar")
    ap.add_argument("--seeds", type=int, default=None,
                    help="benchmark'ı kaç farklı rastgele tohumla tekrarlayacağı (varsayılan 5, --quick'te 1)")
    ap.add_argument("--dataset", choices=["nab", "skab"], default="nab",
                    help="nab (varsayılan) veya skab (data/skab içine scripts/download_skab.py ile indirilir)")
    ap.add_argument("--sensor", default=SKAB_DEFAULT_SENSOR, choices=SKAB_SENSORS, help="yalnız --dataset skab: kullanılacak sensör")
    ap.add_argument("--skab-dir", default=None, help="yalnız --dataset skab: SKAB klasörü (varsayılan data/skab)")
    args = ap.parse_args()
    if args.out == str(ROOT / "results") and args.dataset == "skab":
        args.out = str(ROOT / "results" / "skab")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.dataset == "skab":
        cfg = Config(**{**SKAB_CONFIG, "threshold_source": args.threshold_source})
        if args.quick:
            cfg = Config(**{**vars(cfg), "ae_epochs": 20})
    else:
        cfg = Config(ae_epochs=10 if args.quick else 30, threshold_source=args.threshold_source)
    repeats = 1 if args.quick else 3

    if args.dataset == "skab":
        keys = list_skab_files(args.skab_dir)
        if not keys:
            sys.exit("data/skab içinde dosya yok. Önce download_skab.bat çalıştırın.")
        series = [load_skab(k, args.sensor, args.skab_dir) for k in keys]
        print(f"SKAB, sensör: {args.sensor}, ayarlar: {SKAB_CONFIG}")
    else:
        keys = list_nab_series()
        if not keys:
            sys.exit("data/nab içinde etiketli seri bulunamadı. Önce scripts/download_nab.py çalıştırın.")
        series = [load_nab(k) for k in keys]
    t0 = time.time()
    print(f"{len(series)} seri, modeller: {models}\n")

    # 1) Benchmark (gerçek NAB verisi, değiştirilmemiş)
    bench = benchmark(series, cfg, models, progress=lambda i, n, s: print(f"[benchmark {i + 1}/{n}] {s}"))
    save_results(bench, "benchmark", cfg, args.out)
    print("\n=== Benchmark özeti (model başına ortalama) ===")
    print(summarize_benchmark(bench).round(3).to_string(index=False))

    # 1b) Tohum tekrarı: sonuçların rastgeleliğe ne kadar duyarlı olduğunu gösterir
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    if n_seeds > 1:
        sb = seed_benchmark(series, cfg, models, seeds=range(n_seeds),
                            progress=lambda sd, i, n, s: print(f"[tohum {sd} {i + 1}/{n}] {s}"))
        save_results(sb, "benchmark_seeds", cfg, args.out)

    # 2) Eşik duyarlılığı (benchmark ile aynı skorlar üzerinde)
    thr_rows = []
    for s in series:
        thr_rows.append(threshold_sensitivity(run_models(s, cfg, models), s, cfg))
    thr = pd.concat(thr_rows, ignore_index=True)
    save_results(thr, "threshold_sensitivity", cfg, args.out)

    # 3) Gürültü, 4) Veri kayması, 5) Tespit gecikmesi (gerçek seriler üzerine kontrollü enjeksiyon)
    noise, drift, delay = [], [], []
    for s in series:
        print(f"[robustness] {s.name}")
        noise.append(noise_experiment(s, cfg, repeats=repeats, models=models))
        drift.append(drift_experiment(s, cfg, models=models))
        d = delay_experiment(s, cfg, models=models)
        if len(d):
            delay.append(d)
    for name, parts in (("noise", noise), ("drift", drift), ("delay", delay)):
        if parts:
            save_results(pd.concat(parts, ignore_index=True), name, cfg, args.out)

    xlsx = build_xlsx(args.out)
    if xlsx:
        (Path(args.out) / "AlertScore_Sonuclar.xlsx").write_bytes(xlsx)
    print(f"\nBitti ({time.time() - t0:.0f} sn).")
    print(f"Excel raporu (okumak için bunu açın): {Path(args.out) / 'AlertScore_Sonuclar.xlsx'}")
    print(f"Ham CSV dosyaları (başka programlar için): {args.out}")


if __name__ == "__main__":
    main()
