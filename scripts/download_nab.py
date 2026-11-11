"""NAB'dan ek seriler indirir (isteğe bağlı; projede 4 seri zaten hazır geliyor).

Kullanım:
    python scripts/download_nab.py            # önerilen ek seriler
    python scripts/download_nab.py --all      # etiketi olan TÜM NAB serileri (yaklaşık 58 dosya)

Kaynak: https://github.com/numenta/NAB  (MIT lisansı, Copyright 2014-2024 Numenta Inc.)
İnternet erişimi gerektirir. Hata olursa program durmaz, indirilemeyen dosyayı yazar.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAB_DIR = ROOT / "data" / "nab"
BASE = "https://raw.githubusercontent.com/numenta/NAB/master/"

EXTRA = [
    "realKnownCause/ec2_request_latency_system_failure.csv",
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
    "realKnownCause/rogue_agent_key_hold.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
    "realTraffic/speed_7578.csv",
    "artificialWithAnomaly/art_daily_jumpsup.csv",
]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="etiketli tüm serileri indir")
    args = ap.parse_args()

    for rel in ("labels/combined_windows.json", "LICENSE.txt"):
        dest = NAB_DIR / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(fetch(BASE + rel))
            print("indirildi:", rel)
        except (urllib.error.URLError, OSError) as exc:
            print(f"UYARI: {rel} indirilemedi ({exc})")

    labels_path = NAB_DIR / "labels" / "combined_windows.json"
    if not labels_path.exists():
        sys.exit("combined_windows.json yok; indirme yapılamaz.")
    keys = json.loads(labels_path.read_text(encoding="utf-8"))
    wanted = [k for k, v in keys.items() if v] if args.all else [k for k in EXTRA if k in keys]

    ok = skipped = failed = 0
    for k in wanted:
        dest = NAB_DIR / k
        if dest.exists():
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(fetch(BASE + "data/" + k))
            ok += 1
            print("indirildi:", k)
        except (urllib.error.URLError, OSError) as exc:
            failed += 1
            print(f"UYARI: {k} indirilemedi ({exc})")
    print(f"\nİndirilen: {ok}, zaten vardı: {skipped}, başarısız: {failed}")


if __name__ == "__main__":
    main()
