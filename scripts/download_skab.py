"""SKAB (Skoltech Anomaly Benchmark) dosyalarını data/skab klasörüne indirir.

Kullanım:
    python scripts/download_skab.py              # valve1 (16 dosya, 0-15) + valve2 (4 dosya, 0-3)
    python scripts/download_skab.py --other      # ek olarak 'other' klasörü (sızıntı, rotor dengesizliği vb.)

Kaynak: https://github.com/waico/SKAB  (GPL-3.0). Veri repoya KONMAZ (.gitignore); her kullanıcı kendisi indirir.
İndirilemeyen dosya programı durdurmaz, yazılır.
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "skab"
BASE = "https://raw.githubusercontent.com/waico/SKAB/master/data/"
# SKAB dosyaları 0'dan numaralanır (valve1: 0-15, valve2: 0-3). Asıl liste GitHub'dan sorulur; sorulamazsa bu yedek kullanılır.
FALLBACK = {"valve1": [f"{i}.csv" for i in range(0, 16)], "valve2": [f"{i}.csv" for i in range(0, 4)], "other": []}
API = "https://api.github.com/repos/waico/SKAB/contents/data/"


def list_folder(folder):
    """Klasördeki .csv adlarını GitHub API'den okur; olmazsa yedek listeyi verir."""
    import json
    try:
        req = urllib.request.Request(API + folder, headers={"User-Agent": "alertscore-downloader"})
        with urllib.request.urlopen(req, timeout=60) as r:
            names = [x["name"] for x in json.load(r) if x["name"].endswith(".csv")]
        if names:
            return sorted(names, key=lambda n: int(n.split(".")[0]) if n.split(".")[0].isdigit() else 999)
    except Exception as exc:  # noqa: BLE001
        print(f"UYARI: {folder} listesi alinamadi ({exc}); yedek liste kullaniliyor.")
    return FALLBACK[folder]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", action="store_true")
    args = ap.parse_args()
    folders = ["valve1", "valve2"] + (["other"] if args.other else [])
    ok = fail = 0
    for folder in folders:
        for name in list_folder(folder):
            rel = f"{folder}/{name}"
            dest = DEST / rel
            if dest.exists():
                ok += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with urllib.request.urlopen(BASE + rel, timeout=60) as r:
                    dest.write_bytes(r.read())
                ok += 1
                print("indirildi:", rel)
            except (urllib.error.URLError, OSError) as exc:
                fail += 1
                print(f"UYARI: {rel} indirilemedi ({exc})")
    print(f"Tamam: {ok} dosya hazir, {fail} indirilemedi. Lisans: GPL-3.0 (github.com/waico/SKAB), repoya eklemeyin.")


if __name__ == "__main__":
    main()
