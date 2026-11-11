"""Veri katmanı: CSV okuma, doğrulama, NAB etiketleri (anomali pencereleri).

NAB etiketleri satır başına 0/1 DEĞİL. `combined_windows.json` her seri için
"şu iki zaman arasında anomali var" diyen pencereler (başlangıç, bitiş) listesidir.
Nokta etiketlerini zaman damgalarından biz türetiriz (bkz. metrics.point_labels).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NAB_DIR = ROOT / "data" / "nab"
SKAB_DIR = ROOT / "data" / "skab"      # SKAB (GPL-3.0): repoya konmaz, kullanıcı indirir

MAX_ROWS = 500_000      # yüklenen CSV için üst sınır (güvenlik / bellek)
MAX_UPLOAD_MB = 20


class DataFormatError(ValueError):
    """CSV beklenen formatta değilse fırlatılır; mesajı kullanıcıya gösterilebilir."""


@dataclass
class Series:
    name: str
    df: pd.DataFrame                       # sütunlar: timestamp, value (sıralı, tekil)
    windows: list = field(default_factory=list)   # [(Timestamp, Timestamp), ...]
    quality: dict = field(default_factory=dict)   # veri kalitesi raporu
    source: str = "nab"                    # "nab" | "skab" | "upload"

    @property
    def has_labels(self) -> bool:
        return self.source in ("nab", "skab")


def clean_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """timestamp/value sütunlarını doğrular, temizler ve kalite raporu üretir.

    Politika (hepsi rapora yazılır, sessizce düzeltme yok):
    - Geçersiz zaman damgası veya sayı olmayan değer içeren satırlar atılır.
    - Zaman sırası bozuksa zamana göre sıralanır.
    - Aynı zaman damgası birden fazlaysa değerlerin ortalaması alınır.
    - Boşluklar (eksik zaman aralıkları) DOLDURULMAZ; yalnızca raporlanır,
      çünkü doldurmak olmayan veri uydurmak olur.
    """
    cols = {str(c).strip().lower(): c for c in raw.columns}
    if "timestamp" not in cols or "value" not in cols:
        raise DataFormatError(
            "CSV'de 'timestamp' ve 'value' sütunları olmalı. "
            f"Bulunan sütunlar: {list(raw.columns)}"
        )
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(raw[cols["timestamp"]], errors="coerce"),
        "value": pd.to_numeric(raw[cols["value"]], errors="coerce"),
    })
    n_raw = len(df)
    bad_ts = int(df["timestamp"].isna().sum())
    bad_val = int(df["value"].isna().sum())
    df = df.dropna()
    df = df[np.isfinite(df["value"])]
    if len(df) < 200:
        raise DataFormatError(f"Temizlikten sonra yalnızca {len(df)} geçerli satır kaldı (en az 200 gerekli).")

    backward = int((df["timestamp"].diff() < pd.Timedelta(0)).sum())
    n_dups = int(df["timestamp"].duplicated().sum())
    df = df.sort_values("timestamp", kind="stable")
    if n_dups:
        df = df.groupby("timestamp", as_index=False)["value"].mean()
    df = df.reset_index(drop=True)

    step = df["timestamp"].diff().dropna()
    med = step.median()
    gaps = step[step > 1.5 * med]
    quality = {
        "rows_raw": n_raw,
        "rows": len(df),
        "invalid_timestamps": bad_ts,
        "missing_or_non_numeric_values": bad_val,
        "backward_time_steps": backward,
        "duplicate_timestamps": n_dups,
        "median_step": str(med),
        "gaps": int(len(gaps)),
        "largest_gap": str(gaps.max()) if len(gaps) else "yok",
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
    }
    return df, quality


def load_uploaded_csv(buffer, name: str = "yüklenen.csv") -> Series:
    """Kullanıcının yüklediği CSV'yi doğrulayıp Series'e çevirir (etiketsiz)."""
    try:
        raw = pd.read_csv(buffer, nrows=MAX_ROWS + 1)
    except Exception as exc:  # pandas çok çeşitli hata fırlatabilir
        raise DataFormatError(f"CSV okunamadı: {exc}") from exc
    if len(raw) > MAX_ROWS:
        raise DataFormatError(f"CSV en fazla {MAX_ROWS:,} satır olabilir.")
    df, quality = clean_frame(raw)
    return Series(name=name, df=df, windows=[], quality=quality, source="upload")


def _labels_path() -> Path | None:
    for p in (NAB_DIR / "labels" / "combined_windows.json", NAB_DIR / "combined_windows.json"):
        if p.exists():
            return p
    return None


def load_windows() -> dict[str, list]:
    p = _labels_path()
    if p is None:
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {k: [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in v] for k, v in data.items()}


def list_nab_series() -> list[str]:
    """Diskte hem CSV'si hem etiketi bulunan NAB serilerinin anahtarları ('kategori/dosya.csv')."""
    labels = load_windows()
    return sorted(k for k in labels if (NAB_DIR / k).exists())


def load_nab(key: str) -> Series:
    labels = load_windows()
    if key not in labels:
        raise DataFormatError(f"'{key}' için etiket bulunamadı (combined_windows.json).")
    raw = pd.read_csv(NAB_DIR / key)
    df, quality = clean_frame(raw)
    return Series(name=key, df=df, windows=labels[key], quality=quality, source="nab")


# ---------------------------------------------------------------- SKAB (Skoltech Anomaly Benchmark)
# Kaynak: https://github.com/waico/SKAB (GPL-3.0). Her CSV ';' ayraçlı, 1 saniye aralıklı, çok sensörlü tek bir deneydir
# ve satır başına 0/1 'anomaly' etiketi vardır. Bu proje tek değişkenli olduğu için her dosyadan TEK sensör alınır.

SKAB_SENSORS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature",
                "Thermocouple", "Voltage", "Volume Flow RateRMS"]
SKAB_DEFAULT_SENSOR = "Volume Flow RateRMS"   # önceden seçildi: vana kapanınca akış doğrudan etkilenir
SKAB_FOLDERS = ("valve1", "valve2")


def _natural(key: str):
    folder, _, name = key.partition("/")
    stem = name.rsplit(".", 1)[0]
    return (folder, int(stem) if stem.isdigit() else 10**9, name)


def list_skab_files(skab_dir=None, folders=SKAB_FOLDERS) -> list[str]:
    """Diskteki SKAB dosyalarının anahtarları ('valve1/3.csv')."""
    base = Path(skab_dir) if skab_dir else SKAB_DIR
    keys = [f"{f}/{p.name}" for f in folders if (base / f).is_dir() for p in (base / f).glob("*.csv")]
    return sorted(keys, key=_natural)


def label_runs(ts: pd.Series, flags) -> list:
    """Satır başına 0/1 etiketten anomali pencerelerini çıkarır: ardışık 1'ler tek pencere (başlangıç, bitiş)."""
    f = np.asarray(flags).astype(int)
    out, start = [], None
    for i, v in enumerate(f):
        if v == 1 and start is None:
            start = i
        if v != 1 and start is not None:
            out.append((ts.iloc[start], ts.iloc[i - 1]))
            start = None
    if start is not None:
        out.append((ts.iloc[start], ts.iloc[len(f) - 1]))
    return out


def load_skab(key: str, sensor: str = SKAB_DEFAULT_SENSOR, skab_dir=None) -> Series:
    base = Path(skab_dir) if skab_dir else SKAB_DIR
    path = base / key
    if not path.exists():
        raise DataFormatError(f"SKAB dosyası yok: {path}. Önce scripts/download_skab.py çalıştırın.")
    raw = pd.read_csv(path, sep=";")
    missing = [c for c in ("datetime", sensor, "anomaly") if c not in raw.columns]
    if missing:
        raise DataFormatError(f"{key}: beklenen sütunlar yok {missing}. Bulunan: {list(raw.columns)}")
    ts = pd.to_datetime(raw["datetime"], errors="coerce")
    df, quality = clean_frame(pd.DataFrame({"timestamp": ts, "value": raw[sensor]}))
    # Etiketi temizlenmiş zaman damgalarına hizala (aynı zaman damgasında etiketlerin en büyüğü).
    lab = pd.Series(pd.to_numeric(raw["anomaly"], errors="coerce").fillna(0).values, index=ts).groupby(level=0).max()
    flags = lab.reindex(df["timestamp"]).fillna(0).values
    quality["sensor"] = sensor
    quality["label_points"] = int(flags.sum())
    return Series(name=f"skab/{key}", df=df, windows=label_runs(df["timestamp"], flags), quality=quality, source="skab")
