"""Üç anomali tespit algoritması ve ortak arayüz.

Ortak arayüz (BaseDetector):
    fit(train_values)  -> modeli SADECE eğitim bölümüyle eğitir / parametreleri öğrenir
    score(values)      -> tüm seri için anomali skoru döndürür (yüksek = daha anormal)

Önemli kural (nedensellik): score(values)[t] yalnızca values[:t+1] bilgisini kullanır.
Yani gelecek veri geçmişin skoruna sızmaz. Yeni bir algoritma eklemek için
BaseDetector'dan türetip MODEL_REGISTRY'ye eklemek yeterli.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Config:
    """Bütün deneylerde ortak kullanılan ayarlar (adil karşılaştırma için tek yerde)."""
    train_frac: float = 0.4      # ilk %40: model eğitimi
    val_frac: float = 0.2        # sonraki %20: eşik seçimi (etiket kullanılmaz)
    quantile: float = 0.995      # eşik = doğrulama skorlarının bu yüzdeliği
    # "val_all": doğrulamadaki tüm noktalar (etiket kullanılmaz, varsayılan)
    # "val_normal": doğrulamadaki anomali penceresi DIŞINDAKİ noktalar (doğrulama etiketleri kullanılır;
    #               test etiketleri yine hiç kullanılmaz)
    threshold_source: str = "val_all"
    window: int = 16             # IF ve Autoencoder için pencere uzunluğu (adım)
    z_window: int = 100          # Rolling Z-Score için kayan pencere (adım)
    if_trees: int = 100
    ae_epochs: int = 30
    ae_hidden: int = 12
    ae_latent: int = 4
    merge_gap: int = 3           # bu kadar adımdan yakın alarmlar tek olay sayılır
    seed: int = 0


def make_windows(x: np.ndarray, w: int) -> np.ndarray:
    """Her t için son w değeri (x[t-w+1..t]) döndürür; baştaki eksikler ilk değerle doldurulur.

    Sadece geçmişe baktığı için nedenseldir. Şekil: (len(x), w).
    """
    x = np.asarray(x, dtype=float)
    pad = np.concatenate([np.full(w - 1, x[0]), x])
    return sliding_window_view(pad, w)


def _fit_scaler(train: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(train))
    sd = float(np.std(train))
    return mu, (sd if sd > 1e-12 else 1.0)


class BaseDetector:
    name = "base"

    def fit(self, train: np.ndarray) -> "BaseDetector":
        raise NotImplementedError

    def score(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RollingZScore(BaseDetector):
    """Baseline: nokta, önceki `window` gözlemin ortalamasından kaç standart sapma uzakta?"""
    name = "Rolling Z-Score"

    def __init__(self, window: int = 100):
        self.window = window
        self.std_floor_ = 1e-8

    def fit(self, train):
        # Öğrenilen tek şey std tabanı: pencere sabit çizgiye denk gelirse 0'a bölmeyi önler.
        sd = float(np.std(train))
        self.std_floor_ = max(0.01 * sd, 1e-8)
        return self

    def score(self, values):
        s = pd.Series(np.asarray(values, dtype=float))
        minp = max(2, self.window // 2)
        # shift(1): t anındaki istatistik yalnızca t-1 ve öncesinden hesaplanır (sızıntı yok)
        mean = s.rolling(self.window, min_periods=minp).mean().shift(1)
        std = s.rolling(self.window, min_periods=minp).std(ddof=0).shift(1)
        z = (s - mean).abs() / std.clip(lower=self.std_floor_)
        return np.nan_to_num(z.to_numpy(), nan=0.0, posinf=0.0)


class IsolationForestDetector(BaseDetector):
    """Pencere özellikleri üzerinde Isolation Forest (scikit-learn)."""
    name = "Isolation Forest"

    def __init__(self, window: int = 16, n_trees: int = 100, seed: int = 0):
        self.window, self.n_trees, self.seed = window, n_trees, seed

    def _features(self, values):
        z = (np.asarray(values, dtype=float) - self.mu_) / self.sd_
        w = make_windows(z, self.window)
        return np.column_stack([w, w.mean(axis=1), w.std(axis=1)])

    def fit(self, train):
        from sklearn.ensemble import IsolationForest
        self.mu_, self.sd_ = _fit_scaler(train)
        self.model_ = IsolationForest(n_estimators=self.n_trees, random_state=self.seed, n_jobs=1)
        self.model_.fit(self._features(train))
        return self

    def score(self, values):
        return -self.model_.score_samples(self._features(values))


class AutoencoderDetector(BaseDetector):
    """Küçük PyTorch Autoencoder: normal pencereleri sıkıştırıp yeniden kurar.

    Anomali skoru = yeniden oluşturma hatası (reconstruction error, pencere başına MSE).
    Skor pencerenin son noktasına yazılır (nedensel).
    """
    name = "Autoencoder"

    def __init__(self, window: int = 16, hidden: int = 12, latent: int = 4,
                 epochs: int = 30, batch_size: int = 128, lr: float = 1e-3, seed: int = 0):
        self.window, self.hidden, self.latent = window, hidden, latent
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed
        self.loss_history_: list[float] = []

    def _build(self):
        from torch import nn
        w, h, k = self.window, self.hidden, self.latent
        self.encoder_ = nn.Sequential(nn.Linear(w, h), nn.ReLU(), nn.Linear(h, k))   # sıkıştır
        self.decoder_ = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, w))   # yeniden kur
        self.net_ = nn.Sequential(self.encoder_, self.decoder_)

    def fit(self, train):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.mu_, self.sd_ = _fit_scaler(train)
        z = (np.asarray(train, dtype=float) - self.mu_) / self.sd_
        X = torch.tensor(np.ascontiguousarray(make_windows(z, self.window)), dtype=torch.float32)
        self._build()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        self.loss_history_ = []
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=gen)
            total = 0.0
            for i in range(0, len(X), self.batch_size):
                batch = X[perm[i:i + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.net_(batch), batch)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(batch)
            self.loss_history_.append(total / len(X))
        self.n_params_ = int(sum(p.numel() for p in self.net_.parameters()))
        return self

    def score(self, values):
        import torch
        z = (np.asarray(values, dtype=float) - self.mu_) / self.sd_
        X = torch.tensor(np.ascontiguousarray(make_windows(z, self.window)), dtype=torch.float32)
        self.net_.eval()
        with torch.no_grad():
            err = ((self.net_(X) - X) ** 2).mean(dim=1)
        return err.numpy().astype(float)


MODEL_NAMES = ["Rolling Z-Score", "Isolation Forest", "Autoencoder"]


def make_model(name: str, cfg: Config) -> BaseDetector:
    if name == "Rolling Z-Score":
        return RollingZScore(window=cfg.z_window)
    if name == "Isolation Forest":
        return IsolationForestDetector(window=cfg.window, n_trees=cfg.if_trees, seed=cfg.seed)
    if name == "Autoencoder":
        return AutoencoderDetector(window=cfg.window, hidden=cfg.ae_hidden, latent=cfg.ae_latent,
                                   epochs=cfg.ae_epochs, seed=cfg.seed)
    raise ValueError(f"Bilinmeyen model: {name}")


# SKAB: her dosya ~1100 saniyelik tek deney; anomali dosyanın ~%50'sinde başlar. Bu yüzden eğitim %30, doğrulama %15
# (ikisi de anomalisiz kısımda), test son %55. Eğitim kısa olduğu için Autoencoder 100 epoch eğitilir.
SKAB_CONFIG = dict(train_frac=0.3, val_frac=0.15, ae_epochs=100)
