"""Sütun adları, açıklamalar ve düz Türkçe sonuç yorumları.

Arayüz ve Excel/PDF raporu aynı sözlüğü kullanır. Yorum cümleleri yalnızca gerçek sayılardan üretilir.
"""

from __future__ import annotations

import math

import pandas as pd

# anahtar: (Türkçe ad, ne demek?)
INFO: dict[str, tuple[str, str]] = {
    "series": ("Seri", "Hangi zaman serisi (NAB dosyası)."),
    "model": ("Model", "Anomali tespit algoritması."),
    "threshold": ("Eşik", "Anomali skoru bu değeri aşınca alarm verilir. Her modelin skor ölçeği farklıdır, bu yüzden eşikler modeller arasında kıyaslanmaz."),
    "test_points": ("Test nokta sayısı", "Ölçümün yapıldığı test bölümündeki veri noktası sayısı."),
    "test_days": ("Test süresi (gün)", "Test bölümünün uzunluğu."),
    "precision": ("Kesinlik (Precision)", "Verilen alarmların yüzde kaçı gerçekten anomali penceresinin içinde (0-1, yüksek iyi). Hiç alarm yoksa hesaplanamaz."),
    "recall": ("Nokta yakalama (Recall)", "Anomali penceresindeki noktaların yüzde kaçında alarm var (0-1). NAB pencereleri geniş olduğu için düşük çıkması normaldir; asıl bakılacak ölçü 'Olay yakalama oranı'."),
    "f1": ("F1", "Kesinlik ve nokta yakalamanın dengesi (0-1, yüksek iyi)."),
    "pr_auc": ("PR-AUC", "Eşikten bağımsız sıralama kalitesi (0-1, yüksek iyi). Anomali azken Accuracy yerine bu kullanılır."),
    "alert_episodes": ("Alarm olayı sayısı", "Birbirine yakın alarmlar tek olay sayılarak toplam alarm olayı."),
    "false_alarms": ("Yanlış alarm olayı", "Hiçbir anomali penceresine değmeyen alarm olayı sayısı (düşük iyi)."),
    "false_alarms_per_day": ("Yanlış alarm / gün", "Yanlış alarm olayı sayısı bölü test günü (düşük iyi)."),
    "false_alarm_point_rate": ("Yanlış alarm nokta oranı", "Normal noktaların yüzde kaçı alarm verdi (0-1, düşük iyi). Alarm sürekli açık kalırsa 'yanlış alarm olayı' düşük görünür, bu ölçü o durumu yakalar."),
    "events": ("Anomali olayı sayısı", "Test bölümüne giren etiketli anomali penceresi sayısı."),
    "events_detected": ("Yakalanan olay", "En az bir alarm verilen anomali olayı sayısı."),
    "event_recall": ("Olay yakalama oranı", "Anomali olaylarının yüzde kaçı yakalandı (0-1, yüksek iyi). En anlamlı doğruluk ölçüsü; olay sayısı azsa dikkatli yorumlayın."),
    "mean_delay_min": ("Ort. gecikme (dk)", "Anomali penceresinin başlangıcından ilk alarma geçen ortalama süre (düşük iyi). NAB penceresi gerçek başlangıç anı değildir; kesin gecikme için 'Tespit gecikmesi' deneyine bakın."),
    "fit_s": ("Eğitim süresi (sn)", "Modelin eğitilme süresi."),
    "inference_ms_per_1k_points": ("Skorlama süresi (ms / 1000 nokta)", "1000 noktayı skorlamak için gereken süre (düşük = hızlı)."),
    "val_events": ("Doğrulamadaki anomali olayı", "Eşiğin seçildiği doğrulama bölümüne giren anomali penceresi sayısı. 0'dan büyükse eşik yüksek çıkmış, yani model daha az alarm vermiş olabilir."),
    "events_total": ("Toplam anomali olayı", "Tüm serilerin test bölümlerindeki anomali olaylarının toplamı."),
    "events_detected_total": ("Toplam yakalanan olay", "Toplam olaylardan en az bir alarm verilenlerin sayısı."),
    "series_with_events": ("Olay içeren seri sayısı", "Test bölümünde en az bir anomali olayı bulunan seri sayısı."),
    "noise_level": ("Gürültü seviyesi (× std)", "Eklenen gürültünün standart sapması; eğitim verisi standart sapmasının katı olarak (0 = temiz veri)."),
    "repeat": ("Tekrar no", "Aynı seviye için kaçıncı tekrar (farklı rastgele gürültü)."),
    "drift_level": ("Kayma (× std)", "Normal çalışma seviyesinin kaydırıldığı miktar; eğitim verisi standart sapmasının katı olarak (0 = kayma yok)."),
    "drift_kind": ("Kayma türü", "step = ani basamak, ramp = yavaş yavaş artan kayma."),
    "drift_start": ("Kayma başlangıcı", "Kaymanın başladığı zaman."),
    "magnitude_sd": ("Anomali büyüklüğü (× std)", "Eklenen yapay anomalinin büyüklüğü; eğitim verisi standart sapmasının katı olarak."),
    "duration_steps": ("Anomali süresi (adım)", "Eklenen yapay anomalinin kaç veri noktası sürdüğü."),
    "injected": ("Eklenen anomali", "Seriye eklenen yapay anomali sayısı."),
    "detected": ("Yakalanan", "Eklenen anomalilerden alarm verilenlerin sayısı."),
    "detection_rate": ("Yakalama oranı", "Eklenen anomalilerin yüzde kaçı yakalandı (0-1, yüksek iyi)."),
    "mean_delay_steps": ("Ort. gecikme (adım)", "Anomalinin gerçek başlangıcından ilk alarma kaç adım geçti (düşük iyi, 0 = anında)."),
    "quantile": ("Eşik yüzdeliği", "Eşik, doğrulama skorlarının bu yüzdeliğinden seçilir. Yükseldikçe eşik büyür, alarm azalır."),
}

HIDDEN = ("run_at", "config")

# Model Laboratory metrik tablosunun sırası
MODEL_METRIC_KEYS = [
    "events", "events_detected", "event_recall", "mean_delay_min",
    "false_alarms", "false_alarms_per_day", "false_alarm_point_rate",
    "precision", "recall", "f1", "pr_auc",
    "alert_episodes", "threshold", "test_points", "test_days", "val_events",
]


def name_of(key: str) -> str:
    return INFO.get(key, (key, ""))[0]


def pretty(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Teknik sütun adlarını okunur Türkçe adlara çevirir, sayıları yuvarlar, run_at/config'i gizler."""
    out = df.drop(columns=[c for c in HIDDEN if c in df.columns]).round(digits)
    return out.rename(columns={c: name_of(c) for c in out.columns})


def glossary() -> pd.DataFrame:
    return pd.DataFrame([(n, d) for n, d in INFO.values()], columns=["Ad", "Ne demek?"])


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "hesaplanamadı"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def metric_table(metrics: dict) -> pd.DataFrame:
    rows = [(name_of(k), _fmt(metrics.get(k)), INFO[k][1]) for k in MODEL_METRIC_KEYS if k in metrics]
    return pd.DataFrame(rows, columns=["Ölçü", "Değer", "Ne demek?"])


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def explain_result(model: str, m: dict, threshold_source: str = "val_all") -> list[str]:
    """Bir modelin test sonucunu düz Türkçe cümlelerle özetler (yalnızca gerçek sayılar kullanılır)."""
    out = []
    if m["events"] == 0:
        out.append("Test bölümünde etiketli anomali yok; olay yakalama ve gecikme hesaplanamaz.")
    else:
        out.append(f"Test bölümünde {m['events']} anomali olayı vardı; {model} bunların {m['events_detected']} "
                   f"tanesinde alarm verdi (olay yakalama oranı %{m['event_recall'] * 100:.0f}).")
    if m["alert_episodes"] == 0:
        out.append("Hiç alarm verilmedi: eşik, test bölümünde skorların ulaştığı değerlerden yüksek kalmış olabilir.")
    else:
        rate = m["false_alarm_point_rate"]
        out.append(f"{m['false_alarms']} yanlış alarm olayı oldu ({m['false_alarms_per_day']:.2f} / gün); "
                   f"normal noktaların %{rate * 100:.1f}'i alarm verdi.")
    if m.get("val_events", 0) > 0 and threshold_source == "val_all":
        out.append(f"Dikkat: eşiğin seçildiği doğrulama bölümünde {m['val_events']} anomali olayı var. "
                   "Bu, eşiği yükseltip alarmı azaltmış olabilir; kenar çubuğundaki 'Eşik kaynağı' ayarını "
                   "'yalnız normal noktalar' yaparak etkisini deneyebilirsiniz.")
    elif m.get("val_events", 0) > 0:
        out.append(f"Doğrulama bölümünde {m['val_events']} anomali olayı var; eşik yalnız normal noktalardan hesaplandı "
                   "(doğrulama etiketleri kullanıldı, test etiketleri kullanılmadı).")
    return out


def trend_sentences(df: pd.DataFrame, x: str, y: str, as_percent: bool = False) -> list[str]:
    """x'in en küçük ve en büyük değerinde, model başına y'nin ortalamasını karşılaştırır."""
    g = df.groupby(["model", x], sort=False)[y].mean().reset_index()
    lo, hi = g[x].min(), g[x].max()

    def f(v):
        if _nan(v):
            return "hesaplanamadı"
        return f"%{v * 100:.1f}" if as_percent else f"{v:.2f}"

    out = [f"{name_of(y)}: {name_of(x)} {lo:g} iken → {hi:g} iken"]
    for model, d in g.groupby("model", sort=False):
        a = d.loc[d[x] == lo, y].iloc[0]
        b = d.loc[d[x] == hi, y].iloc[0]
        out.append(f"{model}: {f(a)} → {f(b)}")
    return out
