"""CSV ve PDF rapor üretimi (reportlab). PDF yalnızca tablo ve notlar içerir; grafik içermez."""

from __future__ import annotations

import io
import os
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]
_TR = str.maketrans("ğĞıİşŞçÇöÖüÜ", "gGiIsScCoOuU")


def _register_font() -> tuple[str, str, bool]:
    """Türkçe karakterleri destekleyen bir TTF bulursa onu kaydeder; yoksa Helvetica + ASCII'ye çevirme."""
    for regular, bold in _FONT_CANDIDATES:
        if os.path.exists(regular) and os.path.exists(bold):
            pdfmetrics.registerFont(TTFont("AS-Regular", regular))
            pdfmetrics.registerFont(TTFont("AS-Bold", bold))
            return "AS-Regular", "AS-Bold", True
    return "Helvetica", "Helvetica-Bold", False


# Deney adı -> (başlık, gruplama sütunları, özetlenen ölçüler)
EXPERIMENTS = {
    "noise": ("Gürültü deneyi", ["model", "noise_level"],
              ["f1", "event_recall", "false_alarm_point_rate", "false_alarms_per_day"]),
    "drift": ("Veri kayması deneyi", ["model", "drift_level"],
              ["false_alarm_point_rate", "false_alarms_per_day", "event_recall"]),
    "delay": ("Tespit gecikmesi deneyi", ["model", "magnitude_sd"],
              ["detection_rate", "mean_delay_min", "mean_delay_steps"]),
    "threshold_sensitivity": ("Eşik duyarlılığı deneyi", ["model", "quantile"],
                              ["precision", "recall", "false_alarm_point_rate", "false_alarms_per_day"]),
}
_SHEET = {"noise": "Gürültü", "drift": "Kayma", "delay": "Gecikme", "threshold_sensitivity": "Eşik"}


def read_results(results_dir) -> dict:
    """results/ klasöründeki kayıtlı CSV'leri {ad: DataFrame} olarak okur."""
    from pathlib import Path
    out = {}
    for name in ("benchmark", "benchmark_seeds", *EXPERIMENTS):
        p = Path(results_dir) / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def aggregated(name: str, df: pd.DataFrame) -> pd.DataFrame:
    _, by, cols = EXPERIMENTS[name]
    cols = [c for c in cols if c in df.columns]
    return df.groupby(by, sort=False)[cols].mean().reset_index()


_UNC_NAMES = {"event_recall_pooled": "Olay yakalama (havuzlanmış)", "f1": "F1", "pr_auc": "PR-AUC",
              "false_alarms_per_day": "Yanlış alarm / gün", "false_alarm_point_rate": "Yanlış alarm nokta oranı",
              "model": "Model", "metric": "Ölçüt", "value": "Değer", "ci_low": "%95 alt sınır", "ci_high": "%95 üst sınır",
              "model_a": "Model A", "model_b": "Model B", "difference_a_minus_b": "Fark (A − B)",
              "distinguishable": "Fark ayırt edilebilir mi?", "n_seeds": "Tohum sayısı", "seed": "Tohum"}


def uncertainty_sheets(res: dict) -> list:
    """Belirsizlik sayfaları: seri bazlı bootstrap güven aralığı ve (varsa) tohum tekrarı özeti."""
    from .experiments import bootstrap_ci, seed_summary

    out = []
    if "benchmark" in res and res["benchmark"]["series"].nunique() >= 3:
        ci, diff = bootstrap_ci(res["benchmark"])
        ci["metric"] = ci["metric"].map(lambda k: _UNC_NAMES.get(k, k))
        diff["metric"] = diff["metric"].map(lambda k: _UNC_NAMES.get(k, k))
        diff["distinguishable"] = diff["distinguishable"].map({True: "evet", False: "hayır"})
        out.append(("Güven aralığı", ci.rename(columns=_UNC_NAMES)))
        out.append(("Model farkları", diff.rename(columns=_UNC_NAMES)))
    if "benchmark_seeds" in res and res["benchmark_seeds"]["seed"].nunique() > 1:
        ss = seed_summary(res["benchmark_seeds"])
        ss.columns = [c if c in ("model", "n_seeds") else
                      _UNC_NAMES.get(c.rsplit("_", 1)[0], c) + (" (ort.)" if c.endswith("_mean") else " (std)")
                      for c in ss.columns]
        out.append(("Tohum tekrarı", ss.rename(columns=_UNC_NAMES)))
    return out


def build_xlsx(results_dir) -> bytes | None:
    """Bütün kayıtlı sonuçları biçimli, açıklamalı tek bir Excel dosyasına koyar. Sonuç yoksa None."""
    from .experiments import summarize_benchmark
    from .labels import glossary, pretty

    res = read_results(results_dir)
    if not res:
        return None
    sheets: list[tuple[str, pd.DataFrame]] = []
    first = next(iter(res.values()))
    info = pd.DataFrame({
        "Bilgi": ["Çalıştırma zamanı", "Ayarlar (config)", "Veri", "Bölme", "Eşik"],
        "Değer": [str(first["run_at"].iloc[0]) if "run_at" in first else "-",
                  str(first["config"].iloc[0]) if "config" in first else "-",
                  "Numenta Anomaly Benchmark (MIT lisansı, Copyright 2014-2024 Numenta Inc.)",
                  "İlk %40 eğitim, sonraki %20 doğrulama (eşik), son %40 test",
                  "Doğrulama skorlarının yüzdeliği; test verisi hiç kullanılmaz. 'Ayarlar' içindeki threshold_source: "
                  "val_all = doğrulamada etiket kullanılmaz, val_normal = doğrulamadaki anomali penceresi dışındaki noktalar kullanılır"],
    })
    sheets.append(("Okuma rehberi", glossary()))
    sheets.append(("Çalıştırma bilgisi", info))
    if "benchmark" in res:
        sheets.append(("Benchmark özet", pretty(summarize_benchmark(res["benchmark"]))))
        sheets.append(("Benchmark detay", pretty(res["benchmark"])))
    sheets.extend(uncertainty_sheets(res))
    for name in EXPERIMENTS:
        if name in res:
            sheets.append((f"{_SHEET[name]} özet", pretty(aggregated(name, res[name]))))
            sheets.append((f"{_SHEET[name]} detay", pretty(res[name])))

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        wb = xw.book
        head = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1, "text_wrap": True, "valign": "top"})
        num = wb.add_format({"num_format": "0.000"})
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        for title, df in sheets:
            df.to_excel(xw, sheet_name=title[:31], index=False)
            ws = xw.sheets[title[:31]]
            for j, col in enumerate(df.columns):
                ws.write(0, j, col, head)
                longest = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)])
                is_float = pd.api.types.is_float_dtype(df[col])
                is_text = not pd.api.types.is_numeric_dtype(df[col])
                width = min(max(12, longest + 2), 60 if is_text else 22)
                fmt = num if is_float else (wrap if is_text and width >= 60 else None)
                ws.set_column(j, j, width, fmt)
            ws.set_row(0, 32)
            ws.freeze_panes(1, 0)
    return buf.getvalue()


def build_pdf_from_results(results_dir) -> bytes | None:
    from .experiments import summarize_benchmark
    from .labels import pretty

    res = read_results(results_dir)
    if not res:
        return None
    sections = []
    if "benchmark" in res:
        sections.append(("Benchmark (model başına ortalama)", pretty(summarize_benchmark(res["benchmark"]))))
    for name, (title, _, _) in EXPERIMENTS.items():
        if name in res:
            sections.append((title, pretty(aggregated(name, res[name]))))
    notes = [
        "Bölme: ilk %40 eğitim, sonraki %20 doğrulama (eşik seçimi), son %40 test; rastgele karıştırma yok.",
        "Eşik: doğrulama skorlarının yüzdeliği; test verisi eşik seçiminde hiç kullanılmaz (kaynak ayarı için Excel'deki 'Çalıştırma bilgisi' sayfasına bakın).",
        "Veri: Numenta Anomaly Benchmark (MIT lisansı, Copyright 2014-2024 Numenta Inc.).",
        "Sonuçlar belirli seri, ayar ve tohum için tek koşudur; istatistiksel anlamlılık testi yapılmamıştır.",
        "Nokta bazlı Recall, NAB pencerelerinin genişliği nedeniyle düşük çıkar; olay bazlı ölçüler daha anlamlıdır.",
        "Sütunların anlamları için Excel raporundaki 'Okuma rehberi' sayfasına bakın.",
    ]
    return build_pdf("AlertScore Deney Raporu", sections, notes)


def _fmt(v) -> str:
    if isinstance(v, float):
        return "-" if v != v else f"{v:.3f}"
    return str(v)


def build_pdf(title: str, sections: list[tuple[str, pd.DataFrame]], notes: list[str]) -> bytes:
    font, bold, unicode_ok = _register_font()
    tr = (lambda s: s) if unicode_ok else (lambda s: str(s).translate(_TR))
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontName=bold)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=bold)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font, fontSize=9)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    story = [Paragraph(tr(title), h1),
             Paragraph(tr(f"Oluşturulma: {datetime.now():%Y-%m-%d %H:%M}"), body), Spacer(1, 8)]
    for heading, df in sections:
        story.append(Paragraph(tr(heading), h2))
        if df is None or df.empty:
            story += [Paragraph(tr("Veri yok."), body), Spacer(1, 6)]
            continue
        rows = [[tr(c) for c in df.columns]] + [[tr(_fmt(v)) for v in r] for r in df.itertuples(index=False)]
        width = doc.width / len(df.columns)
        t = Table(rows, colWidths=[width] * len(df.columns), repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTSIZE", (0, 0), (-1, -1), 7), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story += [t, Spacer(1, 10)]
    for n in notes:
        story.append(Paragraph(tr("• " + n), body))
    doc.build(story)
    return buf.getvalue()
