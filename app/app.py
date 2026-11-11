"""AlertScore Streamlit arayüzü.  Çalıştırma:  streamlit run app/app.py  (veya run_app.bat)"""

from __future__ import annotations

import io
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from alertscore.data import MAX_UPLOAD_MB, DataFormatError, list_nab_series, load_nab, load_uploaded_csv  # noqa: E402
from alertscore.experiments import (  # noqa: E402
    benchmark, delay_experiment, drift_experiment, noise_experiment, run_models, save_results,
    split_points, summarize_benchmark, threshold_sensitivity,
)
from alertscore.labels import (  # noqa: E402
    INFO, explain_result, glossary, metric_table, name_of, pretty, trend_sentences,
)
from alertscore.models import MODEL_NAMES, Config  # noqa: E402
from alertscore.plots import line_by_level, loss_curve, metric_bars, series_plot  # noqa: E402
from alertscore.report import build_pdf_from_results, build_xlsx  # noqa: E402

RESULTS = ROOT / "results"
st.set_page_config(page_title="AlertScore", page_icon="📈", layout="wide")

NAMES = {k: v[0] for k, v in INFO.items()}
FRACTIONS = {"false_alarm_point_rate", "event_recall", "detection_rate", "precision", "recall"}

QUALITY_NAMES = {
    "rows_raw": "Ham satır sayısı", "rows": "Temizlik sonrası satır sayısı",
    "invalid_timestamps": "Geçersiz zaman damgası (atıldı)",
    "missing_or_non_numeric_values": "Eksik / sayı olmayan değer (atıldı)",
    "backward_time_steps": "Zamanda geriye giden adım (sıralandı)",
    "duplicate_timestamps": "Tekrarlı zaman damgası (ortalaması alındı)",
    "median_step": "Tipik ölçüm aralığı", "gaps": "Zaman boşluğu sayısı (doldurulmadı)",
    "largest_gap": "En büyük boşluk", "start": "Başlangıç", "end": "Bitiş",
}

PAGE_HELP = {
    "Data Explorer": ("Seçtiğiniz zaman serisini ve NAB'ın işaretlediği gerçek anomali pencerelerini gösterir; veri kalitesini raporlar.",
                      "Mavi çizgi ölçüm değerleridir. Kırmızı bantlar gerçek anomali pencereleridir. Sağdaki tablo verideki sorunları ve nasıl ele alındığını söyler."),
    "Model Laboratory": ("Tek bir modeli seçilen seri üzerinde çalıştırır ve ne yaptığını gösterir.",
                         "Önce üstteki cümleleri okuyun (özet). Grafikte kırmızı noktalar modelin alarm verdiği yerlerdir; kırmızı bantlarla örtüşmeleri iyidir, bantların dışındakiler yanlış alarmdır."),
    "Benchmark Center": ("Üç modeli aynı serilerde, aynı kurallarla yarıştırır.",
                         "Önce 'Kısa yorum'a bakın. Tablodaki sütunların anlamı için sayfanın altındaki 'Sütunlar ne demek?' bölümünü açın."),
    "Robustness Lab": ("Modelleri zor koşullarda dener: gürültü, veri kayması, farklı büyüklükte anomaliler, farklı eşikler.",
                       "Her sekmede önce 'Ne soruyoruz?' kutusunu okuyun. Grafikte çizgi yukarı veya aşağı gittikçe modelin o koşuldan nasıl etkilendiğini görürsünüz."),
    "Reports": ("Kayıtlı sonuçları Excel ve PDF olarak indirir.",
                "Excel dosyasının ilk sayfası 'Okuma rehberi'dir ve her sütunun ne demek olduğunu açıklar."),
}


@st.cache_data(show_spinner=False)
def get_nab(key: str):
    return load_nab(key)


def read_result(name: str):
    p = RESULTS / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def intro(page: str):
    what, how = PAGE_HELP[page]
    st.info(f"**Bu sayfa ne işe yarar?** {what}\n\n**Nasıl okunur?** {how}")


def bullets(lines: list[str]):
    st.markdown("\n".join(f"- {line}" for line in lines))


# ------------------------------------------------------------------ kenar çubuğu

def sidebar_config() -> Config:
    with st.sidebar.expander("Ortak ayarlar (tüm modeller)"):
        st.caption("Tüm modellere aynı ayarlar uygulanır; böylece karşılaştırma adil olur.")
        q = st.slider("Eşik yüzdeliği (yüksek = daha az alarm)", 0.90, 0.999, 0.995, 0.001, format="%.3f")
        window = st.slider("Pencere uzunluğu (Isolation Forest / Autoencoder)", 8, 64, 16, 4)
        zwin = st.slider("Z-Score kayan pencere", 20, 500, 100, 10)
        epochs = st.slider("Autoencoder epoch (eğitim turu)", 5, 100, 30, 5)
        seed = st.number_input("Rastgelelik tohumu (aynı tohum = aynı sonuç)", 0, 9999, 0)
        source = st.selectbox(
            "Eşik kaynağı", ["val_all", "val_normal"],
            format_func=lambda k: {"val_all": "Doğrulamadaki tüm noktalar (etiket kullanılmaz)",
                                   "val_normal": "Doğrulamada yalnız normal noktalar (doğrulama etiketi kullanılır)"}[k])
        st.caption("İkincisi, doğrulama bölümüne düşen bir anomalinin eşiği şişirmesini önler. Test etiketleri iki durumda da kullanılmaz.")
    return Config(quantile=float(q), window=int(window), z_window=int(zwin), ae_epochs=int(epochs),
                  seed=int(seed), threshold_source=source)


def sidebar_series():
    st.sidebar.markdown("### Veri")
    src = st.sidebar.radio("Kaynak", ["NAB (gerçek, etiketli)", "CSV yükle (etiketsiz)"])
    if src.startswith("NAB"):
        keys = list_nab_series()
        if not keys:
            st.sidebar.warning("data/nab içinde seri yok. scripts/download_nab.py çalıştırın.")
            return None
        return get_nab(st.sidebar.selectbox("Seri", keys))
    up = st.sidebar.file_uploader("timestamp,value sütunlu CSV", type=["csv"])
    if up is None:
        st.sidebar.info("CSV'de 'timestamp' ve 'value' sütunları olmalı.")
        return None
    data = up.getvalue()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        st.sidebar.error(f"Dosya {MAX_UPLOAD_MB} MB'dan büyük olamaz.")
        return None
    try:
        return load_uploaded_csv(io.BytesIO(data), up.name)
    except DataFormatError as exc:
        st.sidebar.error(str(exc))
        return None


# ------------------------------------------------------------------ sayfalar

def page_overview():
    st.title("AlertScore")
    st.caption("Zaman serisi anomali tespit algoritmalarını adil ve güvenilir biçimde karşılaştıran platform")
    st.info(
        "**İlk kez mi kullanıyorsunuz? Önerilen sıra:**\n\n"
        "1. **Data Explorer:** Bir seriye bakın. Kırmızı bantlar gerçek anomalilerdir.\n"
        "2. **Model Laboratory:** Tek bir modeli çalıştırın, alarmların bantlarla örtüşüp örtüşmediğine bakın.\n"
        "3. **Benchmark Center:** Üç modeli yarıştırın, 'Kısa yorum'u okuyun.\n"
        "4. **Robustness Lab:** Modelleri gürültü, veri kayması gibi zor koşullarda deneyin.\n"
        "5. **Reports:** Sonuçları Excel olarak indirin.\n\n"
        "Bir terimi anlamadıysanız **Rehber** sayfasına bakın."
    )
    keys = list_nab_series()
    bench = read_result("benchmark")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kullanılabilir NAB serisi", len(keys))
    c2.metric("Model sayısı", len(MODEL_NAMES))
    if bench is None:
        c3.metric("Benchmark serisi", "-")
        c4.metric("Test edilen anomali olayı", "-")
        st.warning("Henüz kaydedilmiş sonuç yok. Benchmark Center'dan çalıştırıp kaydedin veya run_experiments.bat'ı "
                   "çalıştırın. Bu sayfadaki sayılar yalnızca gerçek deney sonuçlarından gelir.")
        return
    c3.metric("Benchmark serisi", int(bench["series"].nunique()))
    c4.metric("Test edilen anomali olayı", int(bench.drop_duplicates("series")["events"].sum()))
    st.subheader("Son benchmark (model başına ortalama)")
    st.dataframe(pretty(summarize_benchmark(bench)))
    st.caption(f"Çalıştırma zamanı: {bench['run_at'].iloc[0]}. Sütunların anlamı için Rehber sayfasına bakın.")
    avail = [n for n in ("noise", "drift", "delay", "threshold_sensitivity") if (RESULTS / f"{n}.csv").exists()]
    st.write("Kayıtlı dayanıklılık deneyleri:", ", ".join(avail) if avail else "yok")


def page_guide():
    st.header("Rehber")
    st.subheader("Bu site ne yapar?")
    st.write("Bir anomali tespit algoritmasının iyi olması yalnızca anomaliyi yakalamasıyla ölçülmez: kaç yanlış alarm "
             "verdiği, ne kadar geç yakaladığı ve veri değişince ne olduğu da önemlidir. AlertScore üç algoritmayı "
             "(Rolling Z-Score, Isolation Forest, Autoencoder) aynı gerçek veri ve aynı kurallarla ölçer.")
    st.subheader("Grafikte ne görüyorum?")
    bullets([
        "**Mavi çizgi:** serinin ölçüm değerleri.",
        "**Kırmızı bant:** NAB'ın etiketlediği gerçek anomali penceresi.",
        "**Kırmızı noktalar:** modelin alarm verdiği noktalar. Bantların içindeyse doğru, dışındaysa yanlış alarm.",
        "**Alt grafik:** modelin anomali skoru. Kesikli çizgi eşiktir; skor eşiği aşınca alarm verilir.",
        "**Gri noktalı dikey çizgiler:** eğitim | doğrulama | test bölümlerinin sınırları. Ölçüm sadece son bölümde (test) yapılır.",
    ])
    st.subheader("Üç model ne yapıyor?")
    bullets([
        "**Rolling Z-Score:** Bir değer, kendinden önceki son birkaç değerin ortalamasından kaç standart sapma uzaksa o kadar anormal sayılır. En basit yöntem.",
        "**Isolation Forest:** Son birkaç değerden oluşan 'şekilleri' rastgele ağaçlarla ayırmaya çalışır; kolay ayrılan şekil anormaldir.",
        "**Autoencoder:** Küçük bir sinir ağı normal şekilleri sıkıştırıp yeniden çizmeyi öğrenir. Yeniden çizme hatası büyükse şekil anormaldir.",
    ])
    st.subheader("Sonuçları yorumlarken")
    bullets([
        "En anlamlı doğruluk ölçüsü **Olay yakalama oranı**dır. 'Nokta yakalama' NAB pencereleri geniş olduğu için düşük çıkar, bu normaldir.",
        "Yanlış alarmı iki ölçüyle görün: **Yanlış alarm / gün** ve **Yanlış alarm nokta oranı**. Alarm sürekli açık kalırsa ilki düşük görünür.",
        "Test bölümündeki anomali olayı sayısı azdır. Küçük farklara kesin gözle bakmayın; sonuçlar tek koşudur.",
        "**Doğrulamadaki anomali olayı** 0'dan büyükse o serinin eşiği yüksek çıkmış, model daha az alarm vermiş olabilir.",
    ])
    st.subheader("Sütunlar ne demek?")
    st.dataframe(glossary())


def page_data_explorer(series):
    st.header("Data Explorer")
    intro("Data Explorer")
    if series is None:
        st.warning("Kenar çubuğundan bir seri seçin veya CSV yükleyin.")
        return
    st.plotly_chart(series_plot(series, title=series.name))
    a, b = st.columns(2)
    with a:
        st.subheader("Veri kalitesi raporu")
        st.dataframe(pd.DataFrame({"Değer": {QUALITY_NAMES.get(k, k): str(v) for k, v in series.quality.items()}}))
        q = series.quality
        if q["backward_time_steps"] or q["duplicate_timestamps"] or q["gaps"]:
            st.warning("Bu veride düzeltilen veya uyarılan sorunlar var (tabloda parantez içinde ne yapıldığı yazıyor). "
                       "Zaman boşlukları doldurulmaz, çünkü doldurmak olmayan veri uydurmak olur.")
    with b:
        st.subheader("Değerlerin özeti")
        desc = series.df["value"].describe().round(3).to_frame("Değer")
        desc.index = ["Adet", "Ortalama", "Standart sapma", "En küçük", "%25", "Medyan", "%75", "En büyük"]
        st.dataframe(desc)
    if series.windows:
        st.subheader("Anomali pencereleri (etiketler)")
        st.dataframe(pd.DataFrame(series.windows, columns=["Başlangıç", "Bitiş"]).astype(str))
    st.subheader("İlk 20 satır")
    st.dataframe(series.df.head(20).rename(columns={"timestamp": "Zaman", "value": "Değer"}))


def page_model_lab(series, cfg):
    st.header("Model Laboratory")
    intro("Model Laboratory")
    if series is None:
        st.warning("Kenar çubuğundan bir seri seçin veya CSV yükleyin.")
        return
    model = st.selectbox("Model", MODEL_NAMES)
    if st.button("Modeli çalıştır"):
        try:
            with st.spinner("Model eğitiliyor ve skorlanıyor..."):
                res = run_models(series, cfg, [model])[model]
            st.session_state["lab"] = (series.name, model, res, asdict(cfg))
        except ImportError as exc:
            st.error(f"Gerekli paket kurulu değil: {exc}. setup.bat dosyasını çalıştırdığınızdan emin olun.")
    lab = st.session_state.get("lab")
    if not lab or lab[0] != series.name:
        return
    _, model, res, _cfg = lab
    train_end, val_end = split_points(len(series.df), Config(**_cfg))
    st.subheader(f"Sonuç özeti: {model}")
    if series.has_labels:
        bullets(explain_result(model, res.metrics, _cfg.get("threshold_source", "val_all")))
    else:
        m = res.metrics
        st.info("Yüklenen CSV etiketsiz olduğu için doğruluk ölçüleri hesaplanamaz; yalnızca alarmlar gösterilir.")
        bullets([f"Test bölümünde {m['alert_episodes']} alarm olayı verildi (eşik: {m['threshold']:.3g})."])
    st.plotly_chart(series_plot(series, res.scores, res.threshold, train_end, val_end, title=f"{model} — {series.name}"))
    if series.has_labels:
        st.subheader("Ölçüler")
        st.dataframe(metric_table(res.metrics))
    if res.loss_history:
        st.plotly_chart(loss_curve(res.loss_history))
        st.caption("Eğitim kaybı: model normal şekilleri yeniden çizerken yaptığı hata. Tur tur düşüyorsa model öğreniyor demektir.")


def _need_labeled():
    keys = list_nab_series()
    if not keys:
        st.warning("Etiketli NAB serisi bulunamadı (data/nab).")
    return keys


def benchmark_comment(df: pd.DataFrame) -> list[str]:
    s = summarize_benchmark(df)
    out = []
    ev = s.dropna(subset=["event_recall"])
    if len(ev):
        b = ev.loc[ev["event_recall"].idxmax()]
        out.append(f"En çok anomali olayını yakalayan: **{b['model']}** "
                   f"({int(b['events_detected_total'])}/{int(b['events_total'])} olay, %{b['event_recall'] * 100:.0f}).")
    fa = s.dropna(subset=["false_alarm_point_rate"])
    if len(fa):
        b = fa.loc[fa["false_alarm_point_rate"].idxmin()]
        out.append(f"En az yanlış alarm veren: **{b['model']}** (normal noktaların %{b['false_alarm_point_rate'] * 100:.1f}'i alarm verdi).")
    sp = s.dropna(subset=["inference_ms_per_1k_points"])
    if len(sp):
        b = sp.loc[sp["inference_ms_per_1k_points"].idxmin()]
        out.append(f"En hızlı: **{b['model']}** (1000 nokta için {b['inference_ms_per_1k_points']:.2f} ms).")
    if "val_events" in df.columns:
        bad = sorted(df.loc[df["val_events"] > 0, "series"].unique())
        if bad:
            out.append("Dikkat: şu serilerde eşiğin seçildiği doğrulama bölümünde anomali var; eşik yüksek çıkmış, "
                       "model daha az alarm vermiş olabilir: " + ", ".join(bad))
    out.append("Not: olay sayısı azdır ve sonuçlar tek koşudur; küçük farkları kesin kabul etmeyin.")
    return out


def page_benchmark(cfg):
    st.header("Benchmark Center")
    intro("Benchmark Center")
    keys = _need_labeled()
    if not keys:
        return
    chosen = st.multiselect("Seriler", keys, default=keys)
    models = st.multiselect("Modeller", MODEL_NAMES, default=MODEL_NAMES)
    if st.button("Benchmark'ı çalıştır") and chosen and models:
        bar = st.progress(0.0)
        try:
            df = benchmark([get_nab(k) for k in chosen], cfg, models,
                           progress=lambda i, n, s: bar.progress(i / n, text=s))
            bar.progress(1.0)
            st.session_state["benchmark"] = (df, asdict(cfg))
        except ImportError as exc:
            st.error(f"Gerekli paket kurulu değil: {exc}")
    cur = st.session_state.get("benchmark")
    if cur is None:
        saved = read_result("benchmark")
        if saved is None:
            return
        st.caption("Gösterilen: diskteki kayıtlı sonuç.")
        df = saved.drop(columns=["run_at", "config"], errors="ignore")
    else:
        df = cur[0]
        if st.button("Sonuçları results/ klasörüne kaydet"):
            save_results(df, "benchmark", Config(**cur[1]), RESULTS)
            st.success("Kaydedildi: results/benchmark.csv (Excel için Reports sayfasından indirin)")
    st.subheader("Kısa yorum")
    bullets(benchmark_comment(df))
    st.subheader("Model başına ortalama")
    st.dataframe(pretty(summarize_benchmark(df)))
    st.plotly_chart(metric_bars(df, ["event_recall", "f1", "pr_auc"], "Doğruluk ölçüleri (yüksek iyi)", NAMES))
    st.plotly_chart(metric_bars(df, ["false_alarm_point_rate", "false_alarms_per_day", "inference_ms_per_1k_points"],
                                "Yanlış alarm ve hız (düşük iyi)", NAMES))
    st.subheader("Seri bazında ayrıntı")
    st.dataframe(pretty(df))
    with st.expander("Sütunlar ne demek?"):
        st.dataframe(glossary())


def _pick_series_and_models():
    keys = _need_labeled()
    if not keys:
        return None, None
    key = st.selectbox("Seri", keys, key="rob_series")
    models = st.multiselect("Modeller", MODEL_NAMES, default=MODEL_NAMES, key="rob_models")
    return get_nab(key), models


FILE_NAMES = {"threshold": "threshold_sensitivity"}   # results/ içindeki dosya adı farklıysa
# deney -> (x sütunu, grafik ölçüleri, yorum cümlesi ölçüleri)
ROBUST = {
    "noise": ("noise_level", ["event_recall", "f1", "false_alarm_point_rate"], ["event_recall", "false_alarm_point_rate"]),
    "drift": ("drift_level", ["false_alarm_point_rate", "false_alarms_per_day", "event_recall"], ["false_alarm_point_rate", "event_recall"]),
    "delay": ("magnitude_sd", ["detection_rate", "mean_delay_min"], ["detection_rate", "mean_delay_min"]),
    "threshold": ("quantile", ["precision", "recall", "false_alarm_point_rate"], ["false_alarm_point_rate", "recall"]),
}


def show_robust(name: str):
    cur = st.session_state.get(f"rob_{name}")
    if cur is None:
        return
    df = cur[0]
    if df.empty:
        st.warning("Bu seri için sonuç üretilemedi (ör. test bölümünde uygun yer bulunamadı).")
        return
    x, chart_ys, trend_ys = ROBUST[name]
    st.subheader("Kısa yorum")
    for y in trend_ys:
        bullets(trend_sentences(df, x, y, as_percent=y in FRACTIONS))
    for y in chart_ys:
        st.plotly_chart(line_by_level(df, x, y, f"{name_of(x)} ve {name_of(y)}", name_of(y), name_of(x)))
    st.dataframe(pretty(df))
    if st.button("Bu deneyi results/ klasörüne kaydet", key=f"save_{name}"):
        fname = FILE_NAMES.get(name, name)
        save_results(df, fname, Config(**cur[1]), RESULTS)
        st.success(f"Kaydedildi: results/{fname}.csv (Excel için Reports sayfasından indirin)")


def page_robustness(cfg):
    st.header("Robustness Lab")
    intro("Robustness Lab")
    st.caption("Deneyler gerçek NAB serileri üzerinde, kontrollü değişiklikle yapılır (gürültü/kayma/anomali eklenir). "
               "Etiketler zaman damgası pencereleriyle tanımlı olduğu için değişiklikten etkilenmez.")
    series, models = _pick_series_and_models()
    if series is None or not models:
        return
    tabs = st.tabs(["Gürültü", "Veri kayması", "Tespit gecikmesi", "Eşik duyarlılığı"])

    def guarded(fn):
        try:
            with st.spinner("Deney çalışıyor (Autoencoder seçiliyse biraz sürebilir)..."):
                return fn()
        except ImportError as exc:
            st.error(f"Gerekli paket kurulu değil: {exc}")
            return None

    with tabs[0]:
        st.info("**Ne soruyoruz?** Ölçümler kirliyse (sensör gürültüsü) modeller bozulur mu?\n\n"
                "**Ne yapıyoruz?** Seriye rastgele gürültü ekliyoruz (seviye 0 = temiz veri) ve modelleri yeniden eğitiyoruz.\n\n"
                "**Grafikte ne aramalı?** Sağa gidildikçe olay yakalama düşüyor veya yanlış alarm artıyorsa model gürültüye duyarlıdır.")
        lv = st.multiselect("Gürültü seviyeleri (× eğitim std)", [0.0, 0.1, 0.25, 0.5, 1.0], default=[0.0, 0.1, 0.25, 0.5], key="n_lv")
        rep = st.slider("Tekrar sayısı (farklı rastgele gürültü)", 1, 5, 2, key="n_rep")
        if st.button("Gürültü deneyini çalıştır") and lv:
            df = guarded(lambda: noise_experiment(series, cfg, levels=tuple(lv), repeats=rep, models=models))
            if df is not None:
                st.session_state["rob_noise"] = (df, asdict(cfg))
        show_robust("noise")
    with tabs[1]:
        st.info("**Ne soruyoruz?** Normal çalışma seviyesi değişirse (ör. motor 60'tan 65 dereceye kayarsa) model bunu anomali sanıp "
                "yanlış alarm verir mi?\n\n"
                "**Ne yapıyoruz?** Test bölümünün bir yerinden sonra değerleri kaydırıyoruz. Etiketler değişmediği için pencere dışındaki her alarm yanlış alarmdır.\n\n"
                "**Grafikte ne aramalı?** 'Yanlış alarm nokta oranı' kayma büyüdükçe yükseliyorsa model kaymaya uyum sağlayamıyor demektir.")
        kind = st.radio("Kayma türü", ["step", "ramp"], horizontal=True, key="d_kind")
        st.caption("step = ani basamak, ramp = yavaş yavaş artan kayma")
        lv = st.multiselect("Kayma büyüklüğü (× eğitim std)", [0.0, 0.5, 1.0, 1.5, 2.0], default=[0.0, 0.5, 1.0, 1.5], key="d_lv")
        if st.button("Kayma deneyini çalıştır") and lv:
            df = guarded(lambda: drift_experiment(series, cfg, levels=tuple(lv), kind=kind, models=models))
            if df is not None:
                st.session_state["rob_drift"] = (df, asdict(cfg))
        show_robust("drift")
    with tabs[2]:
        st.info("**Ne soruyoruz?** Bir arıza çıktığında model ne kadar çabuk haber veriyor, küçük arızaları yakalıyor mu?\n\n"
                "**Ne yapıyoruz?** Test bölümüne başlangıç anı kesin bilinen yapay seviye sıçramaları ekliyoruz (NAB pencerelerinde gerçek başlangıç bilinmez).\n\n"
                "**Grafikte ne aramalı?** Küçük büyüklükte yakalama oranı düşüyorsa model küçük anomalileri kaçırıyor demektir. Gecikme ne kadar düşükse o kadar iyidir.")
        mags = st.multiselect("Anomali büyüklüğü (× eğitim std)", [1, 2, 3, 5, 8], default=[2, 3, 5, 8], key="l_mag")
        dur = st.slider("Anomali süresi (veri noktası)", 6, 96, 24, 6, key="l_dur")
        if st.button("Gecikme deneyini çalıştır") and mags:
            df = guarded(lambda: delay_experiment(series, cfg, magnitudes=tuple(mags), duration=dur, models=models))
            if df is not None:
                st.session_state["rob_delay"] = (df, asdict(cfg))
        show_robust("delay")
    with tabs[3]:
        st.info("**Ne soruyoruz?** Eşiği değiştirirsek ne kazanıp ne kaybediyoruz?\n\n"
                "**Ne yapıyoruz?** Aynı skorlar üzerinde eşik yüzdeliğini değiştiriyoruz (yüzdelik yükseldikçe eşik büyür, alarm azalır).\n\n"
                "**Grafikte ne aramalı?** Yanlış alarm azalırken yakalama da düşüyorsa bir denge söz konusudur; iyi model bu dengede daha iyi bir nokta sunar.")
        qs = st.multiselect("Eşik yüzdelikleri", [0.90, 0.95, 0.98, 0.99, 0.995, 0.999],
                            default=[0.90, 0.95, 0.98, 0.99, 0.995, 0.999], key="t_q")
        if st.button("Eşik deneyini çalıştır") and qs:
            df = guarded(lambda: threshold_sensitivity(run_models(series, cfg, models), series, cfg, tuple(sorted(qs))))
            if df is not None:
                st.session_state["rob_threshold"] = (df, asdict(cfg))
        show_robust("threshold")


def page_reports():
    st.header("Reports")
    intro("Reports")
    xlsx = build_xlsx(RESULTS)
    if xlsx is None:
        st.warning("results/ klasöründe kayıtlı sonuç yok. Önce Benchmark Center / Robustness Lab'dan kaydedin "
                   "veya run_experiments.bat çalıştırın.")
        return
    st.subheader("Excel raporu")
    st.write("Tüm kayıtlı sonuçlar tek dosyada: her deney için bir 'özet' ve bir 'detay' sayfası, ilk sayfada sütun açıklamaları.")
    st.download_button("Excel raporunu indir (.xlsx)", xlsx, file_name="AlertScore_Sonuclar.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.subheader("PDF rapor")
    if st.button("PDF oluştur"):
        st.session_state["pdf"] = build_pdf_from_results(RESULTS)
    if st.session_state.get("pdf"):
        st.download_button("PDF'i indir", st.session_state["pdf"], file_name="alertscore_rapor.pdf", mime="application/pdf")
    with st.expander("Ham CSV dosyaları (başka programlar için)"):
        st.caption("Dikkat: Türkçe Excel bu CSV'leri tek sütunda açar. İnsan okuması için yukarıdaki Excel raporunu kullanın.")
        for f in sorted(RESULTS.glob("*.csv")):
            st.download_button(f"{f.name} indir", f.read_bytes(), file_name=f.name, mime="text/csv", key=f"dl_{f.name}")


# ------------------------------------------------------------------ ana akış

PAGES = ["Overview", "Rehber", "Data Explorer", "Model Laboratory", "Benchmark Center", "Robustness Lab", "Reports"]
page = st.sidebar.radio("Sayfa", PAGES)
config = sidebar_config()
current = sidebar_series() if page in ("Data Explorer", "Model Laboratory") else None

if page == "Overview":
    page_overview()
elif page == "Rehber":
    page_guide()
elif page == "Data Explorer":
    page_data_explorer(current)
elif page == "Model Laboratory":
    page_model_lab(current, config)
elif page == "Benchmark Center":
    page_benchmark(config)
elif page == "Robustness Lab":
    page_robustness(config)
else:
    page_reports()
