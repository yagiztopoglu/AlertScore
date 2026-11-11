# Kullanılan teknolojiler ve araçlar (not)

## Çalışma ortamı

| Araç | Ne işe yarıyor |
|---|---|
| **Python 3.11 / 3.12** | Ana dil. python.org'dan kurulur. |
| **venv** (`python -m venv .venv`) | Proje için izole paket ortamı. Diğer projeleri etkilemez. `.venv` klasörü silinirse ortam sıfırlanır. |
| **pip** (`pip install -r requirements.txt`) | Python paket yöneticisi. Paketleri PyPI'dan indirir. |
| **.bat dosyaları** | Windows'ta komutları çift tıkla çalıştırır (`setup.bat`, `run_app.bat`, `run_experiments.bat`). |
| **unittest** | Python'un yerleşik test aracı; ek kurulum gerektirmez. |

**npm / Node.js kullanılmıyor.** Proje tamamen Python'dur; Streamlit arayüzü Python'dan üretilir.

## Python paketleri (requirements.txt)

| Paket | Projedeki görevi | Alternatifi |
|---|---|---|
| pandas | Tablo/zaman serisi okuma ve işleme | Polars |
| numpy | Sayısal hesap, pencereleme | Saf Python (çok yavaş) |
| scikit-learn | Isolation Forest, PR-AUC | PyOD |
| torch (PyTorch) | Autoencoder (CPU yeterli) | TensorFlow/Keras |
| plotly | Etkileşimli grafikler | Matplotlib (statik) |
| streamlit | Web arayüzü | Dash, Gradio |
| reportlab | PDF rapor | WeasyPrint, fpdf2 |
| xlsxwriter | Biçimli Excel raporu (.xlsx) | openpyxl |

## Bu ortamda test edilen sürümler

Python 3.13, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, reportlab 5.0.1, xlsxwriter 3.2.9.
**Kurulamadığı için test edilmedi:** torch, plotly, streamlit (kendi bilgisayarınızda `pip` en güncel sürümleri kuracak).

## Veri

NAB (Numenta Anomaly Benchmark), MIT lisansı. `data/nab/` içinde 4 seri, `labels/combined_windows.json` ve `LICENSE.txt`.
