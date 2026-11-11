

https://github.com/user-attachments/assets/0a873ff0-34be-4928-bcbc-2fdacecf87d9

# AlertScore

**Zaman serisi anomali tespit algoritmaları için karşılaştırma ve güvenilirlik platformu.**

Üç algoritmayı (Rolling Z-Score, Isolation Forest, PyTorch Autoencoder) gerçek NAB (Numenta Anomaly Benchmark) verisi üzerinde, aynı veri bölmesi, aynı eşik kuralı ve aynı metriklerle karşılaştırır. Ayrıca gürültü, veri kayması, tespit gecikmesi ve eşik duyarlılığı deneyleriyle algoritmaların hangi koşulda ne kadar güvenilir olduğunu ölçer.

> Yeni bir algoritma icat edildiği iddiası yoktur. Amaç, mevcut yöntemleri adil ve açık metodolojiyle ölçmektir.

## Hızlı başlangıç (Windows)

1. Python 3.11 veya 3.12 kurun (python.org, kurulumda **Add Python to PATH** işaretli olsun).
2. `setup.bat` dosyasına çift tıklayın (sanal ortam, paketler ve testler).
3. `run_app.bat` ile web arayüzünü başlatın (tarayıcıda açılır).
4. İsteğe bağlı: `run_experiments.bat` tüm deneyleri çalıştırır. **Okumak için `results/AlertScore_Sonuclar.xlsx` dosyasını açın** (ilk sayfa sütun açıklamalarıdır). `results/*.csv` dosyaları başka programlar içindir; Türkçe Excel onları tek sütunda açar. `--quick` ile hızlı deneme yapılır.

5. İsteğe bağlı: `download_data.bat` ek NAB serileri indirir (`--all` ile etiketli tüm seriler). Seri sayısı arttıkça test bölümündeki anomali olayı sayısı da artar ve sonuçlar daha güvenilir olur; ardından `run_experiments.bat` tekrar çalıştırılır.

Sitede takıldığınız yerde **Rehber** sayfasına bakın: grafiğin nasıl okunacağı, her sütunun anlamı ve sonuçları yorumlarken dikkat edilecekler orada.

Elle: `python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`, `streamlit run app\app.py`.

## Klasör yapısı

```
alertscore/
├── app/app.py                 Streamlit arayüzü (6 sayfa)
├── src/alertscore/
│   ├── data.py                CSV okuma/doğrulama, NAB etiketleri
│   ├── models.py              Ortak arayüz + 3 algoritma
│   ├── metrics.py             Nokta ve olay bazlı metrikler
│   ├── experiments.py         Bölme, eşik, benchmark, dayanıklılık deneyleri
│   ├── plots.py               Plotly grafikleri
│   └── report.py              PDF rapor
├── scripts/                   run_experiments.py, download_nab.py
├── tests/test_core.py         Birim testleri
├── data/nab/                  4 NAB serisi + etiketler + lisans
├── results/                   Deney çıktıları (CSV)
├── PROJECT_STATUS.md          Güncel durum, bilinen sınırlar
└── TEKNOLOJILER.md            Kullanılan araçlar ve kurulum notları
```

Yeni algoritma eklemek: `models.py` içinde `BaseDetector`'dan türetin (`fit`, `score`), `make_model` ve `MODEL_NAMES` listesine ekleyin. Geri kalan her şey (metrikler, deneyler, arayüz) değişmeden çalışır.

## Metodoloji

- **Zamansal bölme:** ilk %40 eğitim, sonraki %20 doğrulama, son %40 test. Rastgele karıştırma yoktur; zaman serisinde gelecek geçmişe sızmamalıdır.
- **Nedensellik:** her modelin `score(values)[t]` değeri yalnızca `values[:t+1]` bilgisine bağlıdır (testle doğrulanır).
- **Ön işleme:** ölçekleme parametreleri yalnızca eğitim bölümünden öğrenilir.
- **Eşik:** doğrulama bölümü skorlarının %99.5'lik yüzdeliği (ayarlanabilir). Test verisi eşik seçiminde hiç kullanılmaz; tüm modeller aynı kuralı kullanır (testle doğrulanır). İki kaynak seçeneği vardır: `val_all` (varsayılan, doğrulamadaki tüm noktalar, etiket yok) ve `val_normal` (doğrulamadaki anomali penceresi dışındaki noktalar; doğrulama etiketi kullanılır). Arayüzde 'Ortak ayarlar', komut satırında `--threshold-source` ile seçilir.
- **Etiketler:** NAB etiketleri satır başına 0/1 değil, anomali **pencereleridir** (`combined_windows.json`). Nokta etiketleri zaman damgasından türetilir.
- **Nokta bazlı** (Precision, Recall, F1, PR-AUC) ve **olay bazlı** (Event Recall, gecikme, yanlış alarm) metrikler ayrı tutulur.
- **Yanlış alarm:** hiçbir anomali penceresine değmeyen alarm olayı. Alarm sürekli açık kalırsa tek olay sayıldığı için yanına **yanlış alarm nokta oranı** (normal noktaların yüzde kaçı alarm verdi) da raporlanır.
- **Veri kalitesi:** sıralama bozukluğu düzeltilir, tekrarlı zaman damgalarının ortalaması alınır, boşluklar doldurulmaz. Hepsi Data Explorer'da raporlanır (örnek: `machine_temperature` dosyasında 12 tekrarlı zaman damgası ve 1 sıra bozukluğu var).

## Sonuçlar (11 NAB serisi, 20 test olayı, tohum 0, eşik: val_all)

| Model | Yakalanan olay | F1 | PR-AUC | Yanlış alarm/gün | Yanlış alarm nokta oranı |
|---|---|---|---|---|---|
| Rolling Z-Score | 15/20 | 0.027 | 0.228 | 0.598 | 0.004 |
| Isolation Forest | 18/20 | 0.287 | 0.409 | 0.343 | 0.065 |
| Autoencoder | 15/20 | 0.205 | 0.358 | 0.332 | 0.078 |

İstatistiksel okuma (seri bazlı bootstrap, 5 tohum): Isolation Forest ve Autoencoder, Rolling Z-Score'dan F1 ve PR-AUC'de anlamlı biçimde iyidir (farkın %95 aralığı 0'ı içermez). Isolation Forest ile Autoencoder arasındaki fark ayırt edilemez (F1 farkı 0.082, aralık [-0.010, 0.200]). Tohumlar arası sapma küçüktür (Isolation Forest F1 0.287 ± 0.008).

Gözlemler (tek koşu; kesinlik için `run_experiments.bat` çıktısındaki güven aralıklarına bakın):
- Veri kayması (basamak, 1.5 × std): Isolation Forest'ta yanlış alarm nokta oranı 0.065 → 0.426; Rolling Z-Score sabit (0.004); Autoencoder 0.078 → 0.126.
- Tespit gecikmesi (enjekte 8 × std sıçrama): Z-Score ~0.2 dk, Autoencoder 3.7 dk, Isolation Forest 83 dk.
- Gürültü (0.5 × std'ye kadar): F1 değişimi en fazla ~0.03.
- Eşik, sonucu etkiler: `nyc_taxi` serisinde doğrulama bölümündeki büyük bir anomali Autoencoder eşiğini 0.22'den 0.65'e çıkarıyor ve model test olaylarının hiçbirini yakalamıyor (0/4). `threshold_source=val_normal` ile 2/4 (20 yanlış alarm olayıyla). 11 serinin tamamında bu ayar yalnız bu seride yakalamayı artırdı ve Isolation Forest'ta yanlış alarmı artırdı; bu yüzden varsayılan değil.

## İkinci veri ailesi: SKAB (endüstriyel pompa/vana)

NAB'a uydurulmadığını göstermek için aynı yöntem SKAB (Skoltech Anomaly Benchmark, github.com/waico/SKAB) üzerinde de çalışır. Veri gerçek bir pompa düzeneğinden alınan sensör kayıtlarıdır; vana kapatılarak anomali oluşturulur. **GPL-3.0 lisanslı olduğu için repoya konmaz**, `download_skab.bat` ile indirilir (`data/skab/` .gitignore'dadır).

```
download_skab.bat
run_experiments.bat --dataset skab            (sonuç: results\skab\AlertScore_Sonuclar.xlsx)
run_experiments.bat --dataset skab --sensor Current
```

SKAB'a özgü kararlar (hepsi `SKAB_CONFIG` ve `data.py` içinde):
- Dosyalar ~1100 saniyelik tek deneydir ve anomali ortalarda başlar; bu yüzden bölme %30 eğitim, %15 doğrulama, %55 test (ilk %50 anomalisiz). Eğitim kısa olduğu için Autoencoder 100 epoch eğitilir.
- Veri çok sensörlüdür, proje tek değişkenlidir: her dosyadan **tek sensör** alınır. Varsayılan `Volume Flow RateRMS`, sonuçlara bakmadan fiziksel nedenle seçildi (vana kapanınca akış etkilenir). Diğer sensörler `--sensor` ile denenebilir; sensör seçimi bir hiperparametre sayılmalıdır ve sonuçlar sensöre göre değişebilir.
- Etiket satır başına 0/1'dir; ardışık 1'ler tek anomali penceresine çevrilir. NAB'dan farklı olarak pencere başlangıcı gerçek anomali başlangıcıdır, dolayısıyla gecikme gerçek tespit gecikmesine daha yakındır.
- Test bölümü yalnızca ~9 dakikadır; "yanlış alarm / gün" bu yüzden çok büyük sayılara çıkar (tek alarm yüzlerce/gün eder). Bu sütunu yorumlamayın; **yanlış alarm nokta oranı** ve alarm sayısına bakın.
- Her dosyada tek anomali vardır: `valve1` + `valve2` ile toplam 20 olay.

## Dayanıklılık deneyleri

Gerçek NAB serileri üzerine kontrollü değişiklik uygulanır. Etiketler zaman damgası penceresi olduğu için değişiklikten etkilenmez.

1. **Gürültü:** tüm seriye Gaussian gürültü (std = seviye × eğitim std'si), modeller yeniden eğitilir.
2. **Veri kayması:** test bölümünde normal seviye kaydırılır (basamak veya rampa). Pencere dışındaki alarmlar yanlış alarmdır.
3. **Tespit gecikmesi:** test bölümüne başlangıç anı bilinen yapay seviye sıçramaları eklenir, ilk alarma kadar geçen süre ölçülür.
4. **Eşik duyarlılığı:** aynı skorlarla eşik yüzdeliği değiştirilir.

## Veri ve lisans

Veri: **Numenta Anomaly Benchmark (NAB)**, https://github.com/numenta/NAB. Lisans: MIT, Copyright 2014-2024 Numenta Inc. (`data/nab/LICENSE.txt`, telif notu korunmalıdır). Projede dört seri hazır gelir; `scripts/download_nab.py` ek seri indirir.

Atıf: Lavin, A. ve Ahmad, S. (2015). *Evaluating Real-time Anomaly Detection Algorithms – the Numenta Anomaly Benchmark.* IEEE ICMLA.

## Sınırlılıklar

- NAB pencereleri geniştir ve gerçek anomali başlangıcını değil çevresini işaretler. Nokta bazlı Recall bu yüzden düşük çıkar; gerçek gecikme yalnızca enjekte edilen anomalilerle ölçülebilir.
- Test bölümündeki olay sayısı azdır (11 seride toplam 20). Bu yüzden güven aralıkları geniştir; Excel raporundaki 'Güven aralığı' ve 'Model farkları' sayfaları (seri bazlı bootstrap) hangi farkların ayırt edilebildiğini gösterir. Farkı 0'ı içeren karşılaştırmalar yorumlanmamalıdır. Daha çok seri için `download_data.bat --all`.
- Eşik kuralı doğrulama bölümünün yüzdeliğine dayanır; doğrulama bölümünde anomali varsa eşik biraz yükselir.
- Tek değişkenli seriler; çok değişkenli veri ve ek modeller henüz yoktur.
- Zaman boşlukları doldurulmaz; pencereler satır sırasına göre kayar (`ambient_temperature` serisinde 10 boşluk var).
- PDF rapor yalnızca tablo ve notlar içerir, grafik içermez.
