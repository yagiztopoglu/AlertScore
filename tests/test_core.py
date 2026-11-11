"""Çekirdek davranış testleri.  Çalıştırma:  python -m unittest discover -s tests -v

Buradaki yapay diziler yalnızca KODUN doğruluğunu sınamak içindir; deney sonucu değildir.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alertscore.data import DataFormatError, clean_frame, list_nab_series, load_nab  # noqa: E402
from alertscore.experiments import evaluate_fitted, fit_models, split_points  # noqa: E402
from alertscore.metrics import alert_episodes, evaluate  # noqa: E402
from alertscore.models import Config, MODEL_NAMES, make_model  # noqa: E402

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MetricsTest(unittest.TestCase):
    def test_hand_computed_example(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="min")
        scores = np.zeros(100)
        scores[45] = 1.0   # anomali penceresinin içinde
        scores[70] = 1.0   # pencere dışında => yanlış alarm
        windows = [(ts[40], ts[59])]
        m = evaluate(ts, scores, 0.5, windows, test_start=0)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 1 / 20)
        self.assertEqual(m["events"], 1)
        self.assertEqual(m["events_detected"], 1)
        self.assertAlmostEqual(m["mean_delay_min"], 5.0)
        self.assertEqual(m["false_alarms"], 1)
        self.assertAlmostEqual(m["false_alarms_per_day"], 1 / (99 / 1440))
        self.assertAlmostEqual(m["false_alarm_point_rate"], 1 / 80)

    def test_alert_episodes_merge(self):
        pred = np.zeros(30, dtype=bool)
        pred[[3, 4, 6, 20]] = True
        self.assertEqual(alert_episodes(pred, merge_gap=1), [(3, 6), (20, 20)])
        self.assertEqual(alert_episodes(pred, merge_gap=0), [(3, 4), (6, 6), (20, 20)])

    def test_undefined_values_are_nan_not_zero(self):
        ts = pd.date_range("2024-01-01", periods=50, freq="min")
        m = evaluate(ts, np.zeros(50), 0.5, [], test_start=0)   # hiç anomali ve hiç alarm yok
        self.assertTrue(np.isnan(m["recall"]))
        self.assertTrue(np.isnan(m["event_recall"]))


class ThresholdSourceTest(unittest.TestCase):
    def setUp(self):
        n = 1000
        self.ts = pd.date_range("2024-01-01", periods=n, freq="min")
        rng = np.random.default_rng(1)
        self.scores = rng.random(n)
        # doğrulama bölümü = [400, 600); içine anomali penceresi ve çok yüksek skorlar koyuyoruz
        self.windows = [(self.ts[450], self.ts[480])]
        self.scores[450:481] = 100.0

    def test_val_all_is_inflated_by_val_anomaly(self):
        from alertscore.experiments import pick_threshold
        self.assertGreater(pick_threshold(self.scores, self.ts, self.windows, Config(threshold_source="val_all")), 50)

    def test_val_normal_ignores_anomaly_window_and_test_region(self):
        from alertscore.experiments import pick_threshold
        cfg = Config(threshold_source="val_normal")
        t1 = pick_threshold(self.scores, self.ts, self.windows, cfg)
        self.assertLessEqual(t1, 1.0)                      # yalnız normal skorlar (0-1 arası)
        s2 = self.scores.copy()
        s2[600:] += 50                                     # test bölgesi değişse eşik değişmemeli
        self.assertAlmostEqual(t1, pick_threshold(s2, self.ts, self.windows, cfg))


class ReportTest(unittest.TestCase):
    """Excel raporu ve Türkçe açıklamalar."""

    def test_every_result_column_has_a_turkish_description(self):
        from alertscore.labels import INFO
        from alertscore.experiments import benchmark, summarize_benchmark
        if not list_nab_series():
            self.skipTest("data/nab içinde seri yok")
        df = benchmark([load_nab(list_nab_series()[0])], Config(), ["Rolling Z-Score"])
        for col in list(df.columns) + list(summarize_benchmark(df).columns):
            self.assertIn(col, INFO, f"'{col}' sütununun açıklaması eksik (labels.INFO)")

    def test_explain_sentences_use_real_numbers(self):
        from alertscore.labels import explain_result
        m = {"events": 4, "events_detected": 3, "event_recall": 0.75, "alert_episodes": 5, "false_alarms": 2,
             "false_alarms_per_day": 0.1, "false_alarm_point_rate": 0.02, "val_events": 1}
        text = " ".join(explain_result("Isolation Forest", m))
        self.assertIn("4 anomali olayı", text)
        self.assertIn("3 tanesinde", text)
        self.assertIn("%75", text)
        self.assertIn("doğrulama", text)   # doğrulamada anomali uyarısı
        m["alert_episodes"] = 0
        self.assertIn("Hiç alarm verilmedi", " ".join(explain_result("Autoencoder", m)))

    def test_xlsx_roundtrip(self):
        import tempfile
        from alertscore.experiments import benchmark, noise_experiment, save_results
        from alertscore.report import build_xlsx
        if not list_nab_series():
            self.skipTest("data/nab içinde seri yok")
        s, cfg = load_nab(list_nab_series()[0]), Config()
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(build_xlsx(d))                       # sonuç yoksa None
            save_results(benchmark([s], cfg, ["Rolling Z-Score"]), "benchmark", cfg, d)
            save_results(noise_experiment(s, cfg, levels=(0.0, 0.1), repeats=1, models=["Rolling Z-Score"]),
                         "noise", cfg, d)
            data = build_xlsx(d)
            import io
            import zipfile
            z = zipfile.ZipFile(io.BytesIO(data))          # xlsx bir zip'tir; ek paket gerekmez
            workbook = z.read("xl/workbook.xml").decode("utf-8")
            for name in ("Okuma rehberi", "Benchmark özet", "Benchmark detay", "Gürültü özet", "Gürültü detay"):
                self.assertIn(f'name="{name}"', workbook)
            strings = z.read("xl/sharedStrings.xml").decode("utf-8")
            self.assertIn("Olay yakalama oranı", strings)
            self.assertNotIn("<t>config</t>", strings)     # teknik sütunlar gizlenmiş olmalı


class DataTest(unittest.TestCase):
    def test_clean_frame_fixes_and_reports(self):
        t = pd.date_range("2024-01-01", periods=300, freq="5min").astype(str).tolist()
        raw = pd.DataFrame({"timestamp": t, "value": np.arange(300.0)})
        raw = pd.concat([raw, raw.iloc[[10]]], ignore_index=True)             # tekrarlı zaman damgası
        raw.loc[5, "timestamp"], raw.loc[6, "timestamp"] = raw.loc[6, "timestamp"], raw.loc[5, "timestamp"]  # sıra bozuk
        raw["value"] = raw["value"].astype(object)
        raw.loc[20, "value"] = "abc"                                          # sayı değil
        df, q = clean_frame(raw)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)
        self.assertFalse(df["timestamp"].duplicated().any())
        self.assertEqual(q["duplicate_timestamps"], 1)
        self.assertEqual(q["missing_or_non_numeric_values"], 1)
        self.assertGreaterEqual(q["backward_time_steps"], 1)

    def test_missing_columns_rejected(self):
        with self.assertRaises(DataFormatError):
            clean_frame(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    def test_too_few_rows_rejected(self):
        with self.assertRaises(DataFormatError):
            clean_frame(pd.DataFrame({"timestamp": ["2024-01-01"], "value": [1.0]}))


class ModelCausalityTest(unittest.TestCase):
    """score(values)[t] yalnızca values[:t+1]'e bağlı olmalı (gelecek geçmişe sızmamalı)."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = np.cumsum(rng.normal(size=1500)) * 0.1 + np.sin(np.arange(1500) / 20)
        self.cfg = Config(ae_epochs=3)

    def _check(self, name):
        model = make_model(name, self.cfg).fit(self.x[:600])
        k = 1000
        changed = self.x.copy()
        changed[k:] += 50.0     # gelecekteki değerleri bozuyoruz
        a, b = model.score(self.x)[:k], model.score(changed)[:k]
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, err_msg=f"{name}: gelecek veri geçmiş skoru değiştirdi")

    def test_zscore(self):
        self._check("Rolling Z-Score")

    def test_isolation_forest(self):
        self._check("Isolation Forest")

    @unittest.skipUnless(HAS_TORCH, "PyTorch kurulu değil")
    def test_autoencoder(self):
        self._check("Autoencoder")

    @unittest.skipUnless(HAS_TORCH, "PyTorch kurulu değil")
    def test_autoencoder_learns(self):
        model = make_model("Autoencoder", Config(ae_epochs=15)).fit(self.x[:600])
        h = model.loss_history_
        self.assertEqual(len(h), 15)
        self.assertLess(h[-1], h[0])


class PipelineTest(unittest.TestCase):
    def test_split_is_temporal_and_ordered(self):
        a, b = split_points(1000, Config())
        self.assertEqual((a, b), (400, 600))

    @unittest.skipUnless(list_nab_series(), "data/nab içinde seri yok")
    def test_threshold_does_not_depend_on_test_region(self):
        """Test bölümündeki değerleri değiştirmek eşiği değiştirmemeli (eşik test verisinden bağımsız)."""
        s = load_nab(list_nab_series()[0])
        cfg = Config()
        v = s.df["value"].to_numpy(float)
        _, val_end = split_points(len(v), cfg)
        names = ["Rolling Z-Score", "Isolation Forest"]
        fitted = fit_models(v, cfg, names)
        r1 = evaluate_fitted(fitted, v, s.df["timestamp"], s.windows, cfg)
        v2 = v.copy()
        v2[val_end:] += 10 * v.std()
        r2 = evaluate_fitted(fitted, v2, s.df["timestamp"], s.windows, cfg)
        for n in names:
            self.assertAlmostEqual(r1[n].threshold, r2[n].threshold, places=9)

    @unittest.skipUnless(list_nab_series(), "data/nab içinde seri yok")
    def test_all_models_registered_and_nab_loads(self):
        self.assertEqual(len(MODEL_NAMES), 3)
        s = load_nab(list_nab_series()[0])
        self.assertTrue(s.df["timestamp"].is_monotonic_increasing)
        self.assertGreater(len(s.windows), 0)

class UncertaintyTest(unittest.TestCase):
    def _bench(self):
        import pandas as pd
        rows = []
        for i in range(6):
            for m, det in (("A", 2), ("B", 0)):
                rows.append({"series": f"s{i}", "model": m, "events": 2, "events_detected": det, "f1": 0.5 if m == "A" else 0.1,
                             "pr_auc": 0.5 if m == "A" else 0.1, "false_alarms_per_day": 0.2, "false_alarm_point_rate": 0.01})
        return pd.DataFrame(rows)

    def test_bootstrap_separates_clear_difference(self):
        from alertscore.experiments import bootstrap_ci
        ci, diff = bootstrap_ci(self._bench(), n_boot=200)
        row = diff[diff["metric"] == "event_recall_pooled"].iloc[0]
        self.assertAlmostEqual(row["difference_a_minus_b"], 1.0)
        self.assertTrue(row["distinguishable"])
        same = diff[diff["metric"] == "false_alarms_per_day"].iloc[0]
        self.assertFalse(same["distinguishable"])
        self.assertTrue((ci["ci_low"] <= ci["value"] + 1e-9).all() and (ci["value"] <= ci["ci_high"] + 1e-9).all())

    def test_seed_summary_std_zero_for_identical_seeds(self):
        import pandas as pd
        from alertscore.experiments import seed_summary
        b = self._bench()
        df = pd.concat([b.assign(seed=0), b.assign(seed=1)], ignore_index=True)
        out = seed_summary(df)
        self.assertTrue((out["f1_std"] == 0).all())

class SkabAdapterTest(unittest.TestCase):
    def test_label_runs(self):
        from alertscore.data import label_runs
        ts = pd.Series(pd.date_range("2020-01-01", periods=10, freq="s"))
        w = label_runs(ts, [0, 1, 1, 0, 0, 1, 0, 0, 1, 1])
        self.assertEqual(len(w), 3)
        self.assertEqual(w[0], (ts[1], ts[2]))
        self.assertEqual(w[2], (ts[8], ts[9]))   # dosya sonunda biten pencere

    def test_load_skab_semicolon_file(self):
        import tempfile
        from pathlib import Path
        from alertscore.data import load_skab
        n = 300
        d = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=n, freq="s").astype(str),
                          "Volume Flow RateRMS": np.linspace(30, 31, n), "anomaly": [0] * 150 + [1] * 100 + [0] * 50})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "valve1").mkdir()
            d.to_csv(Path(tmp) / "valve1" / "1.csv", sep=";", index=False)
            s = load_skab("valve1/1.csv", skab_dir=tmp)
        self.assertEqual(len(s.windows), 1)
        self.assertEqual(s.quality["label_points"], 100)
        self.assertTrue(s.has_labels)


if __name__ == "__main__":
    unittest.main()
