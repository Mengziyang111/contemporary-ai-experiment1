"""独立复核划分、验证指标、混淆矩阵、曲线和最终预测；可重训最终方案。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data-Project1" / "data-Project1")
    parser.add_argument("--refit-final", action="store_true")
    args = parser.parse_args()
    result = ROOT / "results"
    meta = json.loads((result / "run_metadata.json").read_text(encoding="utf-8"))
    train = pd.read_csv(args.data_dir / "train_data.csv")
    test = pd.read_csv(args.data_dir / "test_data_unlabeled.csv")
    for name, digest in meta["data_sha256"].items():
        assert hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() == digest
    split = pd.read_csv(result / "split_indices.csv")
    assert len(split) == len(train) and split.row_index.is_unique
    assert set(split.row_index) == set(range(len(train)))
    ti = split.loc[split.split == "train", "row_index"].to_numpy()
    vi = split.loc[split.split == "validation", "row_index"].to_numpy()
    assert len(ti) == 5894 and len(vi) == 1474
    assert set(ti).isdisjoint(set(vi))
    assert not (set(train.text.iloc[ti]) & set(train.text.iloc[vi]))
    vp = pd.read_csv(result / "validation_predictions.csv")
    assert set(vp.row_index) == set(vi) and vp.row_index.is_unique
    np.testing.assert_array_equal(vp.target, train.target.iloc[vp.row_index])
    best = meta["best_configuration"]
    np.testing.assert_allclose(accuracy_score(vp.target, vp.prediction), best["validation_accuracy"], atol=1e-14)
    np.testing.assert_allclose(f1_score(vp.target, vp.prediction, average="macro"), best["validation_macro_f1"], atol=1e-14)
    cm = confusion_matrix(vp.target, vp.prediction, labels=np.arange(10))
    np.testing.assert_array_equal(cm, pd.read_csv(result / "confusion_matrix.csv").to_numpy())
    assert cm.sum() == 1474
    rows = pd.read_csv(result / "validation_results.csv")
    assert len(rows) == 12 and set(rows.model) == {"NB", "SVM", "LR", "MLP"}
    np.testing.assert_allclose(rows.validation_macro_f1.max(), best["validation_macro_f1"], atol=1e-14)
    for width in [50, 100]:
        h = pd.read_csv(result / f"mlp_{width}_loss.csv")
        assert np.isfinite(h.to_numpy()).all()
        assert (h[["optimizer_loss", "train_cross_entropy", "validation_cross_entropy"]] >= 0).all().all()
        np.testing.assert_array_equal(h.epoch, np.arange(1, len(h) + 1))
    prediction_path = result / "prediction.csv"
    pred = pd.read_csv(prediction_path, header=None).to_numpy().ravel()
    assert len(pred) == len(test) == 2457
    assert set(pred).issubset(set(range(10)))
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == meta["prediction_sha256"]
    if args.refit_final:
        # 使用独立构建的 Pipeline 复核本次最终方案，不重跑整个参数搜索。
        assert best["model"] == "LR" and best["feature"] == "word1" and best["parameter"] == "C=10"
        config = {**meta["tfidf"], "dtype": np.float32, "ngram_range": (1, 1)}
        model = make_pipeline(TfidfVectorizer(**config), LogisticRegression(C=10, max_iter=2000, random_state=meta["seed"]))
        with threadpool_limits(limits=1):
            model.fit(train.text, train.target)
            np.testing.assert_array_equal(pred, model.predict(test.text))
    check = {"split_verified": True, "metrics_recomputed": True, "confusion_matrix_verified": True,
             "finite_loss_curves": True, "prediction_rows": len(pred), "final_refit_matches": args.refit_final,
             "prediction_sha256": meta["prediction_sha256"]}
    (result / "verification.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(json.dumps(check))


if __name__ == "__main__":
    main()
