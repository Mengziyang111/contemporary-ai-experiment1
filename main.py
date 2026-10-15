"""可复现的新闻十分类实验；验证集选型，测试集只用于最终预测。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 42
VALID_SIZE = 0.2
MLP_MAX_EPOCHS = 60
MLP_PATIENCE = 8
TFIDF = dict(lowercase=True, stop_words="english", min_df=2, max_df=0.95,
             max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
             dtype=np.float32)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(data_dir):
    train = pd.read_csv(data_dir / "train_data.csv")
    test = pd.read_csv(data_dir / "test_data_unlabeled.csv")
    if not {"text", "target"}.issubset(train.columns) or "text" not in test:
        raise ValueError("训练数据必须含 text/target，测试数据必须含 text。")
    if train[["text", "target"]].isna().any().any() or test.text.isna().any():
        raise ValueError("发现缺失文本或标签，请先确定处理方法，不能静默删除测试行。")
    if train.text.str.strip().eq("").any() or test.text.str.strip().eq("").any():
        raise ValueError("发现空文本，请检查数据。")
    if not np.array_equal(np.sort(train.target.unique()), np.arange(10)):
        raise ValueError("预期标签为整数 0 至 9。")
    return train, test


def metrics(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def mlp_model(width):
    return MLPClassifier(hidden_layer_sizes=(width,), activation="relu", solver="adam",
                         alpha=0.001, batch_size=128, learning_rate_init=0.001,
                         random_state=SEED, shuffle=True, early_stopping=False)


def fit_mlp(width, xtrain, ytrain, xvalid, yvalid, output):
    """partial_fit 每次恰好一轮，保留 Adam 状态；按验证交叉熵选检查点。"""
    model = mlp_model(width)
    classes = np.unique(ytrain)
    history = []
    best_loss, best_model, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        model.partial_fit(xtrain, ytrain, classes=classes)
        train_prob = model.predict_proba(xtrain)
        valid_prob = model.predict_proba(xvalid)
        train_ce = float(log_loss(ytrain, train_prob, labels=classes))
        valid_ce = float(log_loss(yvalid, valid_prob, labels=classes))
        history.append({"epoch": epoch, "optimizer_loss": float(model.loss_),
                        "train_cross_entropy": train_ce, "validation_cross_entropy": valid_ce,
                        "train_macro_f1": metrics(ytrain, classes[train_prob.argmax(1)])["macro_f1"],
                        "validation_macro_f1": metrics(yvalid, classes[valid_prob.argmax(1)])["macro_f1"]})
        # min_delta 为 1e-4；耐心值只由验证集损失决定，不读取测试数据。
        if valid_ce < best_loss - 1e-4:
            best_loss, best_epoch, stale = valid_ce, epoch, 0
            best_model = copy.deepcopy(model)
        else:
            stale += 1
        pd.DataFrame(history).to_csv(output / f"mlp_{width}_loss.csv", index=False)
        print(f"MLP {width:3d} epoch {epoch:02d}: train CE={train_ce:.4f}, "
              f"valid CE={valid_ce:.4f}, valid F1={history[-1]['validation_macro_f1']:.4f}", flush=True)
        if stale >= MLP_PATIENCE:
            break
    return best_model, best_epoch, len(history)


def plot_losses(output, widths):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.1), layout="constrained")
    for ax, width in zip(axes, widths):
        h = pd.read_csv(output / f"mlp_{width}_loss.csv")
        ax.plot(h.epoch, h.train_cross_entropy, label="Train cross-entropy", color="#197a93", lw=1.8)
        ax.plot(h.epoch, h.validation_cross_entropy, label="Validation cross-entropy", color="#d56539", lw=1.8)
        ax.plot(h.epoch, h.optimizer_loss, label="Mini-batch objective (+ L2)", color="#83929b", ls="--", lw=1)
        results = pd.read_csv(output / "validation_results.csv")
        best = int(results[(results.model == "MLP") & (results.parameter == f"hidden={width}")].iloc[0].best_epoch)
        ax.axvline(best, color="#555555", ls=":", lw=1)
        ax.set(title=f"MLP: {width} hidden units", xlabel="Epoch", ylabel="Loss")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7.5)
    fig.savefig(output / "loss_curves.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = ROOT / "data"
    if not default_data.exists():
        default_data = ROOT.parent / "data-Project1" / "data-Project1"
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    train, test = load_data(args.data_dir)
    indices = np.arange(len(train))
    ti, vi = train_test_split(indices, test_size=VALID_SIZE, random_state=SEED,
                             stratify=train.target.to_numpy())
    pd.DataFrame({"row_index": indices,
                  "split": np.where(np.isin(indices, vi), "validation", "train")}).to_csv(
                      output / "split_indices.csv", index=False)
    ytrain, yvalid = train.target.to_numpy()[ti], train.target.to_numpy()[vi]
    text_train, text_valid = train.text.iloc[ti], train.text.iloc[vi]
    feature_start = time.perf_counter()
    vectorizer = TfidfVectorizer(**TFIDF)
    xtrain = vectorizer.fit_transform(text_train)
    xvalid = vectorizer.transform(text_valid)
    feature_seconds = time.perf_counter() - feature_start
    print(f"Samples: labeled={len(train)}, test={len(test)}, train={len(ti)}, validation={len(vi)}", flush=True)
    print(f"TF-IDF: {xtrain.shape}, dtype={xtrain.dtype}, shared fit/transform {feature_seconds:.2f}s", flush=True)
    records, fitted = [], {}
    warning_records = []

    def record(model, name, param, feature="word1+2", best_epoch=None, epochs_run=None,
               train_x=xtrain, valid_x=xvalid, seconds=0.0):
        pred = model.predict(valid_x)
        train_metrics = metrics(ytrain, model.predict(train_x))
        valid_metrics = metrics(yvalid, pred)
        row = {"model": name, "parameter": param, "feature": feature,
               "train_accuracy": train_metrics["accuracy"], "train_macro_f1": train_metrics["macro_f1"],
               "validation_accuracy": valid_metrics["accuracy"], "validation_macro_f1": valid_metrics["macro_f1"],
               "fit_seconds": seconds, "best_epoch": best_epoch, "epochs_run": epochs_run}
        if hasattr(model, "predict_proba"):
            row["validation_cross_entropy"] = float(log_loss(yvalid, model.predict_proba(valid_x), labels=np.arange(10)))
        records.append(row)
        fitted[len(records) - 1] = (model, pred, vectorizer if feature == "word1+2" else unigram_vectorizer)
        pd.DataFrame(records).to_csv(output / "validation_results.csv", index=False)
        print(f"{name} {param} [{feature}]: accuracy={valid_metrics['accuracy']:.4f}, "
              f"macro-F1={valid_metrics['macro_f1']:.4f}, fit={seconds:.2f}s", flush=True)

    candidates = [("NB", "alpha", [0.1, 0.5, 1.0], MultinomialNB()),
                  ("SVM", "C", [0.1, 1.0, 10.0], SVC(kernel="linear", random_state=SEED)),
                  ("LR", "C", [0.1, 1.0, 10.0], LogisticRegression(max_iter=2000, random_state=SEED))]
    for name, parameter, values, base_model in candidates:
        for value in values:
            model = clone(base_model).set_params(**{parameter: value})
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(xtrain, ytrain)
            seconds = time.perf_counter() - started
            warning_records.extend({"model": name, "parameter": f"{parameter}={value:g}",
                                    "category": w.category.__name__, "message": str(w.message)} for w in caught)
            record(model, name, f"{parameter}={value:g}", seconds=seconds)

    for width in [50, 100]:
        started = time.perf_counter()
        model, best_epoch, epochs_run = fit_mlp(width, xtrain, ytrain, xvalid, yvalid, output)
        record(model, "MLP", f"hidden={width}", best_epoch=best_epoch, epochs_run=epochs_run,
               seconds=time.perf_counter() - started)

    # 控制消融：固定 LR C=10，只改变 ngram_range；词表仍仅拟合训练部分。
    unigram_vectorizer = TfidfVectorizer(**{**TFIDF, "ngram_range": (1, 1)})
    unigram_train = unigram_vectorizer.fit_transform(text_train)
    unigram_valid = unigram_vectorizer.transform(text_valid)
    unigram_lr = LogisticRegression(C=10, max_iter=2000, random_state=SEED)
    started = time.perf_counter()
    unigram_lr.fit(unigram_train, ytrain)
    record(unigram_lr, "LR", "C=10", feature="word1", train_x=unigram_train,
           valid_x=unigram_valid, seconds=time.perf_counter() - started)

    results = pd.DataFrame(records)
    # 预先指定：Macro-F1 优先，Accuracy 次之，再比较拟合时间。
    ranking = results.sort_values(["validation_macro_f1", "validation_accuracy", "fit_seconds"],
                                  ascending=[False, False, True])
    winner_index = int(ranking.index[0])
    winner = records[winner_index]
    best, pred, selected_vectorizer = fitted[winner_index]
    report = classification_report(yvalid, pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
    cm = confusion_matrix(yvalid, pred, labels=np.arange(10))
    pd.DataFrame(cm).to_csv(output / "confusion_matrix.csv", index=False)
    validation_predictions = pd.DataFrame({"row_index": vi, "target": yvalid, "prediction": pred})
    validation_predictions.to_csv(output / "validation_predictions.csv", index=False)
    errors = validation_predictions[validation_predictions.target != validation_predictions.prediction].copy()
    errors["excerpt"] = [train.text.iloc[i][:1000] for i in errors.row_index]
    # 误分类文本只本地保留；发布的仓库不包含原始新闻、邮箱或正文片段。
    errors.to_csv(output / "error_examples_private.csv", index=False)

    plot_losses(output, [50, 100])
    fig, ax = plt.subplots(figsize=(5.2, 4.3), layout="constrained")
    ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Validation confusion matrix")
    fig.savefig(output / "confusion_matrix.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    labels = [f"{r['model']} {r['parameter']}\n{r['feature']}" for r in records]
    ax.bar(np.arange(len(records)), results.validation_macro_f1, color="#197a93")
    ax.set(xticks=np.arange(len(records)), xticklabels=labels, ylabel="Validation Macro-F1", ylim=(0.78, 1))
    ax.tick_params(axis="x", labelrotation=55, labelsize=7)
    ax.grid(axis="y", alpha=0.15)
    fig.savefig(output / "model_comparison.png", dpi=220)
    plt.close(fig)

    # 方案锁定后，在全部有标签数据上重拟合词表与模型。
    print("Selected:", winner, flush=True)
    final_vectorizer = clone(selected_vectorizer)
    xall = final_vectorizer.fit_transform(train.text)
    xtest = final_vectorizer.transform(test.text)
    final_model = clone(best)
    if winner["model"] == "MLP":
        for _ in range(int(winner["best_epoch"])):
            final_model.partial_fit(xall, train.target.to_numpy(), classes=np.arange(10))
    else:
        final_model.fit(xall, train.target)
    predictions = final_model.predict(xtest)
    prediction_path = output / "prediction.csv"
    pd.DataFrame(predictions).to_csv(prediction_path, index=False, header=False)
    assert len(predictions) == len(test) and set(predictions).issubset(set(range(10)))
    off_diagonal = [(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j and cm[i, j]]
    off_diagonal.sort(reverse=True)
    metadata = {
        "seed": SEED, "validation_fraction": VALID_SIZE, "train_samples": len(ti),
        "validation_samples": len(vi), "labeled_samples": len(train), "test_samples": len(test),
        "class_counts": {str(k): int(v) for k, v in train.target.value_counts().sort_index().items()},
        "data_checks": {"missing_values": 0, "empty_texts": 0, "duplicate_train_texts": int(train.text.duplicated().sum()),
                        "overlapping_train_validation_texts": len(set(text_train) & set(text_valid))},
        "data_sha256": {f: sha256(args.data_dir / f) for f in ["train_data.csv", "test_data_unlabeled.csv"]},
        "tfidf": {k: str(v) if k == "dtype" else v for k, v in TFIDF.items()},
        "shared_feature_fit_transform_seconds": feature_seconds,
        "selection_rule": "validation Macro-F1, then accuracy, then lower fit time",
        "best_configuration": winner, "mlp_max_epochs": MLP_MAX_EPOCHS, "mlp_patience": MLP_PATIENCE,
        "most_common_confusions": [{"count": n, "true": i, "predicted": j} for n, i, j in off_diagonal[:5]],
        "warnings": warning_records, "total_seconds": time.perf_counter() - started_all,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                        "platform": platform.platform(), "processor": platform.processor(),
                        "cpu_count": os.cpu_count(), "blas_threads": 1},
        "prediction_sha256": sha256(prediction_path),
    }
    write_json(output / "run_metadata.json", metadata)
    print(ranking.to_string(index=False), flush=True)
    print(f"Saved {len(predictions)} predictions. Total {metadata['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
