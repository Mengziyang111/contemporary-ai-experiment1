"""检查两页 PDF、单列预测与 ZIP 根目录结构；只打包指定的两个文件。"""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "results" / "run_metadata.json").read_text(encoding="utf-8"))
    prediction_path = ROOT / "results" / "prediction.csv"
    lines = prediction_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == metadata["test_samples"] == 2457, "预测行数必须匹配测试集。"
    assert all(re.fullmatch(r"[0-9]", line) for line in lines), "预测必须单列、无表头、标签为 0-9。"
    pred = pd.read_csv(prediction_path, header=None)
    assert pred.shape == (2457, 1)
    with pymupdf.open(args.pdf) as doc:
        assert len(doc) == 2, "报告必须恰好两页。"
        first_page = doc[0].get_text()
        all_text = "\n".join(page.get_text() for page in doc)
        assert "孟子阳" in first_page and "10245501476" in first_page
        assert "GitHub" in first_page[:260], "GitHub 链接必须在报告开头。"
        if not args.allow_draft:
            assert "https://github.com/" in first_page[:260]
            assert not any(text in all_text for text in ["待恢复", "待填写", "审阅稿", "请勿提交"])
        assert sum(len(page.get_images()) for page in doc) >= 2, "报告缺少损失曲线/混淆矩阵。"
        assert "交叉熵" in all_text and "问题解决" in all_text
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.pdf, "实验报告.pdf")
        archive.write(prediction_path, "prediction.csv")
    with zipfile.ZipFile(args.zip) as archive:
        assert archive.namelist() == ["实验报告.pdf", "prediction.csv"]
        assert archive.testzip() is None
        assert archive.read("prediction.csv") == prediction_path.read_bytes()
        assert archive.read("实验报告.pdf") == args.pdf.read_bytes()
    result = {"zip": str(args.zip.resolve()), "members": ["实验报告.pdf", "prediction.csv"],
              "pdf_pages": 2, "prediction_rows": 2457, "draft": args.allow_draft,
              "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()}
    args.zip.with_suffix(".checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
