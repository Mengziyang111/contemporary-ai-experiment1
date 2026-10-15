"""根据本次实际运行结果生成两页报告；没有已发布仓库时只生成审阅稿。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT.parent / "tmp" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default="")
    parser.add_argument("--font", default="C:/Windows/Fonts/msyh.ttc")
    parser.add_argument("--bold-font", default="C:/Windows/Fonts/msyhbd.ttc")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "output" / "pdf" / "实验报告_审阅稿.pdf")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("CJK", args.font, subfontIndex=0))
    pdfmetrics.registerFont(TTFont("CJKB", args.bold_font, subfontIndex=0))
    pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJKB", italic="CJK", boldItalic="CJKB")
    meta = json.loads((ROOT / "results" / "run_metadata.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(ROOT / "results" / "validation_results.csv")
    history = {w: pd.read_csv(ROOT / "results" / f"mlp_{w}_loss.csv") for w in [50, 100]}
    perclass = pd.read_csv(ROOT / "results" / "classification_report.csv", index_col=0)
    best = meta["best_configuration"]
    lr = rows[(rows.model == "LR") & (rows.feature == "word1+2") & (rows.parameter == "C=10")].iloc[0]
    uni = rows[rows.feature == "word1"].iloc[0]
    svm_lo = rows[(rows.model == "SVM") & (rows.parameter == "C=0.1")].iloc[0]
    svm_hi = rows[(rows.model == "SVM") & (rows.parameter == "C=10")].iloc[0]
    best_mlp = rows[rows.model == "MLP"].sort_values("validation_macro_f1", ascending=False).iloc[0]
    width = int(best_mlp.parameter.split("=")[1])
    h = history[width]
    selected = h[h.epoch == int(best_mlp.best_epoch)].iloc[0]
    errors_count = int(round((1 - best["validation_accuracy"]) * meta["validation_samples"]))
    class_only = perclass.loc[[str(i) for i in range(10)]]
    worst_class = class_only["f1-score"].idxmin()

    width_page, height_page = A4
    margin = 37
    content = width_page - margin * 2
    ink = colors.HexColor("#172d3d")
    teal = colors.HexColor("#176f82")
    gray = colors.HexColor("#50606d")
    style = ParagraphStyle("body", fontName="CJK", fontSize=9.4, leading=14.3,
                           textColor=ink, wordWrap="CJK", spaceAfter=0)
    small = ParagraphStyle("small", parent=style, fontSize=8, leading=11.5, textColor=gray)
    c = canvas.Canvas(str(args.output), pagesize=A4)
    c.setTitle("实验一：基于 TF-IDF 的新闻文本分类 - 孟子阳 10245501476")
    c.setAuthor("孟子阳")
    c.setSubject("算法与参数对比、训练损失、问题解决及思考过程；两页实验报告")

    def para(text, y, x=margin, available=content, chosen=style):
        p = Paragraph(text, chosen)
        _, ht = p.wrap(available, 1000)
        if y - ht < 38:
            raise RuntimeError(f"页面内容溢出: {text[:60]}")
        p.drawOn(c, x, y - ht)
        return y - ht

    def section(number, title, y):
        y -= 12
        c.setFillColor(teal)
        c.roundRect(margin, y - 12, 3, 12, 1, fill=1, stroke=0)
        c.setFont("CJKB", 10.8)
        c.drawString(margin + 10, y - 10, f"{number}  {title}")
        return y - 21

    def footer(page):
        c.setStrokeColor(colors.HexColor("#d7e0e6"))
        c.line(margin, 31, width_page - margin, 31)
        c.setFont("CJK", 7.3)
        c.setFillColor(gray)
        c.drawString(margin, 19, "当代人工智能 · 实验一    |    孟子阳 10245501476")
        c.drawRightString(width_page - margin, 19, f"{page} / 2")

    y = height_page - 31
    if args.repo_url:
        url = escape(args.repo_url)
        y = para(f'<b>GitHub 仓库：</b><link href="{url}" color="#176f82">{url}</link>', y,
                 chosen=ParagraphStyle("repo", parent=style, fontSize=8.9, leading=13))
    else:
        y = para('<b>GitHub 仓库：待恢复登录后创建并填写（此文件为审阅稿，请勿提交）</b>', y,
                 chosen=ParagraphStyle("pending", parent=style, textColor=colors.HexColor("#aa402a")))
    y -= 9
    c.setFont("CJKB", 17)
    c.setFillColor(ink)
    c.drawString(margin, y - 17, "实验一：基于 TF-IDF 的新闻文本分类")
    y -= 27
    y = para("姓名：孟子阳　学号：10245501476　任务：新闻十分类", y, chosen=small)

    y = section("1", "实验目标、数据与评价", y)
    y = para("问题：在相同文本表示下，线性模型与神经网络如何权衡分类效果和计算成本？"
             "假设：高维稀疏 TF-IDF 已提供较强区分能力，线性模型可形成有效基线；MLP 增加容量可能改善拟合，也可能过拟合。", y)
    y = para(f"有标签数据 {meta['labeled_samples']:,} 条、无标签测试数据 {meta['test_samples']:,} 条，标签为 0-9，"
             "各类 705-747 条。检查发现无缺失、空文本及训练文件内的重复文本。固定种子 42，分层划分 "
             f"{meta['train_samples']:,}/{meta['validation_samples']:,} 条（80%/20%），两部分无完全相同文本。"
             "预先以验证 Macro-F1 选型，Accuracy 为次指标，同时记录拟合耗时；测试数据不参与调参。", y)

    y = section("2", "模型原理、特征与参数对比", y)
    y = para("文本转小写、去英文停用词；TF-IDF 使用 min_df=2、max_df=0.95、最多 20,000 维、"
             "1-2 gram、次线性词频和 L2 归一化。先划分再拟合词表/IDF，所有模型共享同一训练特征。"
             "NB 用条件独立假设估计类别；线性 SVM 优化间隔；LR 优化多类交叉熵；MLP 使用 ReLU 隐藏层和 softmax 输出。", y)
    y = para("NB 调 alpha；SVM/LR 调 C（越大正则约束越弱）；MLP 对比 50/100 个隐藏神经元，"
             "固定 Adam、学习率 0.001、batch=128、L2 alpha=0.001。最多 60 轮，验证交叉熵连续 8 轮"
             "未改善 0.0001 即停止并恢复最佳检查点。下表为训练阶段耗时：传统模型仅拟合，MLP 含每轮评估与检查点；不含共享特征提取及最终预测。", y)
    y -= 6

    data = [["算法", "参数 / 特征", "训练 F1", "验证 Acc", "验证 F1", "训练秒"]]
    for _, r in rows.iterrows():
        param = r.parameter
        if r.feature == "word1":
            param += " / unigram"
        elif r.model == "MLP":
            param += f" / e{int(r.best_epoch)}"
        data.append([r.model, param, f"{r.train_macro_f1:.4f}", f"{r.validation_accuracy:.4f}",
                     f"{r.validation_macro_f1:.4f}", f"{r.fit_seconds:.2f}"])
    table = Table(data, colWidths=[44, 165, 77, 79, 79, content - 444], rowHeights=[19] + [16.2] * len(rows))
    commands = [("FONTNAME", (0, 0), (-1, -1), "CJK"), ("FONTSIZE", (0, 0), (-1, -1), 8.1),
                ("FONTNAME", (0, 0), (-1, 0), "CJKB"), ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (2, 0), (-1, -1), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, teal),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f5f7"), colors.white])]
    for index, (_, r) in enumerate(rows.iterrows(), 1):
        if r.model == best["model"] and r.parameter == best["parameter"] and r.feature == best["feature"]:
            commands.extend([("FONTNAME", (0, index), (-1, index), "CJKB"),
                             ("TEXTCOLOR", (0, index), (-1, index), teal)])
    table.setStyle(TableStyle(commands))
    _, table_height = table.wrap(content, 1000)
    table.drawOn(c, margin, y - table_height)
    y -= table_height

    y = section("3", "实验过程、问题解决与思考", y)
    y = para("① 入门草稿直接用全部有标签数据拟合特征、缺少验证对比，无法可靠选型。"
             "先固定分层划分，再仅在训练部分拟合 TF-IDF；保存划分索引、数据 SHA-256、版本和配置，确保过程可复查。", y)
    y = para(f"② 原参考程序只有指标图，没有损失记录。增加 MLP 的逐轮 partial_fit，保持 Adam 状态，"
             "每轮保存完整训练/验证交叉熵及优化器损失，使用验证损失选检查点，避免把准确率冒充损失。", y)
    y = para(f"③ SVM 在 C=0.1 时训练/验证 F1 为 {svm_lo.train_macro_f1:.4f}/{svm_lo.validation_macro_f1:.4f}，"
             f"C=10 后验证 F1 达 {svm_hi.validation_macro_f1:.4f}，说明此处过强约束限制了拟合。"
             "随后用固定 C 的特征对照检验词组是否有效，避免同时改多个因素而无法解释差异。", y)
    footer(1)
    c.showPage()

    y = height_page - 33
    y = section("4", "损失函数、收敛与过拟合分析", y)
    y = para("多类交叉熵：L_CE = -(1/N) Σ_i log p(y_i | x_i)。图中实线为每轮结束后在完整训练集/验证集"
             "计算的交叉熵；灰虚线为含 L2 正则的小批量训练目标。后者计算时点和正则项不同，不应与实线数值直接等同；竖线标记按阈值选定的检查点。", y)
    image_height = content * 3.1 / 10.5
    c.drawImage(str(ROOT / "results" / "loss_curves.png"), margin, y - image_height,
                width=content, height=image_height, mask="auto")
    y -= image_height + 3
    vals = []
    for w, hist in history.items():
        row = rows[(rows.model == "MLP") & (rows.parameter == f"hidden={w}")].iloc[0]
        vals.append(f"{w} 单元：运行 {int(row.epochs_run)} 轮、恢复第 {int(row.best_epoch)} 轮")
    y = para("图 1　实际训练损失曲线。" + "；".join(vals) + "。", y, chosen=small)
    y = para(f"较优 MLP（{width} 单元）的训练交叉熵由 {h.iloc[0].train_cross_entropy:.3f} 降至"
             f"检查点的 {selected.train_cross_entropy:.3f}，对应验证交叉熵 {selected.validation_cross_entropy:.3f}。"
             f"运行末轮训练/验证交叉熵为 {h.iloc[-1].train_cross_entropy:.3f}/{h.iloc[-1].validation_cross_entropy:.3f}。"
             "训练损失持续降低不能证明泛化持续改善；验证损失停滞或反升时，应优先恢复检查点，而非机械增加训练轮数。", y)

    y = section("5", "模型选择、特征对照与错误分析", y)
    y = para(f"最终选择 {best['model']}（{best['parameter']}，{best['feature']}）：验证 Accuracy "
             f"{best['validation_accuracy']:.4f}、Macro-F1 {best['validation_macro_f1']:.4f}，"
             f"训练 F1 {best['train_macro_f1']:.4f}，错分 {errors_count}/{meta['validation_samples']} 条。"
             "这是用于调参的验证成绩，无标签测试集无法自行计算准确率。", y)
    delta = (lr.validation_macro_f1 - uni.validation_macro_f1) * 100
    direction = "提升" if delta >= 0 else "降低"
    y = para(f"固定 LR C=10、词表上限和其余设置：unigram 的验证 F1 为 {uni.validation_macro_f1:.4f}，"
             f"加入 bigram 后为 {lr.validation_macro_f1:.4f}，{direction} {abs(delta):.2f} 个百分点。"
             "该对照同时反映固定 20,000 维预算下词组与单词竞争特征名额的影响，不能推断词组在任何设置下都有效。", y)
    y -= 7
    image_w, image_h = 239, 198
    c.drawImage(str(ROOT / "results" / "confusion_matrix.png"), margin, y - image_h,
                width=image_w, height=image_h, preserveAspectRatio=True, anchor="c", mask="auto")
    right_x, right_w = margin + 253, content - 253
    pairs = meta["most_common_confusions"][:3]
    pair_text = "；".join(f"{p['true']}→{p['predicted']}（{p['count']} 条）" for p in pairs)
    ry = para("<b>图 2　最佳模型验证集混淆矩阵</b><br/>行是真实标签，列是预测标签。标签未提供语义映射，报告不擅自命名类别。",
              y - 3, x=right_x, available=right_w)
    ry = para(f"主要错分方向：{pair_text}。类别 {worst_class} 的 F1 最低，为 "
              f"{class_only.loc[worst_class, 'f1-score']:.4f}。总体分数之外仍需关注这些类别的混淆。",
              ry - 8, x=right_x, available=right_w)
    ry = para(f"MLP 的验证交叉熵为 {best_mlp.validation_cross_entropy:.3f}，低于单词 LR 的 {uni.validation_cross_entropy:.3f}，"
              "但 F1 较低：交叉熵衡量概率质量，F1 衡量最终分类，二者排序可不同。结合耗时，本次选择单词 LR。",
              ry - 8, x=right_x, available=right_w)
    y -= max(image_h, y - ry) + 3

    y = section("6", "结论、复现与局限", y)
    y = para("锁定配置后，在全部 7,368 条有标签数据上重新拟合 TF-IDF 和模型，再按原顺序预测 2,457 条测试新闻，"
             "输出单列、无表头、无索引的 prediction.csv。代码及 README 记录完整运行方式，原始数据不随仓库公开。", y)
    y = para("局限：只使用一次固定划分，验证成绩存在选择偏差，未估计多随机种子的波动；新闻头部、引用和相近主题"
             "可能影响迁移能力。可进一步做去元信息对照、多次分层验证及 BERT 表示实验；本次没有把这些设想写成已完成结果。", y)
    env = meta["environment"]
    y = para(f"运行环境：Python {env['python']} / scikit-learn {env['scikit_learn']} / NumPy {env['numpy']}，"
             "BLAS 单线程。MLP 训练方式参考 scikit-learn 官方 MLPClassifier 文档："
             '<link href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html" color="#176f82">scikit-learn.org · MLPClassifier</link>。',
             y - 4, chosen=small)
    footer(2)
    c.save()
    print(json.dumps({"pdf": str(args.output.resolve()), "pages": 2, "repository_link_present": bool(args.repo_url)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
