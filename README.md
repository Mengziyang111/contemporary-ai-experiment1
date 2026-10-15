# 当代人工智能实验一：新闻文本十分类

姓名：孟子阳　学号：10245501476

本实验比较 TF-IDF 上的多项式朴素贝叶斯、线性 SVM、逻辑回归和 MLP，包含 11 组模型参数配置与 1 组 unigram 特征对照。报告使用实际运行结果和真实损失曲线，未使用无标签测试集选模型。

## 运行实验

实测环境为 Windows、Python 3.13.9，依赖版本见 `requirements.txt`。可在独立虚拟环境中安装：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

将教师提供的两个原始文件放入 `data/`：

```text
data/train_data.csv             # text、target 列
data/test_data_unlabeled.csv    # text 列
```

运行：

```powershell
python -u main.py --data-dir data --output-dir results
python verify_results.py --data-dir data --refit-final
```

本地原工作区可直接运行 `python -u main.py`，默认会查找相邻的 `data-Project1/data-Project1`。CSV 原始数据不公开上传；复现者需使用课程提供的数据。准确数据校验值记录在 `results/run_metadata.json`。处理不删除测试样本，始终保持原顺序。

本次完整运行约 474 秒；不同硬件耗时可能不同。代码限制 BLAS 为单线程，固定随机种子 42。不同软件、硬件仍可能产生微小数值差异。

## 实验设计

- 有标签数据 7,368 条、测试数据 2,457 条，标签 0-9。
- 训练/验证按类别分层划分为 5,894/1,474 条；`test_size=0.2`、`random_state=42`。
- 数据完整性检查发现无缺失、空文本、训练文件内的重复文本，训练部分与验证部分无完全相同文本。不在划分前拟合词表或 IDF。
- 共享特征：小写、英文停用词、`min_df=2`、`max_df=0.95`、`max_features=20000`、`ngram_range=(1,2)`、`sublinear_tf=True`，默认 L2 归一化，使用 float32 稀疏矩阵。
- NB：`alpha=[0.1,0.5,1]`；线性核 `SVC`：`C=[0.1,1,10]`；LR：`C=[0.1,1,10]`、`max_iter=2000`。
- MLP：隐藏层 50/100，ReLU、Adam、`learning_rate_init=0.001`、`batch_size=128`、`alpha=0.001`。`partial_fit` 每次训练一轮并保持优化器状态，最多 60 轮；验证交叉熵改进须超过 0.0001，耐心值 8；保存并恢复检查点。本次两个配置均运行了 60 轮，检查点分别为第 60/59 轮，没有触发提前停止。
- 特征对照：固定 LR 的 `C=10` 和其他设置，仅把 `(1,2)` 改为 `(1,1)`，在各自训练文本上重新拟合向量化器。固定词表上限使词组与单词竞争名额，这也是对照的解释边界。
- 预先用验证 Macro-F1 选型，相同时依次比较 Accuracy 和训练阶段耗时。MLP 内部检查点按验证交叉熵选择；此准则与模型之间的 F1 选择不同，未事后挑选 F1 最高轮次。
- 表中 `fit_seconds` 对传统模型是拟合时间，对 MLP 包含逐轮训练/验证评估、记录及检查点；均不含共享特征提取和最终预测。不能把它解释为完全相同计时口径下的纯优化器速度。

## 本次实际结果

| 配置 | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|
| NB，alpha=0.1，unigram+bigram | 0.9071 | 0.9081 |
| 线性 SVM，C=10，unigram+bigram | 0.9322 | 0.9327 |
| LR，C=10，unigram+bigram | 0.9383 | 0.9384 |
| MLP，50 单元，第 60 轮 | 0.9362 | 0.9364 |
| MLP，100 单元，第 59 轮 | 0.9383 | 0.9385 |
| **LR，C=10，仅 unigram（最终方案）** | **0.9423** | **0.9425** |

完整的 12 行结果在 `results/validation_results.csv`。报告中的实验分析包括：SVM 过强正则约束、MLP 损失与分类指标的不同变化、固定维度预算下词组特征的负面结果、混淆类别及单次验证划分的局限。验证成绩已用于模型选择，不是独立测试成绩。

最终按最佳配置在全部有标签数据上重新拟合 TF-IDF 和 LR，再生成测试预测。输出 `results/prediction.csv` 恰好 2,457 行、单列、无表头和索引。未将模型的训练准确率当作测试准确率。

## 输出与损失曲线

- `results/validation_results.csv`：全部模型参数和特征对比，含训练/验证指标。
- `results/mlp_50_loss.csv`、`mlp_100_loss.csv`：逐轮优化器损失、完整训练/验证交叉熵和 Macro-F1。
- `results/loss_curves.png`：两组 MLP 的真实损失图；竖线标记按改进阈值选定的检查点。
- `results/confusion_matrix.png/csv`、`classification_report.csv`：最佳模型的验证集逐类分析。
- `results/model_comparison.png`：参数对比图。
- `results/split_indices.csv`、`validation_predictions.csv`：可独立复查划分和指标。
- `results/run_metadata.json`：数据 SHA-256、版本、配置、类别计数和预测校验值。
- `results/error_examples_private.csv`：误分类正文片段，仅本地检查使用，已通过 `.gitignore` 排除，不发布到仓库。

多类交叉熵为 `-(1/N) * sum(log(p(y_i|x_i)))`。图中训练/验证实线均在每轮训练结束后计算；优化器记录的灰虚线是小批量的交叉熵加 L2 正则目标，计算时点及正则项不同，不应直接当作完整训练集交叉熵。所有曲线均来自真实训练，没有用 Accuracy 或 F1 代替损失。

## 生成两页报告和提交 ZIP

```powershell
python -m pip install -r requirements-report.txt
python build_report.py --repo-url https://github.com/ziyangMeng666/contemporary-ai-experiment1 --output ../output/pdf/实验报告.pdf
python package_submission.py --pdf ../output/pdf/实验报告.pdf --zip ../output/submission/10245501476_孟子阳_实验一.zip
```

`build_report.py` 默认使用 Windows 微软雅黑；其他系统通过 `--font` 和 `--bold-font` 指定中文 TrueType 字体。报告会根据当前 results 文件重新生成。仓库链接必须指向已真实发布、教师可以访问的 GitHub 仓库；不传 `--repo-url` 只生成明确标注“请勿提交”的审阅稿。

`package_submission.py` 验证 PDF 恰好两页、开头有链接、署名正确，预测数量/标签合法，并确保压缩包根目录只有：

```text
实验报告.pdf
prediction.csv
```

PDF 应先渲染为图片逐页检查布局后再打包。课程提交入口：https://send2me.cn/dmu92JW7/SlOKWOpDfaSCYQ 。

## 参考

- 教师提供的实验一要求和 TUNING.md。
- [scikit-learn MLPClassifier 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)：交叉熵、L2 正则、逐轮 `partial_fit`。
- [TfidfVectorizer 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)。
