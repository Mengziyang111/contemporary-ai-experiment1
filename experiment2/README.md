# 实验二 搜索算法

孟子阳 · 10245501476

代码仓库：https://github.com/Mengziyang111/contemporary-ai-experiment1/tree/main/experiment2

完成课程 PDF 的全部四题，附实验报告、真实对照数据和独立正确性测试。
报告不超过 5 页，首页附 GitHub 代码链接；课程平台提交 ZIP 压缩包，其中包含一份 PDF 报告。

## 快速运行

需要 Python 3.10 或更新版本。算法、测试及实验脚本只用标准库，无需安装第三方包。

在本目录打开终端：

```powershell
python main.py
python -m unittest discover -s tests -v
python run_experiments.py --repeats 31
```

`main.py` 默认输出四题样例结果；也可分别运行 `python main.py q1` 至 `q4`。

原题结果：

| 题目 | 返回值 |
| --- | --- |
| 迷宫寻路 | `[(0,0),(0,1),(0,2),(1,2),(1,3),(1,4),(2,4),(3,4)]`，7 步 |
| 加权地形 | `7` |
| 极大极小树 | `5` |
| 宝藏 H=2 | `{"best_target":"T2","best_value":-1}` |
| 宝藏 H=4 | `{"best_target":"T3","best_value":3}` |

Minimax 与 Alpha-beta 均给出相同的题四答案。Python 函数返回坐标元组；命令行采用标准 JSON，元组显示为数组。

## 文件说明

| 文件或目录 | 内容 |
| --- | --- |
| `main.py` | 四题命令行入口，支持自定义 JSON 输入 |
| `search.py` | A*、路径重建、题一与题二接口 |
| `tree.py` | TreeNode、Minimax、Alpha-beta |
| `treasure.py` | 地图、36 次点对 A*、状态转移与两种博弈搜索 |
| `fixtures.py` | 题目一至三的原始数据和 JSON 树解析 |
| `extra_heuristics.py` | 固定终点的反向 Dijkstra 精确启发 |
| `run_experiments.py` | 样例及随机对照实验，导出 JSON、CSV |
| `tests/` | 35 项测试，包括独立算法与随机对照 |
| `examples/` | 四题可直接使用的 JSON 输入 |
| `results/results.json` | 原始测量、主变化、全部配置和汇总值 |
| `results/*.csv` | 可用表格软件查看的原始及汇总数据 |
| `results/sample_output.json` | 实际执行默认入口的输出 |
| `results/test_results.txt` | 本次全部测试通过的实际日志 |
| `report/实验二报告.pdf` | 5 页实验报告，首页含 GitHub 链接 |
| `report/实验二报告.md` | 同步生成的可编辑报告内容 |
| `build_report.py` | 从实测数据生成报告的程序 |
| `report_content.py` | 五页报告内容及动态数据表 |

## 自定义输入

```powershell
python main.py q1 --input examples/q1.json
python main.py q2 --input examples/q2.json
python main.py q3 --input examples/q3.json
python main.py q4 --input examples/q4_h4.json
```

题目一与二均接受 `start`、`goal`、`grid` 三个字段。坐标是 `[row,col]`，四邻域移动，`X` 表示障碍。

```json
{"start":[0,0],"goal":[0,1],"grid":[["S","E"]]}
```

以上输入，题一返回 `[[0,0],[0,1]]`，题二返回 `0`。题三接受题目给出的 `tree_structure` JSON；非叶用 `children`，叶用 `value`，根为 MAX，逐层交替。可选 `is_maximizing_player` 字段必须与层级一致，否则拒绝。

题四使用题目给定的地图和值，自定义配置接受 `depth`、`algorithm`、`order`、`use_cache`：

```json
{"depth":4,"algorithm":"alphabeta","order":"gain","use_cache":false}
```

算法可选 `minimax`、`alphabeta`，排序可选 `forward`、`reverse`、`gain`。完整的自定义距离矩阵和初始状态通过 Python 接口传入，见下文。

## Python 接口

```python
from search import maze_path, weighted_cost, astar
from tree import TreeNode, best_score, minimax, alpha_beta
from treasure import build_distance_matrix, solve_treasure, GameState

path = maze_path(start, goal, grid)         # 无解 []
cost = weighted_cost(start, goal, grid)     # E 免费；无解 -1
score = best_score(root)                   # 直接返回数值
matrix = build_distance_matrix()           # 顺序 A,B,T1,T2,T3,T4
answer = solve_treasure(4, distances=matrix)
print(answer.best_target, answer.best_value)
```

`astar(grid,start,goal,heuristic='manhattan',goal_cost_zero=False,unit_cost=False)` 返回 `SearchResult(path,cost,stats)`。标记 `. S E A B T1...` 默认进入代价为 1，数字及数字字符串为进入权值；题一包装函数强制单位权，题二包装函数强制目标免费。矩形、非负整数代价与端点会被检查，负数等非法地形抛 `ValueError`；空地图或不可达端点返回无解。

默认安全启发：普通目标为 `m * Manhattan`，免费目标为 `m * max(0,Manhattan-1)`，其中 m 为可能的非免费进入代价下界。存在内部零权时自动取 m=0；`heuristic='zero'` 可作对照。自定义函数签名 `(position,goal)->number`，需自行保证可采纳，且 `h(goal)=0`，否则不具最优性保证。

`minimax`、`alpha_beta` 返回含数值和计数的结果；默认子节点从左到右。`order='reverse'` 逆序；`order='ideal'` 预先完整求值后排序，其额外 `ordering_visited` 和全部排序时间单独披露，不算免费优化。

`GameState(a_pos,b_pos,remaining,score_a,score_b,player)` 不可变。题四 `solve_treasure(depth, algorithm='alphabeta',order='forward',use_cache=False,distances=None,initial_state=GameState(),build_pv=True)` 根必须由 A 行动，平分按最小宝藏编号。`root_values` 的每个首步均独立完整窗口搜索，返回精确值；因此没有根动作之间的 alpha 共享。`build_pv=False` 可关闭展示用主变化的完整 Minimax 复算，性能实验使用此开关。不可达宝藏不作为合法动作，无合法动作则按当前收益差结束；此拓展边界约定不影响原题全连通地图。

## 实验与计数

- A* `expanded` 包含起点和目标，`generated` 统计入堆，`stale` 统计过期队列项。不同启发共用右下左上邻接顺序和 `(f,h,入堆序号)` 平局规则。
- 博弈 `visited` 包含根和叶、缓存命中；`leaves` 统计真实叶评价；`cutoffs` 只计确实跳过剩余兄弟的截断。PV 复算不计入主搜索统计及计时。
- 每个计时配置预热 1 次、重复 31 次取中位数；30 个随机样本的时间汇总取这些中位数的平均。输入校验与排序计入耗时，文件读写和打印不计。
- 地图种子为 0 至 29，尺寸 25×25。随机部分障碍概率 25%，额外保留随机权值顶行及右列通路，保证可达。随机树分支数 3、深度 6、叶值 -100 至 100。
- 置换表正确区分 EXACT/LOWER/UPPER；本题缓存命中为 0，报告没有声称它获得加速。
- 精确启发的预处理计数和时间单列，单查询及固定终点 20 查询对照都计入成本。
- 微秒耗时因机器、Python 版本和后台负载而变化，重跑不保证与报告完全相同；确定性种子下答案与节点计数应一致。

## 重建报告

算法运行不需要报告依赖。若希望修改实测数据后同步重建 PDF：

```powershell
python -m pip install reportlab
python build_report.py
```

默认使用 Windows 宋体与黑体；其他系统可通过 `--font /path/chinese.ttf --bold-font /path/chinese-bold.ttf` 指定支持中文的 TrueType 字体。报告 Markdown 与 PDF 共用数据源，避免手工复制数字。提供的 PDF 已逐页渲染检查。

课程要求来源：《实验二要求_new.pdf》，2026年10月15日版本，第3至17页，以及后续补充的五页上限、PDF首页仓库链接与ZIP提交要求。提交渠道以课程通知为准。
