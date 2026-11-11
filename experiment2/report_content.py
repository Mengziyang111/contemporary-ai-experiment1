"""Five-page report content built directly from measured experiment results."""


def create_content(d, repo_url):
    """Return five logical pages for the existing report renderer."""
    from fixtures import MAZE, WEIGHTED
    from treasure import VALUES

    pages = []
    current = []

    def page(title):
        nonlocal current
        current = [("heading", title)]
        pages.append(current)

    def p(text):
        current.append(("paragraph", text))

    def sub(text):
        current.append(("subheading", text))

    def table(headers, rows, widths):
        current.append(("table", (headers, rows, widths)))

    def code(text):
        current.append(("code", text))

    def one(section, **where):
        return next(row for row in d[section]
                    if all(row[key] == value for key, value in where.items()))

    def fmt(value):
        return str(value) if isinstance(value, int) else f"{value:.2f}"

    def path_text(path):
        return "->".join(f"({r},{c})" for r, c in path)

    q1_zero = one("grids", problem="q1", heuristic="zero")
    q1_h = one("grids", problem="q1", heuristic="manhattan")
    q2_zero = one("grids", problem="q2", heuristic="zero")
    q2_h = one("grids", problem="q2", heuristic="manhattan")
    game = {depth: one("treasure", depth=depth, algorithm="minimax",
                       order="forward", cache=False) for depth in (2, 4)}
    labels = d["distance_labels"]
    matrix = d["distance_matrix"]
    indices = {label: index for index, label in enumerate(labels)}
    order_names = {"forward": "正序", "reverse": "逆序", "gain": "即时收益", "ideal": "理想"}

    page("实验二 搜索算法实验报告")
    current.append(("repo_link", repo_url))
    current.append(("identity", "孟子阳    学号 10245501476    2026年11月11日"))
    p(f"实现 A*、Minimax 与 Alpha-beta，四题样例均通过。环境为 Python {d['metadata']['python']}、Windows 11。"
      f"用 perf_counter_ns 计时，各配置预热后重复 {d['metadata']['repeats']} 次取中位数，单位为微秒。"
      "随机实验表中为逐样本中位数的均值。计时含校验与排序，排除读写、距离矩阵构建和展示用主变化复算。")
    sub("A星状态与优先队列")
    p("状态为坐标，g 是已知起点费用，h 是剩余费用下界，f=g+h。队列按 (f,h,入堆序号) 排序，邻居固定右下左上。"
      "新 g 严格更小时更新前驱并入堆，过期项出队丢弃，允许改进后的节点重开。目标有效出队后回溯路径，无解返回空路径或 -1。")
    p("expanded 计有效出队且含首终点，generated 计入堆且含重复改进，stale 计过期出队，因此 expanded=popped-stale。起点不计进入费用。")
    sub("题一 迷宫最短路径")
    p("四邻域每步费用为 1，h=Manhattan。每条路径至少完成相应行列变化，故 h 可采纳。相邻格的 h 最多变化 1，亦满足一致性。h=0 等价于单位权 Dijkstra。")
    current.append(("figure", ("grid", {"grid": MAZE, "path": q1_h["path"], "cell": 16},
                               "图1 蓝色格为返回路径 格内数字表示经过次序")))
    code(path_text(q1_h["path"]))
    table(["启发函数", "费用", "扩展", "入堆", "微秒"],
          [[name, row["cost"], row["expanded"], row["generated"], fmt(row["median_us"])]
           for name, row in (("h=0", q1_zero), ("曼哈顿", q1_h))],
          [147, 75, 75, 75, 115])
    p(f"路径共 {len(q1_h['path'])-1} 步，达到首终点曼哈顿下界，故为最短。扩展由 {q1_zero['expanded']} 降至 {q1_h['expanded']}，"
      f"减少 {(1-q1_h['expanded']/q1_zero['expanded'])*100:.1f}%。本例相同 f 时优先小 h 恰好沿最短路径出队，节点数随地图和平局规则变化。")

    page("加权地形与极大极小树")
    sub("题二 费用累计与可采纳启发")
    p("边 u→v 按进入 v 的地形计费，进入 E 按题意收费 0。最少步数与最小费用须分别处理。下列路径同时给出逐格进入费用与累计 g，首项 0 表示起点不收费。")
    entries = [0 if i in (0, len(q2_h["path"])-1) else int(WEIGHTED[r][c])
               for i, (r, c) in enumerate(q2_h["path"])]
    cumulative = []
    for entry in entries:
        cumulative.append((cumulative[-1] if cumulative else 0) + entry)
    code(path_text(q2_h["path"]) + "\nentry = " + str(entries) + "\ng = " + str(cumulative))
    p("E 的邻格真实剩余费用为 0，原始 Manhattan 却为 1，因而会高估。令 m 为非障碍、非免费终点格的最小进入费用，"
      "采用 h=m×max(0,Manhattan-1)。至少需要 Manhattan 步，去掉入 E 的最后一步后，仍有至少 Manhattan-1 步各付至少 m，故该式为下界。"
      "普通边上 h 至多减少 m，免费入 E 边两端 h 均为 0，故也一致。内部若含零权则 m=0，安全退化为 h=0。")
    table(["启发函数", "费用", "扩展", "入堆", "微秒"],
          [[name, row["cost"], row["expanded"], row["generated"], fmt(row["median_us"])]
           for name, row in (("h=0", q2_zero), ("修正曼哈顿", q2_h))],
          [147, 75, 75, 75, 115])
    sub("题三 数值回传与从左到右剪枝")
    p("Minimax 在叶返回得分，MAX 取最大值，MIN 取最小值，根为 MAX。样例底层 MAX 值为 5、9、2，"
      "左右 MIN 返回 min(5,9)=5 与 min(9,2)=2，根得 5。Alpha 是 MAX 祖先已有下界，beta 是 MIN 祖先已有上界。"
      "左 MIN 先得 5 后令 beta=5，其第二个 MAX 读到 6，alpha≥beta，便跳过叶 9。这里返回的 6 是子树下界，根值仍精确。")
    tree_rows = [row for row in d["trees"]
                 if row["algorithm"] == "alphabeta" or row["order"] == "forward"]
    table(["方法与顺序", "得分", "访问", "叶", "剪枝", "排序预访问", "微秒"],
          [[("MM " if row["algorithm"] == "minimax" else "AB ")+order_names[row["order"]],
            row["value"], row["visited"], row["leaves"], row["cutoffs"],
            row["ordering_visited"], fmt(row["median_us"])] for row in tree_rows],
          [115, 47, 53, 45, 53, 82, 92])
    ideal = one("trees", algorithm="alphabeta", order="ideal")
    p(f"完整搜索按树路径展开重复引用。正序与逆序本例开销相同。理想顺序先完整求子树真值再排序，主搜索访问 {ideal['visited']} 次，"
      f"另有 {ideal['ordering_visited']} 次预访问，总计 {ideal['visited']+ideal['ordering_visited']} 次，计时已包含排序预处理。")

    page("宝藏分配的状态与深度决策")
    p("题四特殊格 A、B、T 的进入费用均为 1。T1 至 T4 价值依次为 "
      + "、".join(str(VALUES[target]) for target in VALUES)
      + f"。对六点全部 {len(labels)**2} 个有序对独立运行 A*，行表示出发点，列表示目标，递归阶段直接查表。")
    table(["出发与目标", *labels],
          [[label, *row] for label, row in zip(labels, matrix)],
          [103, 64, 64, 64, 64, 64, 64])
    p("一般进入权网格可能产生有向距离。该例六个端点进入权均为 1，一条路径和其反向路径费用相同，所以矩阵对称。实现仍逐个计算有序对。")
    sub("状态转移与评价")
    p("S=(pA,pB,R,scoreA,scoreB,player) 记录双方位置、剩余宝藏、累计收益和行动方。取 Ti 获得 gain=Vi-d(p,Ti)，"
      "更新该玩家位置和得分并删除 Ti，随后换人。途经宝藏不自动收集，已取宝藏仍可通行，另一玩家不挡路。"
      "每次选宝算一层，到达 H 或取完时评价 U=scoreA-scoreB，A 最大化，B 最小化。")
    root_values = {depth: {row["target"]: row["value"] for row in game[depth]["root_values"]}
                   for depth in (2, 4)}
    table(["A 首步", "首次 gain", "H=2 精确回传", "H=4 精确回传"],
          [[target, VALUES[target]-matrix[indices['A']][indices[target]],
            root_values[2][target], root_values[4][target]] for target in VALUES],
          [97, 114, 138, 138])
    first, reply = game[2]["pv"]
    first_gain = VALUES[first]-matrix[indices['A']][indices[first]]
    reply_gain = VALUES[reply]-matrix[indices['B']][indices[reply]]
    p(f"H=2 最优为 {game[2]['best_target']}，A 首获 {first_gain}，B 最优应对取 {reply} 获 {reply_gain}，"
      f"因此 U={first_gain}-({reply_gain})={game[2]['best_value']}。H=4 改选 {game[4]['best_target']}，最优差为 {game[4]['best_value']}。")
    a_pos, b_pos, score_a, score_b = "A", "B", 0, 0
    steps = []
    for turn, target in enumerate(game[4]["pv"]):
        player = "A" if turn % 2 == 0 else "B"
        position = a_pos if player == "A" else b_pos
        gain = VALUES[target]-matrix[indices[position]][indices[target]]
        if player == "A":
            a_pos, score_a = target, score_a+gain
        else:
            b_pos, score_b = target, score_b+gain
        steps.append(f"{player}取{target}，gain={gain}、U={score_a-score_b}")
    p("完整搜索的一条最优主变化依次为 " + "。".join(steps)
      + f"。最终 A 得 {score_a}、B 得 {score_b}。双方均按剩余深度最优应对，负 gain 的宝藏也须取走。")
    p(f"先取 T2 在 H=4 时只能保证 {root_values[4]['T2']}。取 T3 占据 B 附近的高净收益目标，并改变未来位置和剩余选择。"
      "深度增加纳入第二轮收益，故最优首步改变，相同深度下两种搜索给出一致答案。")

    page("剪枝顺序与搜索开销对照")
    sub("综合题的访问顺序")
    p("MM 表示 Minimax，AB 表示 Alpha-beta。visited 含根、内部节点、叶及缓存命中，leaves 计实际叶评价，cutoffs 只计确实跳过兄弟的截断。表中均关闭缓存。")
    game_rows = [row for row in d["treasure"] if not row["cache"] and
                 ((row["depth"] == 2 and row["order"] == "forward") or
                  (row["depth"] == 4 and (row["algorithm"] == "alphabeta" or row["order"] == "forward")))]
    table(["H", "方法", "顺序", "访问", "叶", "剪枝", "微秒"],
          [[row["depth"], "MM" if row["algorithm"] == "minimax" else "AB",
            order_names[row["order"]], row["visited"], row["leaves"], row["cutoffs"],
            fmt(row["median_us"])] for row in game_rows],
          [35, 53, 101, 65, 55, 65, 113])
    mm4 = game[4]
    ab4 = one("treasure", depth=4, algorithm="alphabeta", order="gain", cache=False)
    p(f"即时 gain 排序使 H=4 访问从 {mm4['visited']} 降至 {ab4['visited']}，减少 {(1-ab4['visited']/mm4['visited'])*100:.2f}%。"
      "双方都先试自身 gain 较大的目标，有助于尽早收紧界。排序有额外成本，节点减少与耗时改善须分别比较。")
    p(f"根的每个动作独立使用 (-∞,+∞) 窗口，以取得上页全部精确首步值，代价是放弃根动作之间共享 alpha 的剪枝机会。"
      f"H=2 子树只有 MIN 与叶，三种顺序均访问 {game[2]['visited']} 次。平分首步按编号最小选择。")
    cache_hits = {row["cache_hits"] for row in d["treasure"] if row["cache"]}
    p("缓存键含完整状态和剩余深度，条目标记 EXACT、LOWER、UPPER，按界收紧窗口。"
      f"本题开启缓存的所有配置命中数为 {','.join(map(str, sorted(cache_hits)))}，短树中未产生可复用的相同完整状态，默认关闭。")
    sub("自设较深随机树")
    trees = {row["strategy"]: row for row in d["random_trees_summary"]}
    p(f"种子与地图实验相同，共 {trees['minimax']['n']} 棵三叉深六完整树，叶值均匀取 -100 至 100。每个种子上各方法回传值均一致。")
    table(["方法", "平均访问", "平均叶", "排序预访问", "平均微秒"],
          [[name, fmt(trees[key]["visited"]), fmt(trees[key]["leaves"]),
            fmt(trees[key]["ordering_visited"]), fmt(trees[key]["median_us"])]
           for key, name in (("minimax", "Minimax"), ("ab_forward", "AB 正序"),
                             ("ab_reverse", "AB 逆序"), ("ab_ideal", "AB 理想"))],
          [113, 91, 91, 100, 92])
    p(f"理想排序需先访问 {fmt(trees['ab_ideal']['ordering_visited'])} 次，再做 {fmt(trees['ab_ideal']['visited'])} 次主搜索。"
      "这些计时包含预处理，展示了低主搜索节点数可能伴随较高总费用。正逆顺序的样本均值差异也不足以证明固定顺序普遍更优。")

    page("地图拓展与预处理成本")
    grids = {row["heuristic"]: row for row in d["random_grids_summary"]}
    seeds = d["metadata"]["seeds"]
    p(f"种子 {min(seeds)} 至 {max(seeds)} 共 {grids['zero']['n']} 张 25×25 地图，随机格障碍率 25%，权值抽取自 [1,1,1,2,3,5,8]。"
      "顶行与最右列改为权值来自 [1,2,3,5] 的通道以确保可达，这使地图结构存在偏置。各方法采用相同图和队列规则。")
    table(["启发函数", "平均扩展", "过期出队", "查询微秒", "预处理微秒"],
          [[name, fmt(grids[key]["expanded"]), fmt(grids[key]["stale"]),
            fmt(grids[key]["median_us"]), fmt(grids[key]["preprocessing_us"])]
           for key, name in (("zero", "h=0"), ("manhattan", "曼哈顿"), ("exact_reverse", "反向精确 h"))],
          [113, 91, 91, 96, 96])
    p(f"每图费用均一致，平均 {fmt(grids['zero']['cost'])}。曼哈顿平均扩展减少 {(1-grids['manhattan']['expanded']/grids['zero']['expanded'])*100:.2f}%，"
      f"但出现 {fmt(grids['manhattan']['stale'])} 次过期出队，表中实测查询费用也反映启发和堆操作的开销。")
    sub("反向精确启发与重复查询")
    exact = grids["exact_reverse"]
    p("固定终点后反向运行一次 Dijkstra。正向 u→v 收 cost(v)，反向从 v 松弛 u 仍须加 cost(v)。"
      "得到的可达节点 h 等于真实剩余费用，且一致，不可达节点取安全下界 0。"
      f"预处理另扩展 {fmt(exact['preprocessing_expanded'])} 次，单查询总耗时约 {fmt(exact['preprocessing_us']+exact['median_us'])} 微秒。")
    fixed = {row["heuristic"]: row for row in d["fixed_goal"]}
    p(f"种子 2026 的图固定终点 (24,24)，从 (0,0) 至 (0,19) 共 {fixed['manhattan']['queries']} 次查询，各次费用均一致，预处理只做一次。")
    table(["方法", "查询总扩展", "预处理扩展", "总计微秒"],
          [[name, fixed[key]["total_expanded"], fixed[key]["preprocessing_expanded"], fmt(fixed[key]["total_us"])]
           for key, name in (("manhattan", "曼哈顿 A*"), ("exact_reverse", "反向 h 再 A*"))],
          [137, 110, 110, 130])
    p("重复查询摊薄了预处理，地图或目标改变后须重建。若只求距离，反向表本身即可直接查答案，此处继续 A* 用于启发强度对照。")
    sub("零费用反例与独立验证")
    counter = d["zero_counterexample"]
    p("自设两行地图 [S,2,E] 与 [0,0,0]，E 免费。"
      f"原始 Manhattan 返回 {counter['raw_manhattan']['cost']}、扩展 {counter['raw_manhattan']['expanded']} 次，"
      f"安全启发与 h=0 均返回 {counter['zero']['cost']}、扩展 {counter['zero']['expanded']} 次。"
      "首步两个 f 同为 3，较小 h 使错误直路先出队，说明零权时须修正启发。")
    p("自动测试用独立 BFS、Dijkstra、Floyd-Warshall 和完整穷举核对结果，覆盖无解、同点、零权、负权拒绝、重开、缓存界与平分。"
      "运行记录见 results/test_results.txt，逐种子数据见 results。题意依据课程《实验二要求_new.pdf》第3至17页。")
    code("python main.py\npython -m unittest discover -s tests -v\npython run_experiments.py --repeats 31")
    sub("实验结论")
    p("正确的代价与截断规则决定答案含义，可采纳启发和安全剪枝保持结果正确。排序与预处理可降低搜索量，"
      "但须同时计入其额外开销；缓存无命中和节点更少却未更快的结果，也应纳入优化判断。")
    return pages
