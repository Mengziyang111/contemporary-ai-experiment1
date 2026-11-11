"""运行全部真实实验，导出 JSON 和 CSV；不把预设答案充当测量结果。"""

import argparse
import csv
import json
import platform
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter_ns

from extra_heuristics import reverse_distance_heuristic
from fixtures import MAZE, MAZE_START, MAZE_GOAL, WEIGHTED, WEIGHTED_START, WEIGHTED_GOAL, sample_tree
from search import astar
from tree import TreeNode, alpha_beta, minimax


def measure(function, repeats):
    function()  # 不计入测量的预热
    times = []
    result = None
    for _ in range(repeats):
        start = perf_counter_ns()
        result = function()
        times.append((perf_counter_ns() - start) / 1000)
    return result, round(median(times), 3)


def random_grid(seed, size=25):
    rng = random.Random(seed)
    grid = [["X" if rng.random() < 0.25 else rng.choice([1, 1, 1, 2, 3, 5, 8]) for _ in range(size)] for _ in range(size)]
    # 固定保留一条通路，避免不可达样本掩盖搜索效率。
    for i in range(size):
        grid[0][i] = rng.choice([1, 2, 3, 5])
        grid[i][-1] = rng.choice([1, 2, 3, 5])
    grid[0][0], grid[-1][-1] = "S", "E"
    return grid


def random_tree(seed, depth=6, branching=3):
    rng = random.Random(seed)

    def build(level):
        if level == 0:
            return TreeNode(value=rng.randint(-100, 100))
        return TreeNode(children=[build(level - 1) for _ in range(branching)])

    return build(depth)


def run(output_dir, repeats=31):
    from treasure import build_distance_matrix, solve_treasure

    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "repeats": repeats,
            "timing_unit": "microseconds", "timing": "median after one warm-up; no file IO",
            "seeds": list(range(30)), "astar_tie_break": "(f,h,insertion_order)",
            "astar_expanded_includes_goal": True,
        },
        "grids": [], "trees": [], "treasure": [], "random_grids": [], "random_trees": [],
    }
    for name, grid, start, goal, unit, free in [
        ("q1", MAZE, MAZE_START, MAZE_GOAL, True, False),
        ("q2", WEIGHTED, WEIGHTED_START, WEIGHTED_GOAL, False, True),
    ]:
        for heuristic in ("zero", "manhattan"):
            result, elapsed = measure(lambda: astar(grid, start, goal, heuristic=heuristic, unit_cost=unit, goal_cost_zero=free), repeats)
            data["grids"].append({"problem": name, "heuristic": heuristic, "path": result.path, "cost": result.cost, **asdict(result.stats), "median_us": elapsed})

    root = sample_tree()
    for algorithm, function in (("minimax", minimax), ("alphabeta", alpha_beta)):
        for order in ("forward", "reverse", "ideal"):
            result, elapsed = measure(lambda: function(root, order=order), repeats)
            data["trees"].append({"algorithm": algorithm, "order": order, "value": result.value, **asdict(result.stats), "median_us": elapsed})

    matrix = build_distance_matrix()
    data["distance_labels"] = ["A", "B", "T1", "T2", "T3", "T4"]
    data["distance_matrix"] = matrix
    for depth in (2, 4):
        for algorithm in ("minimax", "alphabeta"):
            for order in ("forward", "reverse", "gain"):
                for cache in (False, True):
                    result, elapsed = measure(lambda: solve_treasure(depth, algorithm=algorithm, order=order, use_cache=cache, distances=matrix, build_pv=False), repeats)
                    display = solve_treasure(depth, algorithm=algorithm, order=order, use_cache=cache, distances=matrix)
                    data["treasure"].append({"depth": depth, "algorithm": algorithm, "order": order, "cache": cache, "best_target": result.best_target, "best_value": result.best_value, "pv": display.pv, "root_values": [asdict(item) for item in result.root_values], **asdict(result.stats), "median_us": elapsed})

    # 零代价是自设拓展输入，原题二给出的数字均为正。
    zero_grid = [["S", 2, "E"], [0, 0, 0]]
    raw = lambda pos, goal: abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    data["zero_counterexample"] = {"grid": zero_grid}
    for name, heuristic in (("zero", "zero"), ("safe_manhattan", "manhattan"), ("raw_manhattan", raw)):
        result = astar(zero_grid, (0, 0), (0, 2), heuristic=heuristic, goal_cost_zero=True)
        data["zero_counterexample"][name] = {"cost": result.cost, "path": result.path, **asdict(result.stats)}

    # 30 张固定种子地图；每张三种 h 使用完全相同的图、代价和队列规则。
    for seed in range(30):
        grid = random_grid(seed)
        goal = (24, 24)
        (exact_h, prep_expanded), prep_us = measure(lambda: reverse_distance_heuristic(grid, goal), repeats)
        seed_costs = []
        for label, h in (("zero", "zero"), ("manhattan", "manhattan"), ("exact_reverse", exact_h)):
            result, elapsed = measure(lambda: astar(grid, (0, 0), goal, heuristic=h), repeats)
            seed_costs.append(result.cost)
            data["random_grids"].append({"seed": seed, "heuristic": label, "cost": result.cost, **asdict(result.stats), "median_us": elapsed, "preprocessing_expanded": prep_expanded if label == "exact_reverse" else 0, "preprocessing_us": round(prep_us, 3) if label == "exact_reverse" else 0})
        assert len(set(seed_costs)) == 1, (seed, seed_costs)

    for seed in range(30):
        root = random_tree(seed)
        seed_values = []
        for label, function, order in (("minimax", minimax, "forward"), ("ab_forward", alpha_beta, "forward"), ("ab_reverse", alpha_beta, "reverse"), ("ab_ideal", alpha_beta, "ideal")):
            result, elapsed = measure(lambda: function(root, order=order), repeats)
            seed_values.append(result.value)
            data["random_trees"].append({"seed": seed, "strategy": label, "value": result.value, **asdict(result.stats), "median_us": elapsed})
        assert len(set(seed_values)) == 1, (seed, seed_values)

    # 同一终点 20 次不同起点查询，比较预处理是否摊薄；起点全部在保留通路上。
    grid, goal = random_grid(2026), (24, 24)
    starts = [(0, col) for col in range(20)]

    def repeated_queries(h):
        return [astar(grid, start, goal, heuristic=h) for start in starts]

    exact_h, prep_expanded = reverse_distance_heuristic(grid, goal)
    _, prep_us = measure(lambda: reverse_distance_heuristic(grid, goal), repeats)
    data["fixed_goal"] = []
    for name, h in (("manhattan", "manhattan"), ("exact_reverse", exact_h)):
        results, elapsed = measure(lambda: repeated_queries(h), repeats)
        data["fixed_goal"].append({"heuristic": name, "queries": len(starts), "total_expanded": sum(r.stats.expanded for r in results), "median_query_batch_us": elapsed, "preprocessing_us": prep_us if name == "exact_reverse" else 0, "preprocessing_expanded": prep_expanded if name == "exact_reverse" else 0, "total_us": round(elapsed + (prep_us if name == "exact_reverse" else 0), 3), "costs": [r.cost for r in results]})
    assert data["fixed_goal"][0]["costs"] == data["fixed_goal"][1]["costs"]

    for section, group in (("random_grids", "heuristic"), ("random_trees", "strategy")):
        data[section + "_summary"] = []
        for label in sorted({row[group] for row in data[section]}):
            rows = [row for row in data[section] if row[group] == label]
            numeric = [key for key in rows[0] if key not in {"seed", group} and isinstance(rows[0][key], (float, int))]
            data[section + "_summary"].append({group: label, "n": len(rows), **{key: round(mean(row[key] for row in rows), 3) for key in numeric}})

    (output_dir / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for section in ("grids", "trees", "treasure", "random_grids", "random_trees", "random_grids_summary", "random_trees_summary", "fixed_goal"):
        rows = data[section]
        with (output_dir / f"{section}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"实验完成：{output_dir.resolve() / 'results.json'}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats 必须大于 0")
    run(args.output, args.repeats)
