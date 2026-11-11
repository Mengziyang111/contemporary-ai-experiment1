"""实验二命令行入口。无参数运行全部原题，--input 接受 UTF-8 JSON。"""

import argparse
import json
from pathlib import Path

from fixtures import MAZE, MAZE_START, MAZE_GOAL, WEIGHTED, WEIGHTED_START, WEIGHTED_GOAL, sample_tree, tree_from_dict
from search import maze_path, weighted_cost
from tree import best_score


def solve(question, data=None):
    if question == "q1":
        d = data or {"start": MAZE_START, "goal": MAZE_GOAL, "grid": MAZE}
        return maze_path(tuple(d["start"]), tuple(d["goal"]), d["grid"])
    if question == "q2":
        d = data or {"start": WEIGHTED_START, "goal": WEIGHTED_GOAL, "grid": WEIGHTED}
        return weighted_cost(tuple(d["start"]), tuple(d["goal"]), d["grid"])
    if question == "q3":
        return best_score(tree_from_dict(data) if data else sample_tree())
    from treasure import build_distance_matrix, solve_treasure

    if question == "q4":
        if data is not None:
            unknown = set(data) - {"depth", "algorithm", "order", "use_cache"}
            if unknown:
                raise ValueError(f"题四自定义输入只接受 depth/algorithm/order/use_cache: {sorted(unknown)}")
            result = solve_treasure(**data)
            return {"best_target": result.best_target, "best_value": result.best_value}
        return {
            "labels": ["A", "B", "T1", "T2", "T3", "T4"],
            "distances": build_distance_matrix(),
            **{
                f"H={depth} {algorithm}": {
                    "best_target": (result := solve_treasure(depth, algorithm=algorithm)).best_target,
                    "best_value": result.best_value,
                }
                for depth in (2, 4)
                for algorithm in ("minimax", "alphabeta")
            },
        }
    raise ValueError("unknown question")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", choices=["all", "q1", "q2", "q3", "q4"], nargs="?", default="all")
    parser.add_argument("--input", type=Path, help="UTF-8 JSON 文件；输入格式见 README")
    args = parser.parse_args()
    if args.input and args.question == "all":
        parser.error("--input 需要指定 q1、q2、q3 或 q4")
    data = json.loads(args.input.read_text(encoding="utf-8")) if args.input else None
    output = {f"q{i}": solve(f"q{i}") for i in range(1, 5)} if args.question == "all" else solve(args.question, data)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
