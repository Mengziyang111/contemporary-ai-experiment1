"""题四：用 A* 距离与 Minimax / Alpha-beta 解决宝藏争夺。

所有距离均独立计算有向最短路。进入 A、B 和宝藏格的代价为 1，
与题二免费进入 E 的约定不同。搜索默认保持题目给定的 T1..T4 顺序。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence

from search import astar


GRID = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
LABELS = ("A", "B", "T1", "T2", "T3", "T4")
POSITIONS = ((0, 0), (6, 6), (0, 6), (2, 2), (4, 4), (6, 0))
VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
INDEX = {label: i for i, label in enumerate(LABELS)}


@dataclass(frozen=True)
class GameState:
    """位置为 LABELS 中的标签；remaining 顺序不改变状态语义。"""

    a_pos: str = "A"
    b_pos: str = "B"
    remaining: tuple[str, ...] = ("T1", "T2", "T3", "T4")
    score_a: int = 0
    score_b: int = 0
    player: str = "A"

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


@dataclass
class GameStats:
    # visited 包括根、叶和缓存命中，不包括主搜索后的 PV 复算。
    visited: int = 0
    # leaves 仅计实际评价的叶节点；从缓存返回的叶不重复计数。
    leaves: int = 0
    # cutoffs 仅计 alpha >= beta 且确有剩余兄弟被跳过的次数。
    cutoffs: int = 0
    cache_hits: int = 0
    # cache_cutoffs 单列由缓存界触发的提前返回，避免混淆剪枝次数。
    cache_cutoffs: int = 0


@dataclass(frozen=True)
class RootEvaluation:
    target: str
    value: int
    # 根各动作独立使用完整窗口，故均为 exact。
    bound: str


@dataclass
class GameResult:
    best_target: str | None
    best_value: int
    stats: GameStats
    # PV 用独立、完整的 Minimax 在主搜索后复算，不计入 stats。
    pv: list[str]
    root_values: list[RootEvaluation]


def build_distance_matrix(
    grid: Sequence[Sequence[object]] = GRID,
    positions: Sequence[tuple[int, int]] = POSITIONS,
) -> list[list[int]]:
    """计算全部有向点对，-1 表示不可达；不假定矩阵对称。"""

    return [
        [astar(grid, start, end).cost for end in positions]
        for start in positions
    ]


def available_targets(state: GameState, distances: Sequence[Sequence[int]]) -> list[str]:
    position = state.a_pos if state.player == "A" else state.b_pos
    return [
        target for target in sorted(state.remaining)
        if distances[INDEX[position]][INDEX[target]] >= 0
    ]


def apply_move(
    state: GameState, target: str, distances: Sequence[Sequence[int]],
) -> GameState:
    """只收集目标宝藏；不将经过的其他宝藏加入收益。"""

    if target not in state.remaining:
        raise ValueError("目标必须是尚未获取的宝藏")
    position = state.a_pos if state.player == "A" else state.b_pos
    distance = distances[INDEX[position]][INDEX[target]]
    if distance < 0:
        raise ValueError("目标宝藏不可达")
    gain = VALUES[target] - distance
    remaining = tuple(t for t in state.remaining if t != target)
    if state.player == "A":
        return GameState(target, state.b_pos, remaining,
                         state.score_a + gain, state.score_b, "B")
    return GameState(state.a_pos, target, remaining,
                     state.score_a, state.score_b + gain, "A")


def _ordered_targets(
    state: GameState, distances: Sequence[Sequence[int]], order: str,
) -> list[str]:
    targets = available_targets(state, distances)
    if order == "reverse":
        return list(reversed(targets))
    if order == "gain":
        position = state.a_pos if state.player == "A" else state.b_pos
        # 双方都优先考虑自身即时收益较大的目标。平分按 T 编号。
        return sorted(targets, key=lambda target: (
            -(VALUES[target] - distances[INDEX[position]][INDEX[target]]), target))
    return targets


def exact_action_values(
    depth: int,
    *,
    distances: Sequence[Sequence[int]] | None = None,
    initial_state: GameState = GameState(),
) -> dict[str, int]:
    """独立完整 Minimax，给出根各动作精确值，用于报告和交叉核对。"""

    if depth < 0:
        raise ValueError("depth 必须非负")
    matrix = build_distance_matrix() if distances is None else distances

    def value(state: GameState, remaining_depth: int) -> int:
        moves = available_targets(state, matrix)
        if remaining_depth == 0 or not moves:
            return state.utility
        values = [value(apply_move(state, target, matrix), remaining_depth - 1)
                  for target in moves]
        return (max if state.player == "A" else min)(values)

    if depth == 0:
        return {}
    return {target: value(apply_move(initial_state, target, matrix), depth - 1)
            for target in available_targets(initial_state, matrix)}


def solve_treasure(
    depth: int,
    *,
    algorithm: str = "alphabeta",
    order: str = "forward",
    use_cache: bool = False,
    build_pv: bool = True,
    distances: Sequence[Sequence[int]] | None = None,
    initial_state: GameState = GameState(),
) -> GameResult:
    """计算 A 的最优首步；默认根是 A，最多搜索所有剩余宝藏。

    depth 单位是一次选宝藏行动，H=2 表示 A、B 各行动一次。
    缓存键包含完整不可变状态与剩余深度；Alpha-beta 缓存区分
    EXACT、LOWER、UPPER，不把剪枝结果误当精确值。根各动作分别使用
    (-inf, +inf) 完整窗口，保证 root_values 全为精确值，平分首步稳定
    选择 T 编号最小者。该设计放弃根各动作之间的 alpha 传播，以便
    直接比较所有首步的准确分值；深层节点仍正常 Alpha-beta 剪枝。
    build_pv=False 可跳过独立的完整 Minimax 主变化复算，适于仅计时
    主搜索。不可达宝藏不是合法动作；无合法动作时按当前 U 终止，不隐式跳回合。
    这条边界约定不影响题目给定的全连通地图。
    """

    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError("depth 必须是非负整数")
    if algorithm not in ("minimax", "alphabeta"):
        raise ValueError("algorithm 取 minimax 或 alphabeta")
    if order not in ("forward", "reverse", "gain"):
        raise ValueError("order 取 forward、reverse 或 gain")
    if initial_state.player != "A":
        raise ValueError("solve_treasure 的根必须由 A 行动")
    matrix = build_distance_matrix() if distances is None else distances
    if len(matrix) != 6 or any(len(row) != 6 for row in matrix):
        raise ValueError("distances 必须是按 LABELS 排列的 6×6 矩阵")
    stats = GameStats(visited=1)
    # key -> (value, flag)。不缓存当前根动作和 PV 复算数据。
    table: dict[tuple[GameState, int], tuple[int, str]] = {}

    def search(state: GameState, remaining_depth: int, alpha: float, beta: float) -> int:
        stats.visited += 1
        key = (state, remaining_depth)
        original_alpha, original_beta = alpha, beta
        if use_cache and key in table:
            stats.cache_hits += 1
            cached_value, flag = table[key]
            if flag == "EXACT":
                return cached_value
            if flag == "LOWER":
                alpha = max(alpha, cached_value)
            else:
                beta = min(beta, cached_value)
            if alpha >= beta:
                stats.cache_cutoffs += 1
                return cached_value
        targets = _ordered_targets(state, matrix, order)
        if remaining_depth == 0 or not targets:
            stats.leaves += 1
            value = state.utility
            if use_cache:
                table[key] = (value, "EXACT")
            return value
        value = -inf if state.player == "A" else inf
        for move_index, target in enumerate(targets):
            child_value = search(apply_move(state, target, matrix),
                                 remaining_depth - 1, alpha, beta)
            if state.player == "A":
                value = max(value, child_value)
                alpha = max(alpha, value)
            else:
                value = min(value, child_value)
                beta = min(beta, value)
            if algorithm == "alphabeta" and alpha >= beta:
                if move_index < len(targets) - 1:
                    stats.cutoffs += 1
                break
        result = int(value)
        if use_cache:
            flag = "EXACT"
            if algorithm == "alphabeta":
                if result <= original_alpha:
                    flag = "UPPER"
                elif result >= original_beta:
                    flag = "LOWER"
            table[key] = (result, flag)
        return result

    targets = _ordered_targets(initial_state, matrix, order)
    if depth == 0 or not targets:
        stats.leaves = 1
        return GameResult(None, initial_state.utility, stats, [], [])
    best_target: str | None = None
    best_value = -inf
    root_values: list[RootEvaluation] = []
    for target in targets:
        child = apply_move(initial_state, target, matrix)
        value = search(child, depth - 1, -inf, inf)
        root_values.append(RootEvaluation(target, value, "exact"))
        if value > best_value or (value == best_value and
                                  (best_target is None or target < best_target)):
            best_target, best_value = target, value

    # 精确主变化只服务展示；算法计数不包括这一步的完整重算。
    if not build_pv:
        return GameResult(best_target, int(best_value), stats, [], root_values)
    pv = [best_target]
    current = apply_move(initial_state, best_target, matrix)
    for remaining_depth in range(depth - 1, 0, -1):
        moves = available_targets(current, matrix)
        if not moves:
            break
        values = exact_action_values(remaining_depth, distances=matrix, initial_state=current)
        optimum = (max if current.player == "A" else min)(values.values())
        target = min(t for t in moves if values[t] == optimum)
        pv.append(target)
        current = apply_move(current, target, matrix)
    return GameResult(best_target, int(best_value), stats, pv, root_values)
