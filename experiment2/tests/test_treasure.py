"""题四独立穷举及 Floyd-Warshall 对照，纯标准库 unittest。"""

import random
import unittest
from itertools import permutations

from treasure import (
    GRID, INDEX, LABELS, POSITIONS, VALUES, GameState, apply_move,
    build_distance_matrix, exact_action_values, solve_treasure,
)


def floyd_warshall(grid, positions):
    """独立全点对动态规划，不调用 A* 或生产代码的代价解析。"""
    cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row)
             if value != "X"]
    index = {position: i for i, position in enumerate(cells)}
    n = len(cells)
    distance = [[float("inf")] * n for _ in range(n)]
    for i, (r, c) in enumerate(cells):
        distance[i][i] = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other = (r + dr, c + dc)
            if other in index:
                value = grid[other[0]][other[1]]
                cost = value if isinstance(value, int) else 1
                distance[i][index[other]] = cost
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return [[-1 if distance[index[a]][index[b]] == float("inf")
             else int(distance[index[a]][index[b]]) for b in positions] for a in positions]


def exhaustive(depth, matrix):
    """先枚举全部排列计算叶值，再从完整前缀表回传，独立于递归求解器。"""
    if depth == 0:
        return None, 0
    depth = min(depth, 4)
    terminal = {}
    for sequence in permutations(("T1", "T2", "T3", "T4"), depth):
        locations = [0, 1]
        scores = [0, 0]
        for turn, target in enumerate(sequence):
            player = turn % 2
            target_index = INDEX[target]
            scores[player] += VALUES[target] - matrix[locations[player]][target_index]
            locations[player] = target_index
        terminal[sequence] = scores[0] - scores[1]
    for level in range(depth - 1, 0, -1):
        grouped = {}
        for sequence, value in terminal.items():
            grouped.setdefault(sequence[:level], []).append(value)
        operation = min if level % 2 else max
        terminal = {prefix: operation(values) for prefix, values in grouped.items()}
    best = max(terminal.values())
    return min(sequence[0] for sequence, value in terminal.items() if value == best), best


class TreasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = build_distance_matrix()

    def test_matrix_against_floyd_warshall(self):
        self.assertEqual(self.matrix, floyd_warshall(GRID, POSITIONS))
        self.assertEqual([self.matrix[i][i] for i in range(6)], [0] * 6)

    def test_directed_distance_not_assumed_symmetric(self):
        # 最小网格的端点进入权值不同，证明矩阵构建没有偷用对称性。
        matrix = build_distance_matrix([[2, 7]], [(0, 0), (0, 1)])
        self.assertEqual(matrix, [[0, 7], [2, 0]])

    def test_given_horizons(self):
        for depth, target, value in ((2, "T2", -1), (4, "T3", 3)):
            for algorithm in ("minimax", "alphabeta"):
                for order in ("forward", "reverse", "gain"):
                    for cache in (False, True):
                        with self.subTest(depth=depth, algorithm=algorithm, order=order, cache=cache):
                            result = solve_treasure(depth, algorithm=algorithm, order=order,
                                                    use_cache=cache, distances=self.matrix)
                            self.assertEqual((result.best_target, result.best_value), (target, value))
                            self.assertEqual(len(result.pv), depth)

    def test_all_depths_against_independent_exhaustive(self):
        for depth in range(7):
            expected = exhaustive(depth, self.matrix)
            for algorithm in ("minimax", "alphabeta"):
                result = solve_treasure(depth, algorithm=algorithm, distances=self.matrix)
                self.assertEqual((result.best_target, result.best_value), expected)

    def test_state_update_and_no_automatic_collection(self):
        initial = GameState()
        child = apply_move(initial, "T1", self.matrix)
        self.assertEqual(child.player, "B")
        self.assertEqual(child.a_pos, "T1")
        self.assertEqual(child.b_pos, "B")
        self.assertEqual(child.score_a, 9 - self.matrix[0][2])
        self.assertEqual(child.score_b, 0)
        self.assertEqual(child.remaining, ("T2", "T3", "T4"))
        self.assertEqual(initial.remaining, ("T1", "T2", "T3", "T4"))
        with self.assertRaises(ValueError):
            apply_move(child, "T1", self.matrix)

    def test_minimax_counts_complete_tree(self):
        depth_two = solve_treasure(2, algorithm="minimax", distances=self.matrix)
        depth_four = solve_treasure(4, algorithm="minimax", distances=self.matrix)
        self.assertEqual((depth_two.stats.visited, depth_two.stats.leaves), (17, 12))
        self.assertEqual((depth_four.stats.visited, depth_four.stats.leaves), (65, 24))
        self.assertEqual(depth_four.stats.cutoffs, 0)

    def test_random_weighted_maps_against_floyd(self):
        rng = random.Random(20261107)
        for _ in range(12):
            grid = [[rng.randint(1, 9) for _ in range(4)] for _ in range(4)]
            for r, c in ((1, 1), (2, 2)):
                grid[r][c] = "X" if rng.random() < 0.5 else grid[r][c]
            positions = ((0, 0), (0, 3), (3, 0), (3, 3), (0, 1), (3, 2))
            self.assertEqual(build_distance_matrix(grid, positions), floyd_warshall(grid, positions))

    def test_random_matrices_ordering_cache_and_ties(self):
        # 此测试直接扰动收益矩阵以覆盖更丰富的博弈排序和等值界条件。
        rng = random.Random(4826)
        for _ in range(70):
            matrix = [[0 if i == j else rng.randint(1, 15) for j in range(6)]
                      for i in range(6)]
            for depth in (2, 3, 4):
                expected = exhaustive(depth, matrix)
                for order in ("forward", "reverse", "gain"):
                    for cache in (False, True):
                        result = solve_treasure(depth, order=order, use_cache=cache, distances=matrix)
                        self.assertEqual((result.best_target, result.best_value), expected)

    def test_exact_values_and_pv(self):
        for depth in (2, 4):
            values = exact_action_values(depth, distances=self.matrix)
            result = solve_treasure(depth, distances=self.matrix)
            self.assertEqual(result.best_value, max(values.values()))
            state = GameState()
            for target in result.pv:
                state = apply_move(state, target, self.matrix)
            self.assertEqual(state.utility, result.best_value)

    def test_optional_pv_does_not_change_search(self):
        for algorithm in ("minimax", "alphabeta"):
            with_pv = solve_treasure(4, algorithm=algorithm, distances=self.matrix)
            without_pv = solve_treasure(4, algorithm=algorithm, distances=self.matrix, build_pv=False)
            self.assertEqual(without_pv.pv, [])
            self.assertEqual(without_pv.best_target, with_pv.best_target)
            self.assertEqual(without_pv.best_value, with_pv.best_value)
            self.assertEqual(without_pv.stats, with_pv.stats)
            self.assertEqual(without_pv.root_values, with_pv.root_values)

    def test_cutoffs_only_count_skipped_siblings(self):
        # H=2 在每个根动作的完整窗口内没有兄弟可剪，计数应为零。
        for order in ("forward", "reverse", "gain"):
            self.assertEqual(solve_treasure(2, order=order, distances=self.matrix).stats.cutoffs, 0)
        self.assertEqual(solve_treasure(4, order="forward", distances=self.matrix).stats.cutoffs, 4)
        self.assertEqual(solve_treasure(4, order="reverse", distances=self.matrix).stats.cutoffs, 4)
        self.assertEqual(solve_treasure(4, order="gain", distances=self.matrix).stats.cutoffs, 8)

    def test_invalid_configuration_and_unreachable(self):
        for depth in (-1, 1.2, True):
            with self.assertRaises(ValueError):
                solve_treasure(depth, distances=self.matrix)
        with self.assertRaises(ValueError):
            solve_treasure(2, algorithm="unknown", distances=self.matrix)
        with self.assertRaises(ValueError):
            solve_treasure(2, order="unknown", distances=self.matrix)
        matrix = [[0 if i == j else -1 for j in range(6)] for i in range(6)]
        result = solve_treasure(4, distances=matrix)
        self.assertIsNone(result.best_target)
        self.assertEqual(result.best_value, 0)


if __name__ == "__main__":
    unittest.main()
