"""Independent audit of exact reverse heuristics and treasure-game values."""

import heapq
import random
import unittest

from extra_heuristics import reverse_distance_heuristic
from fixtures import tree_from_dict
from search import astar
from treasure import GameState, INDEX, LABELS, VALUES, solve_treasure
from tree import alpha_beta


def dijkstra_reference(grid, start, goal, free_goal=False):
    """Forward directed edges use the entry cost of their destination."""
    if grid[start[0]][start[1]] == "X" or grid[goal[0]][goal[1]] == "X":
        return -1
    frontier = [(0, start)]
    settled = set()
    while frontier:
        value, current = heapq.heappop(frontier)
        if current in settled:
            continue
        settled.add(current)
        if current == goal:
            return value
        r, c = current
        for neighbor in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            nr, nc = neighbor
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                continue
            cell = grid[nr][nc]
            if cell == "X":
                continue
            try:
                step = int(cell)
            except (ValueError, TypeError):
                step = 1
            if free_goal and neighbor == goal:
                step = 0
            heapq.heappush(frontier, (value + step, neighbor))
    return -1


def independent_game_values(depth, state, matrix):
    """Evaluate directly from immutable scalar tuples, without module moves."""
    def recurse(a, b, left, sa, sb, player, remaining_depth):
        position = a if player == "A" else b
        moves = [t for t in left if matrix[INDEX[position]][INDEX[t]] >= 0]
        if not remaining_depth or not moves:
            return sa-sb
        children = []
        for t in moves:
            gain = VALUES[t] - matrix[INDEX[position]][INDEX[t]]
            rest = tuple(x for x in left if x != t)
            if player == "A":
                children.append(recurse(t, b, rest, sa+gain, sb, "B", remaining_depth-1))
            else:
                children.append(recurse(a, t, rest, sa, sb+gain, "A", remaining_depth-1))
        return (max if player == "A" else min)(children)

    if not depth:
        return {}
    result = {}
    for t in state.remaining:
        distance = matrix[INDEX[state.a_pos]][INDEX[t]]
        if distance < 0:
            continue
        rest = tuple(x for x in state.remaining if x != t)
        result[t] = recurse(t, state.b_pos, rest,
                            state.score_a+VALUES[t]-distance, state.score_b,
                            "B", depth-1)
    return result


class ReverseHeuristicAudit(unittest.TestCase):
    def test_direction_of_entry_weights(self):
        grid = [[9, 2, 7]]
        h, expanded = reverse_distance_heuristic(grid, (0, 2))
        self.assertEqual(expanded, 3)
        self.assertEqual(h((0, 0), (0, 2)), 9)
        self.assertEqual(h((0, 1), (0, 2)), 7)
        self.assertEqual(h((0, 2), (0, 2)), 0)
        h_reverse, _ = reverse_distance_heuristic(grid, (0, 0))
        self.assertEqual(h_reverse((0, 2), (0, 0)), 11)

    def test_free_goal_zero_weights_and_start_equals_goal(self):
        for free_goal in (False, True):
            grid = [[0, 0, 8], [2, "X", 1]]
            goal = (0, 2)
            h, _ = reverse_distance_heuristic(grid, goal, goal_cost_zero=free_goal)
            for point in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)):
                expected = dijkstra_reference(grid, point, goal, free_goal)
                self.assertEqual(h(point, goal), expected)
                self.assertEqual(astar(grid, point, goal, heuristic=h,
                                       goal_cost_zero=free_goal).cost, expected)

    def test_disconnected_zero_fallback_and_goal_mismatch(self):
        grid = [[1, "X", 2]]
        h, expanded = reverse_distance_heuristic(grid, (0, 2))
        self.assertEqual(expanded, 1)
        self.assertEqual(h((0, 0), (0, 2)), 0)
        self.assertEqual(astar(grid, (0, 0), (0, 2), heuristic=h).cost, -1)
        with self.assertRaises(ValueError):
            h((0, 0), (0, 1))

    def test_random_directed_costs_against_independent_dijkstra(self):
        rng = random.Random(271828)
        for _ in range(80):
            grid = [["X" if rng.random() < .2 else rng.randrange(10)
                     for _ in range(5)] for _ in range(4)]
            goal = (rng.randrange(4), rng.randrange(5))
            grid[goal[0]][goal[1]] = rng.randrange(10)
            for free_goal in (False, True):
                h, _ = reverse_distance_heuristic(grid, goal, goal_cost_zero=free_goal)
                for r in range(4):
                    for c in range(5):
                        if grid[r][c] == "X":
                            continue
                        point = (r, c)
                        expected = dijkstra_reference(grid, point, goal, free_goal)
                        self.assertEqual(h(point, goal), max(expected, 0))
                        result = astar(grid, point, goal, heuristic=h,
                                       goal_cost_zero=free_goal)
                        self.assertEqual(result.cost, expected)

    def test_numeric_string_entry_costs_match_main_parser(self):
        grid = [["S", "+2", " 3 ", "E"]]
        h, _ = reverse_distance_heuristic(grid, (0, 3))
        self.assertEqual(h((0, 0), (0, 3)), 6)

    def test_invalid_terrain_and_goals_rejected_before_search(self):
        invalid_terrain = (
            ([], (0, 0)), ([[]], (0, 0)),
            ([[1, 2], [1]], (0, 0)),
            ([[1, -1]], (0, 0)), ([[1, "-2"]], (0, 0)),
            ([[1, .5]], (0, 0)), ([[1, True]], (0, 0)),
            ([[1, "unknown"]], (0, 0)),
        )
        for grid, goal in invalid_terrain:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                reverse_distance_heuristic(grid, goal)
        grid = [[1, "X"], [1, 1]]
        for goal in ((0, 1), (-1, 0), (2, 0), (0, 3), (True, 0), (0,), (0.0, 0)):
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                reverse_distance_heuristic(grid, goal)

    def test_nonzero_goal_heuristic_rejected(self):
        with self.assertRaises(ValueError):
            astar([[1, 1]], (0, 0), (0, 1), heuristic=lambda p, g: -1)


class TreasureIndependentAudit(unittest.TestCase):
    def test_random_matrices_cache_windows_and_each_root_action(self):
        rng = random.Random(314159)
        for trial in range(80):
            matrix = [[0 if i == j else rng.randrange(13) for j in range(6)]
                      for i in range(6)]
            if trial % 3 == 0:
                matrix = [[-1 if rng.random() < .12 and i != j else value
                           for j, value in enumerate(row)]
                          for i, row in enumerate(matrix)]
            remaining = tuple(t for t in VALUES if rng.random() < .85)
            state = GameState(rng.choice(LABELS), rng.choice(LABELS), remaining,
                              rng.randrange(-5, 6), rng.randrange(-5, 6), "A")
            depth = trial % 6
            expected_actions = independent_game_values(depth, state, matrix)
            expected_value = max(expected_actions.values(), default=state.utility)
            expected_target = min((t for t, v in expected_actions.items()
                                   if v == expected_value), default=None)
            for algorithm in ("minimax", "alphabeta"):
                for order in ("forward", "reverse", "gain"):
                    for cache in (False, True):
                        result = solve_treasure(depth, algorithm=algorithm, order=order,
                                                use_cache=cache, distances=matrix,
                                                initial_state=state)
                        self.assertEqual((result.best_target, result.best_value),
                                         (expected_target, expected_value))
                        self.assertEqual({r.target:r.value for r in result.root_values},
                                         expected_actions)
                        self.assertTrue(all(r.bound == "exact" for r in result.root_values))

    def test_json_tree_roles(self):
        root = tree_from_dict({"tree_structure": {"is_maximizing_player": True,
                    "children": [{"value": 3}, {"value": 5}]}})
        self.assertEqual(alpha_beta(root).value, 5)
        with self.assertRaises(ValueError):
            tree_from_dict({"is_maximizing_player": False, "value": 1})


if __name__ == "__main__":
    unittest.main()
