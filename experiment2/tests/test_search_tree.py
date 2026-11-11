"""Independent reference checks for A* and Alpha-Beta; no third-party packages."""

from collections import deque
from heapq import heappop, heappush
import random
import unittest

from search import astar, maze_path, weighted_cost
from tree import TreeNode, alpha_beta, best_score, minimax


MAZE = [list(row) for row in ("S..X.", ".X...", ".X.X.", "...XE", "XX...")]
WEIGHTED = [["S", 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1], [1, 1, 1, 8, 1], [8, 8, 1, 1, "E"]]


def reference_distance(grid, start, goal, unit=False, free_goal=False):
    """Independent Dijkstra implementation; stores settled distances only."""
    queue = [(0, start)]
    settled = {}
    while queue:
        distance, point = heappop(queue)
        if point in settled:
            continue
        settled[point] = distance
        if point == goal:
            return distance
        r, c = point
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                continue
            if grid[nr][nc] == "X" or (nr, nc) in settled:
                continue
            step = 1 if unit else int(grid[nr][nc])
            if free_goal and (nr, nc) == goal:
                step = 0
            heappush(queue, (distance + step, (nr, nc)))
    return -1


def reference_bfs(grid, start, goal):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), depth = queue.popleft()
        if (r, c) == goal:
            return depth
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            p = (nr, nc)
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "X" and p not in seen:
                seen.add(p)
                queue.append((p, depth + 1))
    return -1


def sample_tree():
    a = TreeNode(children=[TreeNode(3), TreeNode(5)])
    b = TreeNode(children=[TreeNode(6), TreeNode(9)])
    c = TreeNode(children=[TreeNode(1), TreeNode(2)])
    return TreeNode(children=[TreeNode(children=[a, b]), TreeNode(children=[b, c])])


def naive_minimax(node, maximizing=True):
    if not node.children:
        return node.value
    values = [naive_minimax(c, not maximizing) for c in node.children]
    return max(values) if maximizing else min(values)


class AStarTests(unittest.TestCase):
    def test_given_maze(self):
        path = maze_path((0, 0), (3, 4), MAZE)
        self.assertEqual(len(path) - 1, 7)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 4))
        self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(path,path[1:])))
        self.assertTrue(all(MAZE[r][c] != "X" for r,c in path))

    def test_given_weighted(self):
        result = astar(WEIGHTED, (0, 0), (4, 4), goal_cost_zero=True)
        self.assertEqual(result.cost, 7)
        self.assertEqual(weighted_cost((0, 0), (4, 4), WEIGHTED), 7)
        self.assertEqual(result.stats.expanded, result.stats.popped-result.stats.stale)

    def test_unreachable_and_invalid_endpoints(self):
        grid = [[1, "X", 1]]
        self.assertEqual(astar(grid, (0, 0), (0, 2)).cost, -1)
        self.assertEqual(astar(grid, (0, 0), (0, 2)).path, [])
        self.assertEqual(maze_path((0, 0), (0, 2), grid), [])
        self.assertEqual(astar(grid, (0, 1), (0, 2)).cost, -1)
        self.assertEqual(astar(grid, (-1, 0), (0, 2)).cost, -1)
        self.assertEqual(astar([], (0, 0), (0, 0)).cost, -1)

    def test_start_equals_goal_and_free_adjacent_goal(self):
        same = astar([[8]], (0, 0), (0, 0))
        self.assertEqual((same.cost, same.path), (0, [(0, 0)]))
        self.assertEqual(same.stats.expanded, 1)
        self.assertEqual(astar([[1, 9]], (0, 0), (0, 1), goal_cost_zero=True).cost, 0)
        self.assertEqual(astar([[1, 9]], (0, 0), (0, 1)).cost, 9)

    def test_input_validation(self):
        for grid in ([[1, -1]], [[1, 0.5]], [[1, 2], [1]], [[1, True]], [[1, "bad"]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                astar(grid, (0, 0), (0, 1))
        with self.assertRaises(ValueError):
            astar([[1, 1]], (0, 0), (0, 1), heuristic="invalid")

    def test_random_maze_against_bfs(self):
        rng = random.Random(20261107)
        for _ in range(150):
            grid = [["X" if rng.random() < .28 else 1 for _ in range(7)] for _ in range(6)]
            grid[0][0] = grid[-1][-1] = 1
            expected = reference_bfs(grid, (0, 0), (5, 6))
            for heuristic in ("zero", "manhattan"):
                actual = astar(grid, (0, 0), (5, 6), unit_cost=True, heuristic=heuristic)
                self.assertEqual(actual.cost, expected)

    def test_random_terrain_against_dijkstra(self):
        rng = random.Random(42)
        for _ in range(150):
            grid = [["X" if rng.random() < .18 else rng.randrange(0, 10) for _ in range(6)] for _ in range(5)]
            grid[0][0] = grid[-1][-1] = 1
            for free_goal in (False, True):
                expected = reference_distance(grid, (0, 0), (4, 5), free_goal=free_goal)
                actual = astar(grid, (0, 0), (4, 5), goal_cost_zero=free_goal)
                self.assertEqual(actual.cost, expected)
                if actual.cost >= 0:
                    cost = sum(int(grid[r][c]) for r,c in actual.path[1:])
                    if free_goal and len(actual.path) > 1:
                        cost -= int(grid[4][5])
                    self.assertEqual(actual.cost, cost)

    def test_inconsistent_admissible_heuristic_reopens(self):
        grid = [[1, 8, 1, 1], [1, 1, 1, 1]]
        goal = (0, 3)
        # Negative lower bounds remain admissible and force a costly early pop.
        def h(p, target):
            return -100 if p in {(0, 1), (0, 2)} else 0
        result = astar(grid, (0, 0), goal, heuristic=h)
        self.assertEqual(result.cost, reference_distance(grid, (0, 0), goal))
        self.assertEqual(result.cost, 5)


class GameTreeTests(unittest.TestCase):
    def test_given_tree(self):
        root = sample_tree()
        self.assertEqual(best_score(root), 5)
        for order in ("forward", "reverse", "ideal"):
            plain = minimax(root, order=order)
            pruned = alpha_beta(root, order=order)
            self.assertEqual((plain.value, pruned.value), (5, 5))
            self.assertLessEqual(pruned.stats.visited, plain.stats.visited)
        self.assertEqual(minimax(root).stats.visited, 15)
        self.assertEqual(minimax(root).stats.leaves, 8)

    def test_single_leaf_negative_values_and_validation(self):
        self.assertEqual(alpha_beta(TreeNode(-7)).value, -7)
        root = TreeNode(children=[TreeNode(-9), TreeNode(-3)])
        self.assertEqual(alpha_beta(root).value, -3)
        self.assertEqual(alpha_beta(root, is_maximizing=False).value, -9)
        with self.assertRaises(ValueError):
            alpha_beta(TreeNode())
        with self.assertRaises(ValueError):
            alpha_beta(root, order="invalid")

    def test_random_trees_against_independent_minimax(self):
        rng = random.Random(99)
        def make(depth):
            if depth == 0 or rng.random() < .2:
                return TreeNode(rng.randrange(-30, 31))
            return TreeNode(children=[make(depth-1) for _ in range(rng.randrange(1, 5))])
        for _ in range(150):
            root = make(4)
            expected = naive_minimax(root)
            for order in ("forward", "reverse", "ideal"):
                self.assertEqual(minimax(root, order=order).value, expected)
                self.assertEqual(alpha_beta(root, order=order).value, expected)

    def test_equal_bounds_prune_and_order_does_not_mutate(self):
        root = sample_tree()
        before = [id(c) for c in root.children]
        alpha_beta(root, order="reverse")
        ideal = alpha_beta(root, order="ideal")
        self.assertEqual(before, [id(c) for c in root.children])
        self.assertGreater(ideal.stats.ordering_visited, 0)
        self.assertGreater(alpha_beta(root).stats.cutoffs, 0)


if __name__ == "__main__":
    unittest.main()
