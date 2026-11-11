"""扩展实验：固定终点的精确启发函数。预处理开销与查询开销分别记录。"""

import heapq
from itertools import count

from search import _cell_cost


def reverse_distance_heuristic(grid, goal, *, goal_cost_zero=False):
    """在反向图执行一次 Dijkstra，返回精确 h 与预处理扩展节点数。

    正向边 u->v 的权重为 cost(v)，所以反向从 v 松弛 u 时，
    必须继续使用 cost(v)，不能错误改成 cost(u)。
    """
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("预处理需要非空矩形地图")
    rows, cols = len(grid), len(grid[0])
    costs = [[_cell_cost(cell, False) for cell in row] for row in grid]
    goal = tuple(goal)
    if (len(goal) != 2 or any(not isinstance(x, int) or isinstance(x, bool) for x in goal)
            or not (0 <= goal[0] < rows and 0 <= goal[1] < cols)
            or costs[goal[0]][goal[1]] is None):
        raise ValueError("终点必须是地图内的可通行坐标")

    def cost(pos):
        if goal_cost_zero and pos == goal:
            return 0
        return costs[pos[0]][pos[1]]

    distances = {goal: 0}
    serial = count()
    queue = [(0, next(serial), goal)]
    expanded = 0
    while queue:
        d, _, current = heapq.heappop(queue)
        if d != distances[current]:
            continue
        expanded += 1
        r, c = current
        for nr, nc in ((r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)):
            if not (0 <= nr < rows and 0 <= nc < cols) or costs[nr][nc] is None:
                continue
            neighbor = (nr, nc)
            candidate = d + cost(current)
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, next(serial), neighbor))

    def heuristic(pos, requested_goal):
        if requested_goal != goal:
            raise ValueError("预处理启发函数只能用于原固定终点")
        # 与终点不连通时 0 仍是安全下界。
        return distances.get(pos, 0)

    return heuristic, expanded
