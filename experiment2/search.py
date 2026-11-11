"""A* for the two grid tasks, with explicit movement and counting rules.

Coordinates are (row, column). The start cell is never charged. A valid heap
pop, including the goal pop, counts as one expansion. ``generated`` counts
heap insertions (including the start), ``popped`` counts all heap removals,
and ``stale`` counts removals whose stored g is no longer the best known g.
Consequently expanded == popped - stale. Improved nodes may be reopened.

The default Manhattan lower bound is scaled by the minimum cell cost. If
entering the goal is free, the last step is removed from this lower bound.
Custom heuristics must be admissible and zero at the goal for optimality on
the first goal pop. Negative values at non-goal nodes are permitted.
"""

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from numbers import Real
from typing import Callable, Sequence

Position = tuple[int, int]
Heuristic = str | Callable[[Position, Position], Real]


@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0
    popped: int = 0
    stale: int = 0


@dataclass
class SearchResult:
    path: list[Position] = field(default_factory=list)
    cost: int = -1
    stats: SearchStats = field(default_factory=SearchStats)


def _cell_cost(cell: object, unit_cost: bool) -> int | None:
    if cell == "X":
        return None
    if isinstance(cell, bool):
        raise ValueError("Cell costs must be nonnegative integers, not bool.")
    if isinstance(cell, int):
        value = cell
    elif isinstance(cell, str):
        if cell in {".", "S", "E", "A", "B"} or (
            cell.startswith("T") and cell[1:].isdigit()
        ):
            value = 1
        else:
            try:
                value = int(cell)
            except ValueError as exc:
                raise ValueError(f"Unknown terrain {cell!r}.") from exc
    else:
        raise ValueError(f"Cell cost {cell!r} is not an integer or terrain label.")
    if value < 0:
        raise ValueError("A* requires nonnegative movement costs.")
    return 1 if unit_cost else value


def astar(
    grid: Sequence[Sequence[object]],
    start: Position,
    goal: Position,
    *,
    heuristic: Heuristic = "manhattan",
    goal_cost_zero: bool = False,
    unit_cost: bool = False,
) -> SearchResult:
    """Return a minimum-cost path, or ``SearchResult([], -1, stats)``.

    X is impassable. Integer cells (including integer strings) specify their
    entry cost; ., S, E, A, B, and T-number labels cost one. Neighbors are
    considered right, down, left, up. Heap ties use (f, h, insertion order),
    giving deterministic results. Invalid endpoint positions are unreachable;
    malformed grids or negative/nonintegral costs raise ValueError.
    """
    result = SearchResult()
    if not grid:
        return result
    rows, cols = len(grid), len(grid[0])
    if not cols:
        if any(len(row) != 0 for row in grid):
            raise ValueError("Grid must be rectangular.")
        return result
    if any(len(row) != cols for row in grid):
        raise ValueError("Grid must be rectangular.")
    costs = [[_cell_cost(cell, unit_cost) for cell in row] for row in grid]

    def valid(p: Position) -> bool:
        return (
            len(p) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in p)
            and 0 <= p[0] < rows
            and 0 <= p[1] < cols
            and costs[p[0]][p[1]] is not None
        )

    start, goal = tuple(start), tuple(goal)
    if not valid(start) or not valid(goal):
        return result
    if callable(heuristic):
        def h(p: Position) -> Real:
            value = heuristic(p, goal)
            if not isinstance(value, Real) or not isfinite(value):
                raise ValueError("Heuristic must return a finite real number.")
            return value
    elif heuristic == "zero":
        def h(p: Position) -> int:
            return 0
    elif heuristic == "manhattan":
        candidates = [
            cost
            for r, row in enumerate(costs)
            for c, cost in enumerate(row)
            if cost is not None and not (goal_cost_zero and (r, c) == goal)
        ]
        minimum = min(candidates, default=0)

        def h(p: Position) -> int:
            distance = abs(p[0] - goal[0]) + abs(p[1] - goal[1])
            return minimum * max(0, distance - int(goal_cost_zero))
    else:
        raise ValueError("heuristic must be 'zero', 'manhattan', or callable.")

    if h(goal) != 0:
        raise ValueError("Heuristic must be zero at the goal.")

    serial = count()
    start_h = h(start)
    frontier = [(start_h, start_h, next(serial), 0, start)]
    best_g = {start: 0}
    parent: dict[Position, Position] = {}
    result.stats.generated = 1
    while frontier:
        _, _, _, g, current = heappop(frontier)
        result.stats.popped += 1
        if g != best_g.get(current):
            result.stats.stale += 1
            continue
        result.stats.expanded += 1
        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            result.path = path[::-1]
            result.cost = g
            return result
        row, col = current
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            entry = costs[nr][nc]
            if entry is None:
                continue
            neighbor = (nr, nc)
            candidate_g = g + (0 if goal_cost_zero and neighbor == goal else entry)
            if candidate_g >= best_g.get(neighbor, float("inf")):
                continue
            best_g[neighbor] = candidate_g
            parent[neighbor] = current
            neighbor_h = h(neighbor)
            heappush(
                frontier,
                (candidate_g + neighbor_h, neighbor_h, next(serial), candidate_g, neighbor),
            )
            result.stats.generated += 1
    return result


def maze_path(start: Position, goal: Position, grid: Sequence[Sequence[object]]) -> list[Position]:
    """Task 1: return a shortest four-neighbor coordinate sequence."""
    return astar(grid, start, goal, unit_cost=True).path


def weighted_cost(start: Position, goal: Position, grid: Sequence[Sequence[object]]) -> int:
    """Task 2: return the optimal cost with a free final goal entry."""
    return astar(grid, start, goal, goal_cost_zero=True).cost
