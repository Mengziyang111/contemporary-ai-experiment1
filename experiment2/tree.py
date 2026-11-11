"""Left-to-right Minimax and Alpha-Beta, with comparable search counters.

``visited`` includes the root, internal nodes, and visited leaves. ``cutoffs``
counts internal nodes at which one or more remaining siblings are skipped.
The optional ideal order uses exact subtree values from a full preprocessing
pass. Its ``ordering_visited`` counter is reported separately: an oracle order
is a demonstration of ordering potential, not a free optimization.
"""

from dataclasses import dataclass, field
from math import inf, isfinite
from numbers import Real


@dataclass
class TreeNode:
    value: Real | None = None
    children: list["TreeNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.children = list(self.children) if self.children is not None else []


@dataclass
class TreeStats:
    visited: int = 0
    leaves: int = 0
    cutoffs: int = 0
    ordering_visited: int = 0


@dataclass
class TreeSearchResult:
    value: Real
    stats: TreeStats


def _leaf_value(node: TreeNode) -> Real:
    if not isinstance(node.value, Real) or not isfinite(node.value):
        raise ValueError("Every leaf needs a finite numeric value.")
    return node.value


def _solve(root: TreeNode, is_maximizing: bool, order: str, pruning: bool) -> TreeSearchResult:
    if order not in {"forward", "reverse", "ideal"}:
        raise ValueError("order must be 'forward', 'reverse', or 'ideal'.")
    stats = TreeStats()
    ordering_values: dict[tuple[int, bool], Real] = {}

    def preprocess(node: TreeNode, maximizing: bool) -> Real:
        stats.ordering_visited += 1
        if not node.children:
            answer = _leaf_value(node)
        else:
            values = [preprocess(child, not maximizing) for child in node.children]
            answer = max(values) if maximizing else min(values)
        ordering_values[(id(node), maximizing)] = answer
        return answer

    if order == "ideal":
        preprocess(root, is_maximizing)

    def visit(node: TreeNode, maximizing: bool, alpha: Real, beta: Real) -> Real:
        stats.visited += 1
        if not node.children:
            stats.leaves += 1
            return _leaf_value(node)
        children = node.children
        if order == "reverse":
            children = list(reversed(children))
        elif order == "ideal":
            children = sorted(
                children,
                key=lambda child: ordering_values[(id(child), not maximizing)],
                reverse=maximizing,
            )
        value = -inf if maximizing else inf
        for index, child in enumerate(children):
            candidate = visit(child, not maximizing, alpha, beta)
            if maximizing:
                value = max(value, candidate)
                alpha = max(alpha, value)
            else:
                value = min(value, candidate)
                beta = min(beta, value)
            if pruning and alpha >= beta and index + 1 < len(children):
                stats.cutoffs += 1
                break
        return value

    return TreeSearchResult(visit(root, is_maximizing, -inf, inf), stats)


def minimax(root: TreeNode, *, is_maximizing: bool = True, order: str = "forward") -> TreeSearchResult:
    """Evaluate the complete tree; children alternate MAX and MIN roles."""
    return _solve(root, is_maximizing, order, False)


def alpha_beta(root: TreeNode, *, is_maximizing: bool = True, order: str = "forward") -> TreeSearchResult:
    """Evaluate with Alpha-Beta pruning; the default order is left to right."""
    return _solve(root, is_maximizing, order, True)


def best_score(root: TreeNode) -> Real:
    """Task 3's required root-to-score function."""
    return alpha_beta(root).value
