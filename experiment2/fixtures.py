"""题目原始数据。题四的特殊格进入代价是 1，题二的 E 则是 0。"""

MAZE_START = (0, 0)
MAZE_GOAL = (3, 4)
MAZE = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
WEIGHTED_START = (0, 0)
WEIGHTED_GOAL = (4, 4)
WEIGHTED = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]


def sample_tree():
    from tree import TreeNode

    def pair(a, b):
        return TreeNode(children=[TreeNode(value=a), TreeNode(value=b)])

    return TreeNode(children=[
        TreeNode(children=[pair(3, 5), pair(6, 9)]),
        TreeNode(children=[pair(6, 9), pair(1, 2)]),
    ])


def tree_from_dict(data):
    """JSON 中各层角色按根 MAX 逐层交替；可选角色字段必须一致。"""
    from tree import TreeNode

    def build(node, is_max=True):
        if "is_maximizing_player" in node and node["is_maximizing_player"] != is_max:
            raise ValueError("角色字段必须从根 MAX 开始逐层交替")
        children = node.get("children", [])
        if children:
            return TreeNode(children=[build(child, not is_max) for child in children])
        if "value" not in node:
            raise ValueError("叶子节点必须包含 value")
        return TreeNode(value=node["value"])

    return build(data.get("tree_structure", data))
