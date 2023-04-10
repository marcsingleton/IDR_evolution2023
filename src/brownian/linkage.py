"""Functions for manipulating linkage matrices returned by the SciPy linkage module."""

import skbio


def make_tree(lm):
    """Returns a linkage matrix represented as a scikit-bio TreeNode.

    Parameters
    ----------
    lm: ndarray
        Linkage matrix which encodes clustering of n nodes into a
        (n-1) x 4 matrix.

    Returns
    -------
    tree: TreeNode
        Linkage matrix represented as a TreeNode.
    """
    num_tips = len(lm) + 1
    nodes = {node_id: skbio.TreeNode(name=node_id, children=[]) for node_id in range(num_tips)}
    heights = {node_id: 0 for node_id in range(2*num_tips-1)}
    for idx in range(len(lm)):
        node_id = idx + num_tips
        child_id1, child_id2, distance, _ = lm[idx]
        child1, child2 = nodes[child_id1], nodes[child_id2]
        height1, height2 = heights[child_id1], heights[child_id2]
        child1.length = distance - height1
        child2.length = distance - height2
        parent = skbio.TreeNode(name=node_id, children=[child1, child2])
        nodes[node_id] = parent
        heights[node_id] = distance
    tree = nodes[2*(num_tips-1)]
    return tree
