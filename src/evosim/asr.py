"""Common functions for ASR."""

import numpy as np
import scipy.linalg as linalg
import skbio
from numpy import log


def get_conditional(tree, matrix, inplace=False):
    """Return conditional probabilities of tree given tips and node state."""
    if not inplace:
        tree = tree.copy()  # Make copy so computations do not change original tree

    for node in tree.postorder():
        if node.is_tip():
            node.s = np.zeros(node.value.shape[1])
            node.conditional = node.value
        else:
            ss, ps = [], []
            for child in node.children:
                s, conditional = child.s, child.conditional
                m = linalg.expm(matrix * child.length)
                p = np.matmul(m, conditional)

                ss.append(s)
                ps.append(p)

            conditional = np.product(np.stack(ps), axis=0)
            s = conditional.sum(axis=0)
            node.conditional = conditional / s  # Normalize to 1 to prevent underflow
            node.s = log(s) + np.sum(np.stack(ss), axis=0)  # Pass forward scaling constant in log space

    return tree.s, tree.conditional


def get_tree(tree1, tree2):
    """Root fitted tree such that root creates splits compatible with original tree.

    Since an arbitrarily rooted tree can split only one of the two true root clades, we check for both when traversing
    the tree. The root is temporarily set as the parent node of the first of the root clades that is found. As the
    re-rooted tree has three children at the root, this has the effect of splitting that root clade while preserving the
    other root clade. For simplicity, the unsplit root clade is designated as nodeA and the other as nodeB. Once these
    labels are defined, the new root is introduced between these nodes with a small amount of "tree surgery."
    """
    # Find split and calculate ratio
    tree1 = tree1.shear([tip.name for tip in tree2.tips()])
    nodeA, nodeB = tree1.children
    tipsA, tipsB = {tip.name for tip in nodeA.tips()}, {tip.name for tip in nodeB.tips()}
    for node in tree2.traverse():
        tips = {tip.name for tip in node.tips()}
        if tips == tipsA:
            nodeA, nodeB = nodeB, nodeA  # Swap labels so nodeA is the preserved root clade
            ratio = nodeA.length / (nodeA.length + nodeB.length)
            break
        elif tips == tipsB:
            ratio = nodeA.length / (nodeA.length + nodeB.length)
            break
    tree2 = tree2.root_at(node)

    # Insert root node
    nodeB1, nodeB2 = None, None
    for child in tree2.children:
        if child.count(tips=True) == nodeA.count(tips=True):
            nodeA = child
        elif nodeB1 is None:
            nodeB1 = child
        else:
            nodeB2 = child
    lengthA = ratio * nodeA.length
    lengthB = (1 - ratio) * nodeA.length

    nodeA = skbio.TreeNode(children=nodeA.children, length=lengthA)
    nodeB = skbio.TreeNode(children=[nodeB1, nodeB2], length=lengthB)
    tree = skbio.TreeNode(children=[nodeA, nodeB])

    return tree
