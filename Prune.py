
from sklearn.tree._tree import TREE_LEAF


class Prune:

    def prune_index(self, inner_tree, index, threshold):
        if inner_tree.value[index].min() < threshold:
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
        # if there are shildren, visit them as well
        if inner_tree.children_left[index] != TREE_LEAF:
            self.prune_index(inner_tree, inner_tree.children_left[index], threshold)
            self.prune_index(inner_tree, inner_tree.children_right[index], threshold)
