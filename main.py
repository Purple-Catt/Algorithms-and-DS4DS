import time
from functools import cmp_to_key
from math import floor, ceil
from numpy import inf
from numpy.random import randint


def selection_sort(A: list, display_time: bool = True):
    """Run the selection sort algorithm over a given list.
    Parameters:
        A: list to be sorted
        display_time: Optional, default True. If True the execution time is printed out.
    Returns:
        Sorted list."""
    if display_time:
        start = time.time()

    for i in range(len(A)):
        min_pos = i

        for j in range(i+1, len(A)):
            if A[j] < A[min_pos]:
                min_pos = j

        temp = A[min_pos]
        A[min_pos] = A[i]
        A[i] = temp

    if display_time:
        print(f"Execution time: {time.time() - start}")

    return A


def insertion_sort(A: list, cmp=lambda val, k: val - k, display_time: bool = True):
    """Run the insertion sort algorithm over a given list.
    Parameters:
        A: list to be sorted
        cmp: function, default None. If given, it's used as a comparator into the algorithm
        display_time: Optional, default True. If True the execution time is printed out.
    Returns:
        Sorted list."""
    if display_time:
        start = time.time()

    for j in range(1, len(A)):
        key = A[j]
        i = j - 1

        while i >= 0 and cmp(A[i], key) > 0:
            A[i + 1] = A[i]
            i -= 1

        A[i + 1] = key

    if display_time:
        print(f"Execution time: {time.time() - start}")

    return A


def even_odd_comparator(a: int | float, b: int | float):
    """Simple comparator that follows this statement: <Even numbers precede odd ones. Even numbers are sorted in
    non-decreasing order while odd ones are sorted in non-increasing order.>"""
    if a % 2 == b % 2 == 0:
        return a - b

    elif a % 2 == b % 2 == 1:
        return b - a

    elif a % 2 == 0:
        return -1

    else:
        return 1


def str_len_comparator(a: str, b: str):
    if len(a) > len(b):
        return -1

    elif len(a) < len(b):
        return 1

    elif a >= b:
        return -1

    else:
        return 1


def intersection_slow(l1: list, l2: list):
    intersection_list = list()
    for item in l1:
        if item in l2:
            intersection_list.append(item)

    return intersection_list


def intersection(l1: list, l2: list):
    intersection_list = list()
    i = 0
    j = 0
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            i += 1

        elif l2[j] < l1[i]:
            j += 1

        else:
            intersection_list.append(l1[i])
            i += 1
            j += 1

    return intersection_list


def test_sortedness(l1: list, sorting_algo, cmp=None):
    """Test the correctness of the sorting algorithm comparing the obtained result with the builtin Python function
    sorted.
    Parameters:
        l1: unsorted list
        sorting_algo: algorithm function to evaluate
        cmp: Optional, default None. If given, this comparator will be used in both the algorithms.
    Returns:
        True if the algorithm works correctly, False otherwise."""
    if cmp is not None:
        assert sorting_algo(l1, cmp=cmp) == sorted(l1, key=cmp_to_key(cmp)), "Must be increasing!"

    else:
        assert sorting_algo(l1) == sorted(l1), "Must be increasing!"

    print("Test passed!")


def test_intersection(func):
    """Test the correctness of an intersection function.
    Parameters:
        func: function that computes the intersection given two lists in the form function(l1, l2) -> list.
    Returns:
        List containing the intersection between the two input lists."""
    l1 = sorted([3, 5, 1, 2])
    l2 = sorted([1, 4, 6, 2])

    assert set(func(l1, l2)) == {1, 2}, "Intersection function doesn't work!"

    print("Test passed!")


class SearchEngine:
    """A base structure for a simple search engine. Starting from a list containing text documents, with space-separated
         terms, an inverted index is created. The inverted index stores a list for each term of the collection that
         contains the identifiers of all the documents containing that term, sorted.
         Parameters:
             collection: list containing text documents."""
    def __init__(self, collection: list):

        self.collection = dict()
        self.inverted_index = dict()

        # Transform the given list in a dictionary
        for key, value in enumerate(collection):
            self.collection[key] = value.split()

        for key in self.collection.keys():
            for term in self.collection[key]:

                # If it's not the first occurrence of the term, append the key in the existing list
                if term in self.inverted_index.keys():
                    # This 'if else' statement deals with multiple occurrence of the same term in a text
                    if key in self.inverted_index[term]:
                        pass

                    else:
                        self.inverted_index[term].append(key)
                        self.inverted_index[term].sort()

                # If it's the first occurrence of the term, the relative list is created
                else:
                    self.inverted_index[term] = [key]

    def query(self, a: str, b: str):
        """AND query function between two terms that reports all the documents containing both of them.
        Parameters:
            a: first term
            b: second term.
        Returns:
            A list that reports the indexes to the documents that contains the input terms."""
        intersec = intersection(self.inverted_index[a], self.inverted_index[b])

        return intersec


def test_search_engine(coll):
    search_eng = SearchEngine(collection=coll)
    true_inv_index = {'dog': [0, 2, 3], 'cat': [0, 1, 3], 'monkey': [1, 2], 'cow': [2], 'fish': [3]}

    assert search_eng.inverted_index == true_inv_index, "Wrong implementation!"

    assert search_eng.query("dog", "cat") == [0, 3], "Wrong query function!"

    print("Test passed!")


def merge(A: list, p: int, q: int, r: int):
    """Function used by the MergeSort algorithm to merge two (sorted) arrays.
    Parameters:
        A: list (semi) sorted
        p: integer representing the initial index of the sub-array considered
        q: integer representing the middle point between p and r
        r: integer representing the final index of the sub-array considered
    Returns:
        None. Each operation and data manipulation is made directly in the original list."""
    L = A[p:q+1]
    R = A[q+1:r+1]
    # Add sentinels at the end of the new lists, Numpy infinity has been used
    L.append(inf)
    R.append(inf)
    i = 0
    j = 0

    for k in range(p, r+1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1

        else:
            A[k] = R[j]
            j += 1


def merge_sort(A: list, p: int, r: int):
    """MergeSort algorithm implementation.
    Parameters:
        A: unsorted list
        p: integer representing the initial index of the sub-array considered. Note that it must be set to 0 when the
        function is called by the user in the first step
        r: integer representing the final index of the sub-array considered. It must be set to len(A) - 1 when the
        function is called by the user in the first step.
    Returns:
        The sorted list"""
    if p < r:
        q = floor((p + r) / 2)

        merge_sort(A, p, q)
        merge_sort(A, q+1, r)
        merge(A, p, q, r)

    return A


def partition(A: list, p: int, r: int):
    """Partition function used to compute the pivot as required by the QuickSort algorithm. Note that the possibility
    to choose a different type of pivot - like the last one, or the median - has voluntarily not been implemented
    into this function to preserve the theoretical running time of this specific algorithm. If a different type of pivot
    is necessary please create another, analogue, function or modify the code directly here on
    row 'x = randint(0, len(A))'.
    IMPORTANT NOTE: even if they seem to be the same, the numpy implementation of randint excludes the last element
    while the random implementation includes it. In this function the numpy implementation has been used, different
    ones will lead to error in the results."""
    # x represents the pivot, in this implementation a random one is chosen (different approaches may be considered)
    rand = randint(p, r+1)
    x = A[rand]
    A[r], A[rand] = A[rand], A[r]

    i = p - 1
    # Iterate over all the considered range [p, r), the purpose is to obtain an array with all the elements lower than
    # pivot to its __left while the greater stays on its __right
    for j in range(p, r):
        # Comparison of the element in the j-th position with pivot
        if A[j] <= x:
            # Smaller elements will be swapped with the greater highlighted by 'i'
            i += 1
            A[i], A[j] = A[j], A[i]

    # The highest number of the considered sub-array is now highlighted by 'i', so it's swapped with pivot
    A[i+1], A[r] = A[r], A[i+1]

    return i + 1


def quick_sort(A: list, p: int, r: int):
    """QuickSort algorithm implementation.
    Parameters:
        A: unsorted list
        p: integer representing the initial index of the sub-array considered. Note that it must be set to 0 when the
        function is called by the user in the first step
        r: integer representing the final index of the sub-array considered. It must be set to len(A) - 1 when the
        function is called by the user in the first step.
    Returns:
        The sorted list"""
    # This 'if' allow the algorithm to do recursive calls up until the sub-array considered contains only one value
    if p < r:
        # 'q' is here pointing the chosen pivot computed by the partition function, so that on its __left-side
        # there are all the smaller elements while on the __right-side the bigger ones.
        q = partition(A, p, r)
        # Recursive call considering the __left-side of pivot
        quick_sort(A, p, q-1)
        # Recursive call considering the __right-side of pivot
        quick_sort(A, q+1, r)

    return A


def k_largest_quick_select(A: list, p: int, r: int, k: int):
    """QuickSearch algorithm implementation.
    Parameters:
        A: unsorted list, each element must appear at most once
        p: integer representing the initial index of the sub-array considered. Note that it must be set to 0 when the
        function is called by the user in the first step
        r: integer representing the final index of the sub-array considered. It must be set to len(A) - 1 when the
        function is called by the user in the first step
        k: integer representing the largest k-th element to find. It follows the index of the list, so i=0
        refers to the first smallest element and so on.
    Returns:
        Sorted list containing the k largest elements of the given list."""
    if p == r:
        return sorted(A[p:])

    q = partition(A, p, r)
    i = r - q + 1
    if k == i:
        return sorted(A[q:])

    elif k > i:
        return k_largest_quick_select(A, p, q - 1, k - i)

    else:
        return k_largest_quick_select(A, q + 1, r, k)


def k_largest_sort(A: list, k: int):
    return sorted(A, reverse=True)[:k]


def test_sortednesss(sorting_algo, cmp=None):
    """Test the correctness of the sorting algorithm comparing the obtained result with the builtin Python function
    sorted.
    Parameters:
        sorting_algo: algorithm function to evaluate
        cmp: Optional, default None. If given, this comparator will be used in both the algorithms.
    Returns:
        None if the algorithm works correctly, raise an error otherwise."""
    for _ in range(100):
        l1 = [randint(0, 10000) for _ in range(randint(0, 10))]

        if cmp is not None:
            l1_sorted = sorting_algo(l1, cmp=cmp)
            l1_comp = sorted(l1, key=cmp_to_key(cmp))

        else:
            l1_sorted = sorting_algo(l1)
            l1_comp = sorted(l1)

        try:
            assert l1_sorted == l1_comp, "FAIL!"
        except AssertionError:
            print("error")

    print("Test passed!")


def activity_selection(L: list):
    """Activity selection problem; this  function solves this problem in nlog n time.
    Parameters:
        L: list in the form [(start1, end1), (start2, end2), (start3, end3)]
    Returns:
        List containing the choice of activities that optimize the selection problem."""
    L_sorted = sorted(L, key=lambda x: x[1])
    # The first element of the sorted list will be the first activity to be executed for sure
    sol = [L_sorted[0]]
    # Check all the other activities
    for item in L_sorted[1:]:
        # If the initial time of the considered activity is bigger or equal to the previous activity ending time
        # (it's basically checking for non-overlapping) append the activity to the solution
        if item[0] >= sol[-1][1]:
            sol.append(item)

    return sol


def fractional_knapsack(L: list, W: int = 50):
    """Fractional Knapsack problem.
    Parameters:
        L: list of pairs (value, weight)
        W: Optional int (default 50)
    Returns:
        The maximum possible obtainable value by selecting items."""
    sol = 0
    act_w = 0
    L_sorted = sorted(L, key=lambda x: x[0] / x[1], reverse=True)
    for item in L_sorted:
        if act_w + item[1] == W:
            return sol + item[0]

        elif act_w + item[1] > W:
            return sol + item[0] * (W - act_w) / item[1]

        else:
            sol += item[0]
            act_w += item[1]


def pareto_frontier(s: list):
    s_sorted = sorted(s, reverse=True)
    t = s_sorted[0]
    p = [t]
    for c in s_sorted[1:]:
        if c[0] <= t[0] and c[1] <= t[1]:
            pass

        else:
            p.append(c)
            t = c

    return p


class StaticSortedMap:
    def __init__(self, a: list):  # assume A is already sorted
        self.sorted_map = a[:]  # copy input array

    def min(self):
        return self.sorted_map[0]

    def max(self):
        return self.sorted_map[-1]

    def search(self, key: int):
        def __binary_search(p: int, e: int, key):
            if p <= e:
                q = ceil((p + e) / 2)
                if self.sorted_map[q] == key:

                    return True, q

                elif self.sorted_map[q] < key:
                    return __binary_search(q+1, e, key)

                else:
                    return __binary_search(p, q-1, key)

            else:
                if key < self.sorted_map[p]:
                    return False, p

                else:
                    return False, p+1
        return __binary_search(0, len(self.sorted_map)-1, key)

    def predecessor(self, key):
        exist, position = self.search(key)
        if exist:
            if position != 0:
                return position - 1, self.sorted_map[position - 1]

            else:
                return None

        else:
            raise KeyError(f"Key {key} doesn't exist!")

    def successor(self, key):
        exist, position = self.search(key)
        if exist:
            if position != len(self.sorted_map) - 1:
                return position + 1, self.sorted_map[position + 1]

            else:
                return None

        else:
            raise KeyError(f"Key {key} doesn't exist!")


class BinarySearchTree:
    class __Node:
        def __init__(self, val, left=None, right=None):
            """This is a Node class that is internal to the BinarySearchTree class"""
            self.__val = val
            self.__left = left
            self.__right = right

        @property
        def get_val(self):
            return self.__val

        def set_val(self, new_val):
            self.__val = new_val

        @property
        def get_left(self):
            return self.__left

        @property
        def get_right(self):
            return self.__right

        def set_left(self, new_left):
            self.__left = new_left

        def set_right(self, new_right):
            self.__right = new_right

        def __iter__(self):
            """This method deserves a little explanation. It does an inorder traversal of the nodes of the tree
            yielding all the values. In this way, we get the values in ascending order."""
            if self.__left is not None:
                for elem in self.__left:
                    yield elem
            yield self.__val
            if self.__right is not None:
                for elem in self.__right:
                    yield elem

    def __init__(self):
        self.root = None

    def insert(self, value):
        def __insert(root, val):
            if root is None:
                # A new node is returned if the tree is empty
                return BinarySearchTree.__Node(val)

            if val < root.get_val:
                root.set_left(__insert(root.get_left, val))

            else:
                root.set_right(__insert(root.get_right, val))

            return root

        self.root = __insert(self.root, value)

    def search(self, value):
        def __search(root, val):
            if root is None:
                # If the tree is empty - also alter the leaves - False is returned
                return False

            # If the root is not empty, check all the possible cases (<, =, >)
            if val == root.get_val:
                return True

            elif val > root.get_val:
                return __search(root.get_right, val)

            elif val < root.get_val:
                return __search(root.get_left, val)

        return __search(self.root, value)

    def delete(self, value):
        def __delete(root, val):
            if root is None:
                return root

            # Recursively travel the tree to find the node with the value to be deleted
            if root.get_val > val:
                # Transverse left if the value to be deleted is smaller than the value of the current node
                root.set_left(__delete(root.get_left, val))
                return root

            elif root.get_val < val:
                # Transverse right if the value to be deleted is bigger than the value of the current node
                root.set_right(__delete(root.get_right, val))
                return root

            # Once the node is found we first check if it has no children and reconnect the parent of the node
            # with the non-empty child
            if root.get_left is None:
                return root.get_right

            elif root.get_right is None:
                return root.get_left

            # If both children are non-empty we look for the successor (aka the smallest node in the right subtree
            # of the node to be deleted) and replace teh value of the node to be deleted with the value of the successor
            else:
                successor_parent = root
                # The following while loop finds the successor
                successor = root.get_right

                while successor.get_left is not None:
                    successor_parent = successor
                    successor = successor.get_left

                if successor_parent != root:
                    successor_parent.set_left(successor.get_right)

                else:
                    successor_parent.set_right(successor.get_right)

                root.set_val(successor.get_val)

                # The computed root is returned and will overwrite the old one
                return root

        self.root = __delete(self.root, value)

    def inorder(self):
        """Utility function extracted from G4G, it allows to do inorder traversal of BST."""
        def __inorder(root):
            if root is not None:
                __inorder(root.get_left)
                print(root.get_val, end=" ")
                __inorder(root.get_right)

        __inorder(self.root)
        print("\n")

    @property
    def min(self):
        """Return the minimum element."""
        def __min(root):
            while root.get_left is not None:
                root = root.get_left

            return root.get_val

        return __min(self.root)

    @property
    def max(self):
        """Return the maximum element."""
        def __max(root):
            while root.get_right is not None:
                root = root.get_right

            return root.get_val

        return __max(self.root)


def get_random_array(n, b=50):
    return [randint(0, b) for _ in range(n)]


if __name__ == "__main__":
    pass
