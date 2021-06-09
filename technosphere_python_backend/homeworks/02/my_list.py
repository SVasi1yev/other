import unittest


class MyListTest(unittest.TestCase):
    def test_add(self):
        a = MyList([1, 2])
        b = MyList([10, 20, 30])
        c = a + b
        self.assertEqual(c[0], 11)
        self.assertEqual(c[1], 22)
        self.assertEqual(c[2], 30)

    def test_sub(self):
        a = MyList([11, 22, 30])
        b = MyList([1, 2])
        c = a - b
        self.assertEqual(c[0], 10)
        self.assertEqual(c[1], 20)
        self.assertEqual(c[2], 30)
        a = MyList([11, 22])
        b = MyList([1, 2, 30])
        c = a - b
        self.assertEqual(c[0], 10)
        self.assertEqual(c[1], 20)
        self.assertEqual(c[2], -30)

    def test_cmp(self):
        a = MyList([1, 2, 3])
        b = MyList([2, 2, 2])
        self.assertEqual(a, b)
        self.assertLessEqual(a, b)
        self.assertGreaterEqual(a, b)
        a = MyList([1, 2, 3])
        b = MyList([2, 2, 100])
        self.assertLess(a, b)
        self.assertLessEqual(a, b)
        a = MyList([1, 2, 100])
        b = MyList([2, 2, 2])
        self.assertGreater(a, b)
        self.assertGreaterEqual(a, b)


class MyList(list):
    def __add__(self, other):
        res = MyList()
        for i in range(min(len(self), len(other))):
            res.append(self[i] + other[i])
        if len(self) > len(other):
            for i in range(len(other), len(self)):
                res.append(self[i])
        else:
            for i in range(len(self), len(other)):
                res.append(other[i])
        return res

    def __sub__(self, other):
        res = MyList()
        for i in range(min(len(self), len(other))):
            res.append(self[i] - other[i])
        if len(self) > len(other):
            for i in range(len(other), len(self)):
                res.append(self[i])
        else:
            for i in range(len(self), len(other)):
                res.append(-other[i])
        return res

    def _count_sums(self, other):
        self_sum = 0
        for e in self:
            self_sum += e
        other_sum = 0
        for e in other:
            other_sum += e
        return self_sum, other_sum

    def __cmp__(self, other):
        self_sum, other_sum = self._count_sums(other)
        return self_sum - other_sum

    def __eq__(self, other):
        self_sum, other_sum = self._count_sums(other)
        return self_sum == other_sum

    def __ne__(self, other):
        self_sum, other_sum = self._count_sums(other)
        return self_sum != other_sum

    def __lt__(self, other):
        self_sum, other_sum = self._count_sums(other)
        return self_sum < other_sum

    def __gt__(self, other):
        self_sum, other_sum = self._count_sums(other)
        return self_sum > other_sum

    def __le__(self, other):
        self_sum, other_sum = self._count_sums(other)
        return self_sum <= other_sum

    def __ge__(self, other):
        self_sum, other_sum = self._count_sums(other)
        return self_sum >= other_sum
