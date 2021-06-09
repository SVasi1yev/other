from collections import deque
import cProfile
import pstats
import io
import sys


def climb_stairs_slow(n):
    res = 0

    queue = deque()
    queue.append(n)
    while len(queue) > 0:
        cur = queue.popleft()
        if cur == 0:
            res += 1
        elif cur > 0:
            queue.append(cur - 1)
            queue.append(cur - 2)

    return res


def climb_stairs_fast(n):
    t1 = 0
    t2 = 1
    for _ in range(n):
        t1, t2 = t2, t1 + t2
    return t2


if __name__ == '__main__':
    pr_slow = cProfile.Profile()
    pr_slow.enable()
    res = climb_stairs_slow(int(sys.argv[1]))
    pr_slow.disable()
    print(f'>> SLOW FUNCTION RESULT: {res}\n')
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr_slow, stream=s).sort_stats(sortby)
    ps.print_stats()
    if len(sys.argv) > 2:
        print(s.getvalue())

    pr_fast = cProfile.Profile()
    pr_fast.enable()
    res = climb_stairs_fast(int(sys.argv[1]))
    pr_fast.disable()
    print(f'>> FAST FUNCTION RESULT: {res}\n')
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr_fast, stream=s).sort_stats(sortby)
    ps.print_stats()
    if len(sys.argv) > 2:
        print(s.getvalue())
