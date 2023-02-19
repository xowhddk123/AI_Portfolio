# https://school.programmers.co.kr/learn/courses/30/lessons/132267


def solution(a, b, n):
    answer = 0
    while n//a != 0:
        m, n = divmod(n, a)
        n += b*m
        answer +=b*m
    return answer