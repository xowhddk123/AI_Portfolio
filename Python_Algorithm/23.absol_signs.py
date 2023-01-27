# https://school.programmers.co.kr/learn/courses/30/lessons/76501?language=python3

def solution(absolutes, signs):
    signs = ['+' if i else '-' for i in signs]
    lst = [int(j+f'{i}') for i, j in zip(absolutes, signs)]
    answer = sum(lst)
    return answer

# 숏코딩
def solution(absolutes, signs):
    return sum([int(f'+{i}') if j else int(f'-{i}') for i, j in zip(absolutes, signs)])

