# https://school.programmers.co.kr/learn/courses/30/lessons/120812  

def solution(array):
    dict = {num:array.count(num) for num in array}
    dict = sorted(dict.items(), key = lambda x:x[1])
    try:
        if dict[-1][1] == dict[-2][1]:
            answer = -1
        else:
            answer = dict[-1][0]
    except:
        answer = dict[-1][0]
    print(answer)
    return answer

# 미친풀이
def solution(array):
    while len(array) != 0:
        for i, a in enumerate(set(array)): # 마지막 까지 살아남는 녀석이 최빈값이다.
            array.remove(a)
        if i == 0: return a
    return -1