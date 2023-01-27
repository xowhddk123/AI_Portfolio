# 문제 링크 : https://school.programmers.co.kr/learn/courses/30/lessons/12937

def solution(num):
    if num%2 == 0:
        answer = "Even"
    else : 
        answer = "Odd"
    return answer


# 인상깊은 풀이
def evenOrOdd(num):
    return ["Even", "Odd"][num & 1]
