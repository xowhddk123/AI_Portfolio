# https://school.programmers.co.kr/learn/courses/30/lessons/120907?language=python3
def solution(quiz):
    result = []
    for i in quiz:
        q = [int(j) for idx,j in enumerate(i.split()) if idx%2==0]
        operator = [k for idx,k in enumerate(i.split()) if idx%2!=0]
        if operator[0] == "+":
            answer = q[0]+q[1]                
        else:
            answer = q[0]-q[1]
        if answer == q[2]:
            result.append('O')
        else:
            result.append('X')        
    return result

# 두 번째 풀이
def solution(quiz):
    result = []
    for i in quiz:
        q = i.split()
        if q[1] == '+':
            answer = int(q[0])+int(q[2])
        else:
            answer = int(q[0])-int(q[2])
        if answer == int(q[4]):
            result.append('O')
        else:
            result.append('X')
        
    return result

# 세 번째 풀이
# 현업에서 eval 쓰면 혼납니다.
def solution(quiz):
    result = []
    quiz = [i.replace('=', '==') for i in quiz]
    for i in quiz:
        if eval(i):
            result.append('O')
        else:
            result.append('X')
    return result

# 세 번째 풀이, replace를 이용한 풀이
def solution(quiz):
    result = []
    left = []
    right = []
    for i in quiz:
        i = i.replace(' + ', ' ').replace(' - ', ' -').replace('--', '+')
        L,R = i.split('=')
        left.append(sum(list(map(int, L.split()))))
        right.append(int(R))
    for l,r in zip(left,right):
        if l == r:
            result.append('O')
        else:
            result.append('X')
    return result