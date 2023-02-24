# 프로그래머스 알고리즘 1단계
# https://school.programmers.co.kr/learn/courses/30/lessons/42840?language=python3

def solution(answers):
    l = len(answers)
    # patten
    p1 = [1,2,3,4,5]
    p2 = [2, 1, 2, 3, 2, 4, 2, 5]
    p3 = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]
    
    l1 = len(p1)
    l2 = len(p2)
    l3 = len(p3)
    
    #answer list
    first = [answers[i] - p1[i%l1] for i in range(l)].count(0)
    second = [answers[i] - p2[i%l2] for i in range(l)].count(0)
    third = [answers[i] - p3[i%l3] for i in range(l)].count(0)

    
    lst = [first, second, third]
    m = max(lst)
    answer = [i+1 for i,v in enumerate(lst) if v==m]

    return answer



# 다른 풀이
def solution(answers):
    # 수포자들의 찍기 패턴
    pattern1 = [1, 2, 3, 4, 5]
    pattern2 = [2, 1, 2, 3, 2, 4, 2, 5]
    pattern3 = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]
    
    # 수포자들의 정답 개수를 저장할 리스트
    scores = [0, 0, 0]
    
    # 정답 개수를 계산하여 scores 리스트에 저장
    for i in range(len(answers)):
        if answers[i] == pattern1[i % len(pattern1)]:
            scores[0] += 1
        if answers[i] == pattern2[i % len(pattern2)]:
            scores[1] += 1
        if answers[i] == pattern3[i % len(pattern3)]:
            scores[2] += 1
    
    # 가장 많은 문제를 맞춘 수포자의 인덱스를 저장할 리스트
    winners = []
    
    # 가장 높은 점수를 찾음
    max_score = max(scores)
    
    # 가장 높은 점수를 받은 수포자의 인덱스를 winners 리스트에 추가
    for i in range(len(scores)):
        if scores[i] == max_score:
            winners.append(i+1)
    
    return winners
