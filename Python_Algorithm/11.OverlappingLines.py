# https://school.programmers.co.kr/learn/courses/30/lessons/120876

def solution(lines):
    line1 = [i for i in range(lines[0][0], lines[0][1])]
    line2 = [i for i in range(lines[1][0], lines[1][1])]
    line3 = [i for i in range(lines[2][0], lines[2][1])]
    a= set(line1)&set(line2)
    b = set(line2)&set(line3)
    c= set(line3)&set(line1)
    return len(a|b|c)