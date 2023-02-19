# https://school.programmers.co.kr/learn/courses/30/lessons/140108?language=python3

def solution(s):
    count = 1
    start = 0
    for idx in range(2,len(s)):
        t = s[start:idx]
        if t.count(t[0]) == len(t)/2:
            count +=1
            start = idx
        
    return count



# 다른 풀이
def solution(s):
    answer = 0
    sav1=0
    sav2=0
    for i in s:
        if sav1==sav2:
            answer+=1
            a=i
        if i==a:
            sav1+=1
        else:
            sav2+=1
    return answer