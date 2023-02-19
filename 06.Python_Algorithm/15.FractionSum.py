# https://school.programmers.co.kr/learn/courses/30/lessons/120808
def solution(denum1, num1, denum2, num2):
    denum = denum1*num2 +denum2*num1
    num = num1*num2
    if denum < num:
        s = denum
    elif denum >= num:
        s = num
    else:
        answer = [1,1]
    for i in range(2, s):
        while (denum%i ==0)&(num%i==0):
            denum = denum/i
            num = num/i
    answer = [denum, num]
    return answer