
# 그냥 짝수 경우 홀수 경우 나눈 풀이
"""
짝수이면 total//2 으로 한 수와 total//2+1한 수를 중심으로 
num//2개 만큼(중앙수 포함) 앞 뒤로 나열.
홀수의 경우는 total//2을 중심으로 (중앙 수 제외하고) 
num//2 만큼 앞뒤로 나열 
"""
def solution(num, total):
    if num%2==0: # 짝수이면
        mid1 = total//num
        mid2 = mid1+1
        len = num//2
        answer = [i for i in range(mid1-len+1, mid2+len)]
    else:  # 홀수이면
        mid = total//num
        len = num//2
        answer = [i for i in range(mid-len, mid+len+1)]
    return answer


# 수열의 성질을 이용한 풀이
def solution(num, total):
    start = int(total/num - (num-1)/2)
    answer = list(range(start, start+num))
    return answer