# https://school.programmers.co.kr/learn/courses/30/lessons/120884?language=python3

def solution(chicken):
    answer = 0
    while chicken >=10:
        chicken, n = divmod(chicken,10)
        answer += chicken
        chicken +=n
    return answer


# 정신나간 다른 사람의 풀이
    """
    모든 치킨은 0.1의 쿠폰을 발행한다.
    그러므로 치킨에 계속해서 0.1을 곱해주면 그 가치를 찾을 수 있다.
    사실 반복적으로 해야하지만 이 풀이는 그냥 적당히 많이 넣은것 같다.
    
    """

def solution(chicken):
    return int(chicken*0.11111111111)

# 정신나간 풀이 2
    """
    최초의 10마리를 제외하고는 서비스 치킨은 9마리에 1나씩 받을 수 있다.
    왜냐하면 서비스 치킨에도 쿠폰을 발행하기 때문이다.
    그러므로 처음 한 마리를 제외하고 9로 나누어 주면 
    서비스로 받을 수 있는 총 치킨의 수가 구해진다.
    """
def solution(chicken):
    answer = (max(chicken,1)-1)//9
    return answer

# 제귀함수 풀이(근혜님이 알려주신 풀이)
    """
    단순하게 10마리 먹으러 쿠폰 하나를 주는 함수를 만들고
    그 함수를 재귀함수 형태로 돌리면 해결
    단, 메모리 관점에서 좋지는 않음
    """
def solution(chicken):
    if chicken >=10:
        return chicken //10 + solution(chicken//10 + chicken%10)
    else:
        return 0 
    return answer
