# https://school.programmers.co.kr/learn/courses/30/lessons/81301?language=python3

def solution(s):
    dict = {"zero":"0", "one":"1", "two":"2", "three":"3", 
            "four":"4", "five":"5", "six": "6", "seven":"7", 
            "eight":"8", "nine": "9"}
    for key, val in dict.items():
        s = s.replace(key, val)
    return int(s)

# list로 풀어보기
def solution(s):
    lst = ["zero", "one", "two", "three", "four", 
         "five", "six", "seven", "eight", "nine"]
    for num, str in enumerate(lst):
        s = s.replace(str,f"{num}")
    return int(s)

# 메모리나 시간은 비슷하다