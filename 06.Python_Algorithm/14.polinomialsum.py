# https://school.programmers.co.kr/learn/courses/30/lessons/120863
# 1회 시도
# def solution(polynomial):
#     lst = polynomial.split(" + ")
#     X = 0
#     c = 0
#     for i in lst:
#         if "x" in i:
#             if i[:-1] == "":
#                 X +=1
#             else:
#                 X += int(i[:-1])
#         else:
#             c += int(i)
#     if c == 0 :
#         if X > 1:
#             answer = f"{X}x"
#         else:
#             answer = f"x"
#     else : 
#         if X > 1:
#             answer = f"{X}x + {c}"
#     return answer
# 경우의 수가 너무 많이 나눠야 해서 안좋은것 같음

