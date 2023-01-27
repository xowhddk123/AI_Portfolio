# https://www.acmicpc.net/problem/1225

# 1회차 풀이
# A,B = map(int,input().split())
# m1 = A
# n1_lst = []
# while m1 != 0:
#     m1, n1 = divmod(m1, 10) 
#     n1_lst.append(n1)

# m2 = B
# n2_lst = []
# while m2 != 0:
#     m2, n2 = divmod(m2, 10) 
#     n2_lst.append(n2)

# answer = 0
# for i in n1_lst:
#     for j in n2_lst:
#         answer += i*j
# print(answer)
# 시간초과 나옴


# 2회차 풀이
# A,B = input().split()

# A = [int(i) for i in A]
# B = [int(i) for i in B]
# answer = 0
# for i in A:
#     for j in B:
#         answer += i*j
        
# print(answer)

# 3회차 풀이
A,B = input().split()

A = sum([int(i) for i in A if i != "0"]) # 더 짧은 리스트
B = sum([int(i) for i in B if i != "0"])  # 더 긴 리스트
answer = A*B
print(answer)

# 근혜님 풀이
# A, B = input().split()
# Alist = list(map(int, list(A)))  # Alist = ['1', '2', '1']
# Blist = list(map(int, list(B)))  

# output = 0
# for i in Alist:
#     result = [i*j for j in Blist]
#     output += sum(result) 
# print(output)


# 처음에는 list(str) 메서드를 안쓰고 divmod를 활용해서 풀려고 했다.
# 그런데 시간 초과가 떠서 빠른 list(str)으로 돌아갔다.
# 그래도 시간초과가 떴다.
# 시간을 아껴주는 list comprehension을 썼다.
# 그러자 메모리 초과가 났다.
# 그래서 list comprehension을 작은 단위(for문 하나)로 수행하고 
# 결과 합을 중간중간 업데이트해주는 방식으로 해서 통과가 떴다.