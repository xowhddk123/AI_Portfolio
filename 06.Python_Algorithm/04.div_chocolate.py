# https://www.acmicpc.net/problem/2163

N,M = map(int, input().split()) 
def d(N, M):
    return N*M-1

print(d(N, M))