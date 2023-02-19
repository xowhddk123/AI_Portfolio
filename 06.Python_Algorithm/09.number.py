# https://www.acmicpc.net/problem/10093

# x, y = map(int, input().split(" "))
if x == y:
    print(0)
    print()
elif y > x:
    print(y - x - 1)
    for i in range(x+1, y):
        print(i,end=" ")
else:
    print(x - y - 1)
    for i in range(y+1, x):
        print(i,end=" ")
        
# list로 풀기
lst = sorted(list(map(int, input().split(" "))))
if lst[0] == lst[1]:
    print(0)
else :
    print(lst[1] - lst[0] - 1)
    for i in range(lst[0]+1, lst[1]):
        print(i,end=" ")


# tuple로 풀기
x, y = sorted(tuple(map(int, input().split(" "))))
if x == y:
    print(0)
else :
    print(y - x - 1)
    for i in range(x+1, y):
        print(i,end=" ")

# list없이도 sorted가능함 + join활용
x, y = sorted(map(int, input().split()))
if x == y:
    print(0)
else :
    print(y - x - 1)
    print(" ".join(map(str, range(x+1, y))))
    

# if문 없이 풀어보기 
x, y = sorted(map(int, input().split()))
lst = [i for i in range(x+1, y)]
print(len(lst))
print(" ".join(map(str, range(x+1, y))))