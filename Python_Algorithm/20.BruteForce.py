# https://www.acmicpc.net/problem/1233

s1,s2,s3 = map(int,input().split())

lst1 = [i for i in range(1,s1+1)]
lst2 = [i for i in range(1,s2+1)]
lst3 = [i for i in range(1,s3+1)]

lst = [i+j+k for i in lst1 for j in lst2 for k in lst3]
dict = {i:lst.count(i) for i in set(lst)}
dict = sorted(dict.items(), key=lambda x:x[1], reverse=True)
# print(dict[0][0])
print(dict)



# a,b,c=map(int,input().split())
# x,t=a+b+c-1,max(a,b,c)*2
# if t>x:
#     x+=x-t
# print(x//2+2)