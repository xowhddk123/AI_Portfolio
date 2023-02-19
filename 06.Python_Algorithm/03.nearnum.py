def solution(array, n):
    lst = []
    array = sorted(list(array))
    for i in array:
        lst.append(abs(n-i))
        idx = lst.index(min(lst))
        answer = array[idx]
    return answer