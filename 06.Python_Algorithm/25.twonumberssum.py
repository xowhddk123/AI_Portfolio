# https://school.programmers.co.kr/learn/courses/30/lessons/68644

def solution(numbers):
    answer = []
    numbers2 = numbers.copy()
    for i in numbers:
        idx = numbers2.index(i)
        numbers2.remove(i)
        for j in numbers2:
            answer.append(i+j)
        numbers2.insert(idx, i) 
    answer = sorted(list(set(answer)))
    return answer

def solution(numbers):
    
    return sorted(set([j+k for i,j in enumerate(numbers) for k in numbers[i+1:]]))