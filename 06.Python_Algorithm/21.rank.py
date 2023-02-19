# https://school.programmers.co.kr/learn/courses/30/lessons/120882?language=python3
def solution(score):
    m_dict = {idx:sum(s)/2 for idx,s in enumerate(score)}
    m = sorted(m_dict.items(), key = lambda x:x[1], reverse = True)
    idx = 0
    count = 0
    i0 = -1
    for i in m:
        if i[1] != i0:
            idx +=1 + count
            count = 0
            score[i[0]] = idx
        else :
            score[i[0]] = idx
            count +=1
        i0 = i[1]
    return score

# idex 성질을 이용한 풀이
def solution(score):
    # sum_dict = {idx:sum(i) for idx, i in enumerate(score)}
    sum_lst = [sum(i) for i in score]
    rank_lst = sorted(sum_lst, reverse = True)
    rank = [rank_lst.index(i)+1 for i in sum_lst]
    return rank

# rank의 성질을 이용한 풀이
# rank 문제는 기본적으로 나보다 높은 점수가 몇개인가를 찾는 문제
def solution(score):
    avg_lst = [(i+j)/2 for i, j in score]
    rank_lst = []
    for i, v in enumerate(avg_lst):
        # v보다 큰 수의 개수를 세서 등수로 적어준다.
        greater = 1
        for v2 in avg_lst:
            if v2 > v:
                greater += 1
        rank_lst.append(greater)
    return rank_lst