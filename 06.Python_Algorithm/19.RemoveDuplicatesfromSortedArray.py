# https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        for _ in range(len(nums)):
            n = nums.pop(0)
            if n in nums:
                pass
            else:
                nums.append(n)
        return len(nums)

# 어이없게 쉬운 방법  
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        lst = sorted(list(set(nums)))  # 그냥 중복값 뽑아서 정렬하고
        for i, j in enumerate(lst):
            nums[i] = j  # 앞에다 넣은것
        
        return len(lst)
    