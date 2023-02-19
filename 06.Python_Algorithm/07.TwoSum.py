# # https://leetcode.com/problems/two-sum/


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        diff = [target - i for i in nums]
        for i,j in enumerate(diff):
            if (j in nums[i+1:]):
                answer = [i,nums.index(j,i+1)]
                break
        return answer
    
    
    
# 놀라운 정답
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
# 		myHash = {}
# 	    for index, value in enumerate(nums):
# 	        if target - value in myHash:
#                 return [myHash[target-value], index]
#             myHash[value] = index

 