# https://leetcode.com/problems/palindrome-number/

class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        elif x == 0:
            return True
        else:
            lst = []
            m = x
            while m != 0:
                m, n = divmod(m, 10)
                lst.append(n)
            lst = list(map(str, lst))
            answer = "".join(lst)
            if int(answer) == x:
                return True
            else:
                return False
        
