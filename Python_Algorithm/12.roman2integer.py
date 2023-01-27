# https://leetcode.com/problems/roman-to-integer/description/

class Solution:
    def romanToInt(self, s: str) -> int:
        num_dict = {"CM":"900,","M":"1000,", "CD": "400,","D":"500,","XC":"90,",
        "C":"100,", "XL":"40,","L":"50,",
        "IX":"9,","X":"10,","IV":"4,","V":"5,", "III":"3,", "I":"1,"}
        for rom, num in num_dict.items():
            s = s.replace(rom, num)
        answer =  sum(list(map(int,s.split(",")[:-1])))
        return answer
    
# 다른 풀이
class Solution:
    def romanToInt(self, s: str) -> int:
        dic = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000,
        'IX':9, 'IV':4, 'XL':40, 'XC':90, 'CD':400, 'CM':900}
        int_ = 0
        pass_flag = False
        for i,v in enumerate(s):
            if pass_flag == True:
                pass_flag = False
                continue
            if s[i:i+2] in dic:
                int_ += dic[s[i:i+2]]
                print(s[i:i+2], int_)
                pass_flag = True
            else:
                int_ += dic[v]
        return int_

########## Solution 2 ###########

class Solution:
    def romanToInt(self, s: str) -> int:
        dic1 = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000,}
        dic2 = {'IX':9, 'IV':4, 'XL':40, 'XC':90, 'CD':400, 'CM':900}
        
        int_ = 0
        while s != '':
            if s[:2] in dic2:
                int_ += dic2[s[:2]]
                s = s[2:]
            elif s[:1] in dic1:
                int_ += dic1[s[:1]]
                s = s[1:]
            
        return int_