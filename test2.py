#for문 내에 선언된 variable 관련 실험

for v1 in range(5):
    v1 = 3
print(v1)

#실험 결과 v1은 그대로 존재함을 알 수 있다.
'''
def func():
    v2 = 1
    return v2
print (v2)
# 메소드 내에 선언된 variable은 당연히 접근할 수 없다.
'''