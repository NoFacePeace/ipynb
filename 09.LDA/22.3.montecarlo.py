# https://www.hitoy.org/monte-carlo-method-and-random-number.html
#http://www.jianshu.com/p/3d30070932a8

import random
import math
def getpi(count):
    i = 0
    k = 0.0
    while i < count:
        x = random.random()
        y = random.random()
        if math.sqrt(x*x + y*y) < 1:
            k+=1
        i+=1
    return k/(count*0.25)

def getarea(count):
    i = 0
    k = 0.0
    while i < count:
        x = random.random()
        y = random.random()
        if y < x*x:
            k+=1
        i+=1
    return k/count

N=1000000
pi=getpi(N)
s=getarea(N)

print("pi=",pi)
print("s=",s)