# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:27:38 2025

@author: ПользовательHP
"""

import time
from random import randint
import math as M
import liba

a = []
sm=0;

for i in range (5):
    a.append(randint(1, 10))
    if(a[i]%2==0):
        sm+=a[i]
print(a,sm)