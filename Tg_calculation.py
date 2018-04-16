""" File for calculating the glass transition temperature """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, argparse
from math import *
import numpy as np
import time
import copy, os
import json
import random
import csv
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 2e5
# Construct the glass line
p0 = [0.05, -3.8661988347]
p1 = [0.20015, -3.86504891259]

m_g = (p1[1] - p0[1])/(p1[0] - p0[0])

b_g = p0[1] - m_g*p0[0]



# Construct the liquid line
p0 = [0.50045, -3.85198190248]
p1 = [0.800875, -3.82159465103]

m_l = (p1[1] - p0[1])/(p1[0] - p0[0])

b_l = p0[1] - m_l*p0[0]



# Calculate Tg
Tg5 = (b_l - b_g)/(m_g - m_l)

print(Tg5, "2e5")

######################################################
######################################################
######################################################

#2e6
# Construct the glass line
p0 = [0.05, -3.88828792715]
p1 = [0.20015, -3.88724694961]

m_g = (p1[1] - p0[1])/(p1[0] - p0[0])

b_g = p0[1] - m_g*p0[0]



# Construct the liquid line
p0 = [0.50045, -3.86878913843]
p1 = [0.800875, -3.82756762577]

m_l = (p1[1] - p0[1])/(p1[0] - p0[0])

b_l = p0[1] - m_l*p0[0]



# Calculate Tg
Tg6 = (b_l - b_g)/(m_g - m_l)

print(Tg6, "2e6")

######################################################
######################################################
######################################################

#2e7
# Construct the glass line
p0 = [0.05, -3.90291542711]
p1 = [0.206, -3.90204929392]

m_g = (p1[1] - p0[1])/(p1[0] - p0[0])

b_g = p0[1] - m_g*p0[0]



# Construct the liquid line
p0 = [0.50825, -3.87450783173]
p1 = [0.800875, -3.82763656652]

m_l = (p1[1] - p0[1])/(p1[0] - p0[0])

b_l = p0[1] - m_l*p0[0]



# Calculate Tg
Tg7 = (b_l - b_g)/(m_g - m_l)

print(Tg7, "2e7")

######################################################
######################################################
######################################################

#2e7
# Construct the glass line
p0 = [0.05, -3.9143312115]
p1 = [0.20015, -3.91369617365]

m_g = (p1[1] - p0[1])/(p1[0] - p0[0])

b_g = p0[1] - m_g*p0[0]



# Construct the liquid line
p0 = [0.50045, -3.87741246518]
p1 = [0.80075, -3.82613288217]

m_l = (p1[1] - p0[1])/(p1[0] - p0[0])

b_l = p0[1] - m_l*p0[0]



# Calculate Tg
Tg8 = (b_l - b_g)/(m_g - m_l)

print(Tg8, "2e8")

print((Tg5 - Tg8)/2 + Tg8)





