# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 15:19:05 2022

@author: leul
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
n = 100
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc=0, scale = 1, size = n)
def compute_rss(y_estimate, y):
  return sum(np.power(y-y_estimate, 2))
def estimate_y(x, b_0, b_1):
  return b_0 + b_1 * x
rss = compute_rss(estimate_y(x, beta_0, beta_1), y)
rss = []
slopes = np.arange(-10, 15, 0.001)
for slope in slopes:
    rss.append(np.sum((y-beta_0- slope*x)**2))



import statsmodels.api as sm
mod = sm.OLS(y, x)
est = mod.fit()
print(est.summary())
x = sm.add_constant(x)
mod = sm.OLS(y, x)
est = mod.fit()
print(est.summary())



from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
np





































