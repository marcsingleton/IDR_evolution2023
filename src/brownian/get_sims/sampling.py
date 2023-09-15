"""Sampling parameters for get_sims.py"""

from numpy import linspace

num_samples = 100
seed = 1

sigma2_min, sigma2_max = -3, 3
sigma2_delta = 0.5
sigma2_num = int((sigma2_max - sigma2_min) / sigma2_delta) + 1
sigma2_range = [10 ** x for x in linspace(sigma2_min, sigma2_max, sigma2_num)]

alpha_min, alpha_max = -2, 2
alpha_delta = 0.5
alpha_num = int((alpha_max - alpha_min) / alpha_delta) + 1
alpha_range = [10 ** x for x in linspace(alpha_min, alpha_max, alpha_num)]
