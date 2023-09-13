"""Sampling parameters for get_sims.py"""

num_samples = 100
seed = 1

sigma2_min, sigma2_max = 0.1, 1
sigma2_delta = 0.1
sigma2_num = int((sigma2_max - sigma2_min) / sigma2_delta) + 1

alpha_min, alpha_max = 0.1, 1
alpha_delta = 0.1
alpha_num = int((alpha_max - alpha_min) / alpha_delta) + 1
