"""Simulate data under BM and OU models and calculate estimated parameters and likelihoods."""

import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import skbio
import src.phylo as phylo

num_samples = 10
models_BM = [1]
models_OU = [(1, 0.1)]

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tree_reference = tree_template.shear({tip.name for tip in tree_template.tips() if tip.name != 'sleb'})
tips_reference, cov_reference = phylo.get_brownian_covariance(tree_reference)
inv_reference = np.linalg.inv(cov_reference)

if not os.path.exists('out/'):
    os.mkdir('out/')

# BM simulations
rows = []
for sigma2 in models_BM:
    for sample_id in range(num_samples):
        tree = tree_reference.copy()
        tips = list(tree.tips())
        cov = cov_reference * sigma2
        values = stats.multivariate_normal.rvs(cov=cov)
        for tip, value in zip(tips, values):
            tip.value = value

        mu_hat_BM, sigma2_hat_BM = phylo.get_brownian_mles(cov=cov_reference, inv=inv_reference, values=values)
        loglikelihood_hat_BM = phylo.get_brownian_loglikelihood(mu_hat_BM, sigma2_hat_BM,
                                                                cov=cov_reference, inv=inv_reference, values=values)

        mu_hat_OU, sigma2_hat_OU, alpha_hat_OU = phylo.get_OU_mles(tips=tips, ts=cov_reference)
        loglikelihood_hat_OU = phylo.get_OU_loglikelihood(mu_hat_OU, sigma2_hat_OU, alpha_hat_OU,
                                                          tips=tips, ts=cov_reference)

        rows.append({'sigma2': sigma2, 'sample_id': sample_id,
                     'mu_hat_BM': mu_hat_BM, 'sigma2_hat_BM': sigma2_hat_BM,
                     'loglikelihood_hat_BM': loglikelihood_hat_BM,
                     'mu_hat_OU': mu_hat_OU, 'sigma2_hat_OU': sigma2_hat_OU, 'alpha_hat_OU': alpha_hat_OU,
                     'loglikelihood_hat_OU': loglikelihood_hat_OU})
df_BM = pd.DataFrame(rows)
df_BM.to_csv('out/models_BM.tsv', sep='\t', index=False)

# OU simulations
rows = []
for sigma2, alpha in models_OU:
    for sample_id in range(num_samples):
        tree = tree_reference.copy()
        tips = list(tree.tips())
        _, cov = phylo.get_OU_covariance(alpha, tips=tips, ts=cov_reference * sigma2)
        values = stats.multivariate_normal.rvs(cov=cov)
        for tip, value in zip(tips, values):
            tip.value = value

        mu_hat_BM, sigma2_hat_BM = phylo.get_brownian_mles(cov=cov_reference, inv=inv_reference, values=values)
        loglikelihood_hat_BM = phylo.get_brownian_loglikelihood(mu_hat_BM, sigma2_hat_BM,
                                                                cov=cov_reference, inv=inv_reference, values=values)

        mu_hat_OU, sigma2_hat_OU, alpha_hat_OU = phylo.get_OU_mles(tips=tips, ts=cov_reference)
        loglikelihood_hat_OU = phylo.get_OU_loglikelihood(mu_hat_OU, sigma2_hat_OU, alpha_hat_OU,
                                                          tips=tips, ts=cov_reference)

        rows.append({'sigma2': sigma2, 'alpha': alpha, 'sample_id': sample_id,
                     'mu_hat_BM': mu_hat_BM, 'sigma2_hat_BM': sigma2_hat_BM,
                     'loglikelihood_hat_BM': loglikelihood_hat_BM,
                     'mu_hat_OU': mu_hat_OU, 'sigma2_hat_OU': sigma2_hat_OU, 'alpha_hat_OU': alpha_hat_OU,
                     'loglikelihood_hat_OU': loglikelihood_hat_OU})
df_OU = pd.DataFrame(rows)
df_OU.to_csv('out/models_OU.tsv', sep='\t', index=False)
