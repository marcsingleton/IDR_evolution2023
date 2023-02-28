"""Calculate ancestral sequence reconstruction at root for amino acid process."""

import json
import os

import numpy as np
import skbio
from scipy.special import gammainc
from scipy.stats import gamma
from src.evosim.asr import get_conditional
from src.utils import read_fasta, read_paml


def load_model(path):
    matrix, freqs = read_paml(path, norm=True)
    matrix = freqs * matrix
    np.fill_diagonal(matrix, -matrix.sum(axis=1))
    return matrix, freqs


alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
sym2idx = {sym: idx for idx, sym in enumerate(alphabet)}

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
models = {'disorder': load_model('../iqtree_merge/out/50R_disorder.paml'),
          'order': load_model('../../../data/matrices/LG.paml')}

if not os.path.exists('out/'):
    os.mkdir('out/')

OGids = [path.removesuffix('.tsv') for path in os.listdir('../asr_aa/out/') if path.endswith('.tsv')]
for OGid in OGids:
    # Load partition regions
    partitions = {}
    with open(f'../asr_aa/out/{OGid}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            regions = []
            for region in fields['regions'].split(','):
                start, stop = region.split('-')
                regions.append((int(start), int(stop)))
            transform, start0 = {}, 0
            for start, stop in regions:
                transform[(start, stop)] = (start0, stop - start + start0)
                start0 += stop - start
            partitions[fields['name']] = {'regions': regions, 'transform': transform}

    for name, partition in partitions.items():
        # Load tree
        aa_tree = skbio.read(f'../asr_aa/out/{OGid}_{name}.treefile', 'newick', skbio.TreeNode)
        tree = tree_template.shear([tip.name for tip in aa_tree.tips()])
        aa_length = aa_tree.descending_branch_length()
        length = tree.descending_branch_length()
        speed = aa_length / length

        # Load rate categories
        # In IQ-TREE, only the shape parameter is fit and the rate parameter beta is set to alpha so the mean of gamma distribution is 1
        # The calculations here directly correspond to equation 10 in Yang. J Mol Evol (1994) 39:306-314.
        # Note the equation has a small typo where the difference in gamma function evaluations should be divided by the probability
        # of that category since technically it is the rate given that category
        with open(f'../asr_aa/out/{OGid}_{name}.iqtree') as file:
            line = file.readline()
            while not line.startswith('Model of rate heterogeneity:'):
                line = file.readline()
            num_categories = int(line.rstrip('\n').split(' Invar+Gamma with ')[1][0])
            pinv = float(file.readline().rstrip('\n').split(': ')[1])
            alpha = float(file.readline().rstrip('\n').split(': ')[1])
        igfs = []  # Incomplete gamma function evaluations
        for i in range(num_categories+1):
            x = gamma.ppf(i/num_categories, a=alpha, scale=1/alpha)
            igfs.append(gammainc(alpha+1, alpha*x))
        rates = [(0, pinv)]
        for i in range(num_categories):
            rate = num_categories/(1-pinv) * (igfs[i+1] - igfs[i])
            rates.append((rate, (1-pinv)/num_categories))

        # Get model and partition MSA
        msa = list(read_fasta(f'../asr_aa/out/{OGid}_{name}.afa'))
        matrix, freqs = models[name]

        # Convert to vectors at tips of tree
        tips = {tip.name: tip for tip in tree.tips()}
        for header, seq in msa:
            spid = header.split()[0][1:]  # Split on white space, first field, trim >
            tip = tips[spid]
            value = np.zeros((len(alphabet), len(seq)))
            for j, sym in enumerate(seq):
                if sym in sym2idx:
                    i = sym2idx[sym]
                    value[i, j] = 1
                else:  # Use uniform distribution for ambiguous symbols
                    value[:, j] = 1 / len(alphabet)
            tip.value = value

        # Calculate likelihoods
        likelihoods = []

        # Get likelihood for invariant category
        # (Background probability for symbol if invariant; otherwise 0)
        likelihood = np.zeros((len(alphabet), len(msa[0][1])))
        for j in range(len(msa[0][1])):
            is_invariant = True
            sym0 = msa[0][1][j]
            if sym0 not in sym2idx:  # Gaps or ambiguous characters are not invariant
                is_invariant = False
            else:
                for i in range(1, len(msa)):
                    sym = msa[i][1][j]
                    if sym != sym0:
                        is_invariant = False
                        break
            if is_invariant:
                idx = sym2idx[sym0]
                likelihood[idx, j] = freqs[idx]
        likelihoods.append(likelihood * rates[0][1])  # Multiply by prior for category

        # Get likelihoods for rate categories
        for rate, prior in rates[1:]:  # Skip invariant
            s, conditional = get_conditional(tree, speed * rate * matrix)
            likelihood = np.expand_dims(freqs, -1) * conditional
            likelihoods.append(np.exp(s) * likelihood * prior)

        likelihoods = np.stack(likelihoods)
        likelihoods = likelihoods / likelihoods.sum(axis=(0, 1))

        partition.update({'num_categories': num_categories, 'pinv': pinv, 'alpha': alpha, 'speed': speed,
                          'rates': rates, 'likelihoods': likelihoods})

    # Concatenate partition likelihoods
    regions = []
    for partition_id, partition in partitions.items():
        regions.extend([(partition_id, start, stop) for start, stop in partition['regions']])
    regions = sorted(regions, key=lambda x: x[1])

    concatenate = []
    for partition_id, start0, stop0 in regions:
        start, stop = partitions[partition_id]['transform'][(start0, stop0)]
        likelihoods = partitions[partition_id]['likelihoods']
        concatenate.append(likelihoods[:, :, start:stop])
    concatenate = np.concatenate(concatenate, axis=2)

    np.save(f'out/{OGid}_aa.npy', concatenate)

    # Save model information as JSON
    for partition in partitions.values():
        del partition['likelihoods'], partition['transform']
    with open(f'out/{OGid}_aa_model.json', 'w') as file:
        json.dump(partitions, file)

"""
NOTES
The rates for each submodel are calculated manually because some non-invariant submodels are rounded to 0 in IQ-TREE's
output. This results in submodels with zero probability, which introduces problems when normalizing. I felt it was
important to preserve the original model structure when calculating the ASRs, so I decided against merging these
submodels with the invariant submodel.
"""