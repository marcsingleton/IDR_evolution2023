"""Convert hits to a directed graph."""

import os
from itertools import groupby


def add_edge(qppid, sppid, bitscore, graph):
    try:
        graph[qppid].add((sppid, bitscore))
    except KeyError:
        graph[qppid] = {(sppid, bitscore)}


columns = {'qppid': str, 'qgnid': str, 'qspid': str,
           'sppid': str, 'sgnid': str, 'sspid': str,
           'hspnum': int, 'chspnum': int,
           'qlen': int, 'nqa': int, 'cnqa': int,
           'slen': int, 'nsa': int, 'cnsa': int,
           'bitscore': float}

# Make graph
graph = {}
for qspid in os.listdir('../hsps2hits/out/'):
    for sspid in os.listdir(f'../hsps2hits/out/{qspid}/'):
        with open(f'../hsps2hits/out/{qspid}/{sspid}') as file:
            file.readline()  # Skip header
            for _, group in groupby(file, lambda x: x.split()[0]):
                group = list([line.split() for line in group])
                bitscore = max([float(fields[14]) for fields in group])
                for fields in group:
                    if bitscore == float(fields[14]):  # Only record hits with maximum bitscore
                        hit = {column: f(field) for (column, f), field in zip(columns.items(), fields)}
                        qppid, sppid = hit['qppid'], hit['sppid']
                        qlen, cnqa = hit['qlen'], hit['cnqa']
                        bitscore = hit['bitscore']

                        if cnqa / qlen >= 0.5:
                            add_edge(qppid, sppid, bitscore, graph)

# Make output directory
if not os.path.exists('out/'):
    os.mkdir('out/')

# Write to file
with open('out/hit_graph.tsv', 'w') as file:
    for qppid, edges in graph.items():
        file.write(qppid + '\t' + ','.join([sppid + ':' + str(bitscore) for sppid, bitscore in edges]) + '\n')

"""
DEPENDENCIES
../hsps2hits/hsps2hits.py
    ../hsps2hits/out/*/*.tsv
"""