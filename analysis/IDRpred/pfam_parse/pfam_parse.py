"""Parse raw output from Pfam searches."""

import os
from collections import namedtuple

field_names = ['file_id',
               'target_name', 'target_accession', 'tlen',
               'query_name', 'query_accession', 'qlen',
               'seq_evalue', 'seq_score', 'seq_bias',
               'dom_num', 'dom_id',
               'c_evalue', 'i_evalue', 'dom_score', 'dom_bias',
               'hmm_from', 'hmm_to',
               'ali_from', 'ali_to',
               'env_from', 'env_to',
               'acc', 'description']
Record = namedtuple('Record', field_names)


def read_pfam_table(path, file_id):
    records = []
    with open(path) as file:
        for line in file:
            if line.startswith('#'):
                continue

            split_line = line.rstrip('\n').split()
            field_values = [file_id]
            field_values.extend(split_line[:22])
            field_values.append(' '.join(split_line[22:]))  # Re-join description with spaces; may not regenerate original string
            records.append(Record(*field_values))
    return records


records = []
OGids = [path.removesuffix('.txt') for path in os.listdir('../pfam_search/out/') if path.endswith('.txt')]
for OGid in OGids:
    records.extend(read_pfam_table(f'../pfam_search/out/{OGid}.txt', OGid))

if not os.path.exists('out/'):
    os.mkdir('out/')

with open('out/parsed.tsv', 'w') as file:
    file.write('\t'.join(field_names) + '\n')
    for record in records:
        line = '\t'.join([getattr(record, field_name) for field_name in field_names]) + '\n'
        file.write(line)

"""
NOTES
For parsing the HMMER output, it's safe to assume the first 22 fields are actually whitespace delimited because HMMER
truncates the query name at the first whitespace character.
"""