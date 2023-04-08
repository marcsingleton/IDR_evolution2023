"""Filter gene association file."""

import os
import re
from functools import reduce
from operator import add

import matplotlib.pyplot as plt
import pandas as pd
from src.utils import read_fasta


def get_ancestors(GO, GOid):
    ancestors = set()
    parent_stack = GO[GOid]['parents'].copy()
    while parent_stack:
        parent = parent_stack.pop()
        ancestors.add(parent)
        parent_stack.extend(GO[parent]['parents'])
    return ancestors


def write_table(counts, title):
    if counts.empty:  # Immediately return on empty table
        return
    if os.path.exists('out/output.txt'):
        mode = 'a'
        padding = '\n'
    else:
        mode = 'w'
        padding = ''

    table_string = counts.head(10).to_string()
    length = len(table_string.split('\n')[1])
    total_string = str(counts.sum()).rjust(length - 6, ' ')
    output = f"""\
    {title}
    {table_string}

    TOTAL {total_string}
    """
    with open('out/output.txt', mode) as file:
        file.write(padding + output)


ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
gnid_regex = r'gnid=([A-Za-z0-9_.]+)'
min_length = 30
min_gnids = 50  # Minimum number of unique genes associated with a term to maintain it in set

# Load sequence data
rows = []
ppid2gnid = {}
OGids = sorted([path.removesuffix('.afa') for path in os.listdir('../../../data/alignments/fastas/') if path.endswith('.afa')])
for OGid in OGids:
    for header, _ in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        gnid = re.search(gnid_regex, header).group(1)
        rows.append({'OGid': OGid, 'gnid': gnid})
        ppid2gnid[ppid] = gnid
all_genes = pd.DataFrame(rows)

# Load ontology
GO = {}
with open('../../../data/GO/go-basic.obo') as file:
    line = file.readline()
    while line:
        if line.startswith('[Term]'):
            parents = []
            alt_ids = []
            is_obsolete = False
            line = file.readline()
            while line and not line.startswith('['):
                if line.startswith('id:'):
                    GOid = line[4:-1]
                if line.startswith('name:'):
                    name = line[5:-1]
                if line.startswith('alt_id:'):
                    alt_ids.append(line[8:-1])
                if line.startswith('is_a:'):
                    parents.append(line.split('!')[0][6:-1])
                if line.startswith('is_obsolete'):
                    is_obsolete = line[13:-1] == 'true'
                line = file.readline()
            GO[GOid] = {'name': name, 'primary_id': GOid, 'alt_ids': alt_ids, 'parents': parents, 'is_obsolete': is_obsolete}
            for alt_id in alt_ids:
                GO[alt_id] = {'name': name, 'primary_id': GOid, 'alt_ids': [], 'parents': [], 'is_obsolete': is_obsolete}
        else:  # Else suite is necessary so new [Term] line isn't skipped exiting if suite
            line = file.readline()

# Find term ancestors
rows = []
for GOid in GO:
    rows.append({'GOid': GOid, 'ancestor_id': GOid, 'ancestor_name': GO[GOid]['name']})  # Include self as ancestor so merge keeps original term
    for ancestor_id in get_ancestors(GO, GOid):
        rows.append({'GOid': GOid, 'ancestor_id': ancestor_id, 'ancestor_name': GO[ancestor_id]['name']})
ancestors = pd.DataFrame(rows)

# Load regions as segments
rows = []
with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        for ppid in fields['ppids'].split(','):
            rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder, 'gnid': ppid2gnid[ppid]})
all_segments = pd.DataFrame(rows)

# Load raw table and add term names
gaf1 = pd.read_table('../../../data/GO/dmel_r6.45_FB2022_02_gene_association.fb',
                     skiprows=5,
                     usecols=list(range(15)),  # File contains two spare tabs at end
                     names=['DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO ID',  # Official column labels
                            'DB:Reference', 'Evidence', 'With (or) From', 'Aspect', 'DB_Object_Name',
                            'DB_Object_Synonym', 'DB_Object_Type', 'taxon', 'Date', 'Assigned_by'])
gaf1['name'] = gaf1['GO ID'].apply(lambda x: GO[x]['name'])

# Drop unneeded columns and filter
mapper = {'DB_Object_ID': 'gnid', 'DB_Object_Symbol': 'symbol', 'Qualifier': 'qualifier', 'GO ID': 'GOid',
          'Evidence': 'evidence', 'Aspect': 'aspect', 'Date': 'date', 'Assigned_by': 'origin'}
columns = ['DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO ID',
           'Evidence', 'Aspect', 'taxon', 'Date', 'Assigned_by', 'name']
gaf2 = gaf1[columns].rename(columns=mapper)

bool1 = gaf2['qualifier'].isin(['enables', 'contributes_to', 'involved_in',  # Select appropriate qualifiers (FB defines additional qualifiers)
                               'located_in', 'part_of', 'is_active_in'])
bool2 = gaf2['evidence'].isin(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP',  # Remove lower quality annotations
                               'HTP', 'HDA', 'HMP', 'HGI', 'HEP',
                               'TAS', 'IC'])
bool3 = gaf2['taxon'] == 'taxon:7227'  # Keep only dmel annotations
bool4 = ~gaf2['GOid'].apply(lambda x: GO[x]['is_obsolete'])  # Remove obsolete annotations
gaf3 = gaf2[bool1 & bool2 & bool3 & bool4].drop(['qualifier', 'taxon'], axis=1)

if not os.path.exists('out/'):
    os.mkdir('out/')

# Identify IDs of obsolete and renamed annotations
counts = gaf2.loc[~bool4, ['GOid', 'name']].value_counts()
write_table(counts, 'TOP 10 OBSOLETE ANNOTATIONS (ORIGINAL)')

counts = gaf2.loc[bool1 & bool2 & bool3 & ~bool4, ['GOid', 'name']].value_counts()
write_table(counts, 'TOP 10 OBSOLETE ANNOTATIONS (FILTERED)')

# Update IDs
gaf4 = gaf3.copy()
gaf4['GOid'] = gaf4['GOid'].apply(lambda x: GO[x]['primary_id'])

counts = gaf3.loc[gaf3['GOid'] != gaf4['GOid'], ['GOid', 'name']].value_counts()
write_table(counts, 'TOP 10 RENAMED ANNOTATIONS')

# Join with segments
joins = [(all_genes, ['GOid'], 'genes'),
         (all_segments, ['GOid', 'disorder'], 'regions')]
for df, group_keys, prefix in joins:
    if not os.path.exists(f'out/{prefix}/'):
        os.mkdir(f'out/{prefix}/')
    prefix = f'out/{prefix}/'

    gaf5 = df.merge(gaf4, on='gnid')

    # Propagate ancestors to table and drop poorly represented annotations
    # The min_gnid filter is applied to GO terms grouped by disorder subset, so the subset models fulfill the requirement
    gaf6 = gaf5.merge(ancestors, on='GOid').drop(['GOid', 'name'], axis=1)
    gaf6 = gaf6.rename(columns={'ancestor_id': 'GOid', 'ancestor_name': 'name'}).drop_duplicates()
    gaf7 = gaf6.groupby(group_keys).filter(lambda x: x['gnid'].nunique() >= min_gnids)

    # Make plots
    gafs = [gaf2, gaf3, gaf4, gaf5, gaf6, gaf7]
    labels = ['original', 'filter', 'update', 'join', 'propagate', 'drop']

    # Number of annotations
    fig, ax = plt.subplots()
    ax.bar(range(len(gafs)), [len(gaf) for gaf in gafs], width=0.5, tick_label=labels)
    ax.set_xlabel('Cleaning step')
    ax.set_ylabel('Number of annotations')
    fig.savefig(f'{prefix}/bar_numannot-gaf.png')
    plt.close()

    # Number of annotations by aspect
    fig, ax = plt.subplots()
    counts = [gaf['aspect'].value_counts() for gaf in gafs]
    bottoms = [0 for count in counts]
    for aspect, aspect_label in [('P', 'Process'), ('F', 'Function'), ('C', 'Component')]:
        ax.bar(range(len(counts)), [count[aspect] for count in counts],
               bottom=bottoms, label=aspect_label, width=0.5, tick_label=labels)
        bottoms = [b + count[aspect] for b, count in zip(bottoms, counts)]
    ax.set_xlabel('Cleaning step')
    ax.set_ylabel('Number of annotations')
    ax.legend()
    fig.savefig(f'{prefix}/bar_numannot-gaf_aspect.png')
    plt.close()

    # Number of annotations by evidence code
    counts = [gaf['evidence'].value_counts() for gaf in gafs]
    codes = reduce(lambda x, y: x.combine(y, add, fill_value=0), counts).sort_values(ascending=False)
    top_codes = list(codes.index[:9])
    other_codes = list(codes.index[9:])
    merged_counts = []
    for count in counts:
        merged_count = {code: count.get(code, 0) for code in top_codes}
        merged_count['other'] = sum([count.get(code, 0) for code in other_codes])
        merged_counts.append(merged_count)

    fig, ax = plt.subplots()
    counts = merged_counts
    bottoms = [0 for _ in counts]
    for code in (top_codes + ['other']):
        ax.bar(range(len(counts)), [count[code] for count in counts],
               bottom=bottoms, label=code, width=0.5, tick_label=labels)
        bottoms = [b + count[code] for b, count in zip(bottoms, counts)]
    ax.set_xlabel('Cleaning step')
    ax.set_ylabel('Number of annotations')
    ax.legend()
    fig.savefig(f'{prefix}/bar_numannot-gaf_evidence.png')
    plt.close()

    # Number of terms
    fig, ax = plt.subplots()
    ax.bar(range(len(gafs)), [gaf['GOid'].nunique() for gaf in gafs], width=0.5, tick_label=labels)
    ax.set_xlabel('Cleaning step')
    ax.set_ylabel('Number of unique terms')
    fig.savefig(f'{prefix}/bar_numterms-gaf.png')
    plt.close()

    # Number of terms by aspect
    fig, ax = plt.subplots()
    counts = [gaf[['GOid', 'aspect']].drop_duplicates()['aspect'].value_counts() for gaf in gafs]
    bottoms = [0 for count in counts]
    for aspect, aspect_label in [('P', 'Process'), ('F', 'Function'), ('C', 'Component')]:
        ax.bar(range(len(counts)), [count[aspect] for count in counts],
               bottom=bottoms, label=aspect_label, width=0.5, tick_label=labels)
        bottoms = [b + count[aspect] for b, count in zip(bottoms, counts)]
    ax.set_xlabel('Cleaning step')
    ax.set_ylabel('Number of unique terms')
    ax.legend()
    fig.savefig(f'{prefix}/bar_numterms-gaf_aspect.png')
    plt.close()

    # Write GAFs to file
    for gaf, label in zip(gafs[2:], labels[2:]):
        gaf.to_csv(f'{prefix}/GAF_{label}.tsv', sep='\t', index=False)
