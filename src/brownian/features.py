""""Functions to calculate features associated with IDRs"""

import re

from localcider.sequenceParameters import SequenceParameters
from ipc import predict_isoelectric_point


# General functions
def count_group(seq, group):
    """Return count of residues matching amino acids in group."""
    count = 0
    for sym in seq:
        if sym in group:
            count += 1
    return count


def fraction_X(seq, x):
    """Return fraction of sequence matching amino acid x."""
    return seq.count(x) / len(seq)


def fraction_group(seq, group):
    """Return fraction of sequence matching amino acids in group."""
    count = count_group(seq, group)
    return count / len(seq)


def fraction_repeat(seq, group):
    """Return fraction of sequence matching a group of repeated residues."""
    matches = re.findall(f'[{group}]' + '{2,}', seq)
    count = 0
    for match in matches:
        count += len(match)
    return count / len(seq)


# Amino acid content
def get_features_aa(seq):
    """Return fractions of sequence matching individual amino acids SPTAHQNG as dictionary."""
    features = {}
    for sym in 'SPTAHQNG':
        features['fraction_' + sym] = fraction_X(seq, sym)
    return features


# Charge properties
def count_positive(seq):
    """Return count of positively charged residues."""
    return count_group(seq, set('RK'))


def count_negative(seq):
    """Return count of negatively charged residues."""
    return count_group(seq, set('DE'))


def FCR(seq):
    """Return fraction of charged residues in sequence."""
    return (count_positive(seq) + count_negative(seq)) / len(seq)


def NCPR(seq):
    """Return net charge per residue in sequence."""
    return (count_positive(seq) - count_negative(seq)) / len(seq)


def net_charge(seq):
    """Return net charge of sequence."""
    return count_positive(seq) - count_negative(seq)


def net_charge_P(seq):
    """Return net charging accounting for possible phosphorylated serine and threonine residues."""
    matches = re.findall('[ST]P', seq)
    return net_charge(seq) - 1.5 * len(matches)


def RK_ratio(seq):
    """Return adjusted ratio of arginine to lysine residues."""
    r = 1 + seq.count('R')
    k = 1 + seq.count('K')
    return r / k


def ED_ratio(seq):
    """Return adjusted ratio of aspartate to glutamate residues."""
    e = 1 + seq.count('E')
    d = 1 + seq.count('D')
    return e / d


def get_features_charge(seq):
    """Return dictionary of all features associated with charge."""
    SeqOb = SequenceParameters(seq)
    return {'FCR': FCR(seq), 'NCPR': NCPR(seq),
            'net_charge': net_charge(seq), 'net_charge_P': net_charge_P(seq),
            'RK_ratio': RK_ratio(seq), 'ED_ratio': ED_ratio(seq),
            'kappa': SeqOb.get_kappa(), 'omega': SeqOb.get_Omega(), 'SCD': SeqOb.get_SCD()}


# Physiochemical properties
def fraction_acidic(seq):
    """Return fraction of acidic residues in sequence."""
    return fraction_group(seq, set('DE'))


def fraction_basic(seq):
    """Return fraction of basic residues in sequence."""
    return fraction_group(seq, set('RK'))


def fraction_aliphatic(seq):
    """Return fraction of aliphatic residues in sequence."""
    return fraction_group(seq, set('ALMIV'))


def fraction_aromatic(seq):
    """Return fraction of aromatic residues in sequence."""
    return fraction_group(seq, set('FYW'))


def fraction_polar(seq):
    """Return fraction of polar residues in sequence."""
    return fraction_group(seq, set('QNSTCH'))


def fraction_disorder(seq):
    """Return fraction of disorder-promoting residues in sequence."""
    return fraction_group(seq, set('TAGRDHQKSEP'))


def fraction_chainexp(seq):
    """Return fraction of chain-expanding residues in sequence."""
    return fraction_group(seq, set('EDRKP'))


def get_features_physchem(seq):
    """Return dictionary of all features associated with physiochemical properties."""
    SeqOb = SequenceParameters(seq)
    return {'fraction_acidic': fraction_acidic(seq), 'fraction_basic': fraction_basic(seq),
            'fraction_aliphatic': fraction_aliphatic(seq), 'fraction_aromatic': fraction_aromatic(seq),
            'fraction_polar': fraction_polar(seq), 'fraction_disorder': fraction_disorder(seq), 'fraction_chainexp': fraction_chainexp(seq),
            'hydropathy': SeqOb.get_uversky_hydropathy(), 'isopoint': predict_isoelectric_point(seq),
            'length': len(seq), 'PPII_propensity': SeqOb.get_PPII_propensity()}


# Sequence complexity
def get_features_complexity(seq, repeat_groups):
    """Return dictionary of all features associated with sequence complexity."""
    features = {}
    for group in repeat_groups:
        features['repeat_' + group] = fraction_repeat(seq, group)
    features['wf_complexity'] = SequenceParameters(seq).get_linear_complexity(blobLen=len(seq))[1][0]  # Returns a 2xN matrix containing the complexity vector and the corresponding residue positions distributed equally along the sequence
    return features


# Motifs
def get_features_motifs(seq, motif_regexes):
    """Return dictionary of counts of motifs given in motif_regexes."""
    features = {}
    for motif, regex in motif_regexes.items():
        matches = re.findall(regex, seq)
        features[motif] = len(matches)
    return features


# Summary
def get_features(seq, repeat_groups, motif_regexes):
    """Return dictionary of all features keyed by (feature_label, group_label)."""
    feature_groups = [('aa_group', get_features_aa),
                      ('charge_group', get_features_charge),
                      ('physchem_group', get_features_physchem),
                      ('complexity_group', lambda x: get_features_complexity(x, repeat_groups)),
                      ('motifs_group', lambda x: get_features_motifs(x, motif_regexes))]
    features = {}
    for group_label, feature_function in feature_groups:
        features.update({(feature_label, group_label): feature_value for feature_label, feature_value in feature_function(seq).items()})
    return features


repeat_groups = ['Q', 'N', 'S', 'G', 'E', 'D', 'K', 'R', 'P', 'QN', 'RG', 'FG', 'SG', 'SR', 'KAP', 'PTS']
motif_regexes = {'CLV_Separin_Metazoa': r'E[IMPVL][MLVP]R.',
                 'DEG_APCC_KENBOX_2': r'.KEN.',
                 'DEG_APCC_TPR_1': r'.[ILM]R',
                 'DOC_CKS1_1': r'[MPVLIFWYQ].(T)P..',
                 'DOC_MAPK_DCC_7': r'[RK].{2,4}[LIVP]P.[LIV].[LIVMF]|[RK].{2,4}[LIVP].P[LIV].[LIVMF]',
                 'DOC_MAPK_gen_1': r'[KR]{0,2}[KR].{0,2}[KR].{2,4}[ILVM].[ILVF]',
                 'DOC_MAPK_HePTP_8': r'([LIV][^P][^P][RK]....[LIVMP].[LIV].[LIVMF])|([LIV][^P][^P][RK][RK]G.{4,7}[LIVMP].[LIV].[LIVMF])',
                 'DOC_PP1_RVXF_1': r'..[RK].{0,1}[VIL][^P][FW].',
                 'DOC_PP2B_PxIxI_1': r'.P[^P]I[^P][IV][^P]',
                 'LIG_APCC_Cbox_1': r'[DE]R[YFH][ILFVM][PAG].R',
                 'LIG_AP_GAE_1': r'[DE][DES][DEGAS]F[SGAD][DEAP][LVIMFD]',
                 'LIG_CaM_IQ_9': r'[ACLIVTM][^P][^P][ILVMFCT]Q[^P][^P][^P][RK][^P]{4,5}[RKQ][^P][^P]',
                 'LIG_EH_1': r'.NPF.',
                 'LIG_eIF4E_1': r'Y....L[VILMF]',
                 'LIG_GLEBS_BUB3_1': r'[EN][FYLW][NSQ].EE[ILMVF][^P][LIVMFA]',
                 'LIG_LIR_Gen_1': r'[EDST].{0,2}[WFY][^RKPGWFY][^PG][ILVFM]((.{0,4}[PLAFIVMY])|($)|(.{0,3}[ED]))',
                 'LIG_PCNA_PIPBox_1': r'[QM].[^FHWY][LIVM][^P][^PFWYMLIV](([FYHL][FYW])|([FYH][FYWL]))..',
                 'LIG_SUMO_SIM_par_1': r'[DEST]{0,5}.[VILPTM][VIL][DESTVILMA][VIL].{0,1}[DEST]{1,10}',
                 'MOD_CDK_SPxK_1': r'...([ST])P.[KR]',
                 'MOD_LATS_1': r'H.[KR]..([ST])[^P]',
                 'MOD_SUMO_for_1': r'[VILMAFP](K).E',
                 'TRG_ER_FFAT_1': r'[EDS].{0,4}[ED][FY][FYKREM][DE][AC].{1,2}[EDST]',
                 'TRG_Golgi_diPhe_1': r'Q.{6,6}FF.{6,7}',
                 'TRG_NLS_MonoExtN_4': r'(([PKR].{0,1}[^DE])|([PKR]))((K[RK])|(RK))(([^DE][KR])|([KR][^DE]))[^DE]',
                 'MOD_CDK_STP': r'[ST]P',
                 'MOD_MEC1': r'[ST]Q',
                 'MOD_PRK1': r'[LIVM]....TG',
                 'MOD_IPL1': r'[RK].[ST][LIV]',
                 'MOD_PKA': r'R[RK].S',
                 'MOD_CKII': r'[ST][DE].[DE]',
                 'MOD_IME2': r'RP.[ST]',
                 'DOC_PRO': r'P..P',
                 'TRG_ER_HDEL': r'HDEL',
                 'TRG_MITOCHONDRIA': r'[MR]L[RK]',
                 'MOD_ISOMERASE': r'C..C',
                 'TRG_FG': r'F.FG|GLFG',
                 'INT_RGG': r'RGG|RG'}
