#!/usr/bin/env python
# coding: utf-8

"""
This script updates NNNlib2b_annotation_20220314.txt to NNNlib2b_annotation_20220316.tsv
    - Resolve naming and ID conflicts
    - Add TargetStruct
    - Add correctly calculated 2-state nupack params and ensemble nupack dG_37
    - Add Yuxi's design

Yuxi, 03/16/2022
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from nupack import *

from nnn.util import *
from nnn.fileio import *
from nnn.make_lib2b_annotations import *
from make_lib3 import *

import warnings
warnings.filterwarnings("ignore")

# read in
annotation_updated = pd.read_table('./data/annotation/NNNlib2b_annotation_20220314.tsv')
annotation_updated.drop_duplicates(inplace=True)
mastertable_updated = pd.read_table('./data/annotation/NNN_mastertable_updated.tsv')

# cleaning up naming problems and conflicts
annotation_updated.loc[annotation_updated.ConstructType == 'VariableLoops', 'Series'] = 'VARloop'
annotation_updated.loc[annotation_updated.Series == 'VARloop', 'SEQID'] = \
    annotation_updated.loc[annotation_updated.Series == 'VARloop', 'SEQID'].apply(lambda x: x.replace('PNTC', 'VAR'))

annotation_updated.loc[annotation_updated.ConstructType == 'pseudoknots', 'Series'] = 'External'
annotation_updated.loc[annotation_updated.Series == 'WBcontrols', 'Series'] = 'External'
annotation_updated.loc[annotation_updated.ConstructType == 'winstons_original_constructs', 'ConstructType'] = 'WBcontrols'
annotation_updated.loc[annotation_updated.ConstructType == 'PUMcontrols', 'Series'] = 'External'

annotation_updated.loc[annotation_updated.Series == 'BaeControls', 'ConstructType'] = 'BaeControls'
annotation_updated.loc[annotation_updated.Series == 'BaeControls', 'Series'] = 'External'

annotation_updated.loc[annotation_updated.Series == 'StemDangle', 'ConstructType'] = \
    annotation_updated.loc[annotation_updated.Series == 'StemDangle', 'ConstructType'].apply(lambda x: x.replace('rime_dangleAlen', '').replace('strong_stem_var', 'StemDangle'))
annotation_updated.loc[annotation_updated.Series == 'StemDangle', 'Series'] = 'Control'

annotation_updated.loc[annotation_updated.Series == 'SuperStem', 'ConstructType'] = 'SuperStem'
annotation_updated.loc[annotation_updated.Series == 'SuperStem', 'Series'] = 'Control'

# check the Series agree with the master table
assert(len(set(annotation_updated.Series) - set(mastertable_updated.Series)) == 0)

# get TargetStruct
annotation_updated['TargetStruct'] = annotation_updated.apply(get_target_struct, axis=1)
scaffolds = ['GCGC','CGCGCGCG', 'GATCGATC']
bulges_lib, bulges_names = make_bulges([scaffolds[1], scaffolds[2]])
bulgesNNN_lib, bulgesNNN_names = make_bulges_NNN([scaffolds[0]])

get_bulges_target_struct = lambda bulges_names, scaffold_ind: np.array([[row.replace('\n','').split('\t')[-2], row.replace('\n','').split('\t')[-1]] for row in bulges_names[scaffolds[scaffold_ind]]])

bulges = np.concatenate((get_bulges_target_struct(bulges_names, 1),
                         get_bulges_target_struct(bulges_names, 2),
                         get_bulges_target_struct(bulgesNNN_names, 0)), axis=0)

for i in tqdm(range(bulges.shape[0])):
    annotation_updated.loc[annotation_updated.eval('RefSeq == "%s"' % bulges[i,0]), 'TargetStruct'] = bulges[i,1]

# fix nupack param annotations
annotation_updated.rename(columns={'dG_37_NUPACK': 'dG_37_ensemble_NUPACK'}, inplace=True)
nupack_param = annotation_updated.progress_apply(lambda row: get_nupack_dH_dS_Tm_dG_37(row.RefSeq, row.TargetStruct), axis=1)
annotation_updated[['dH_NUPACK', 'dS_NUPACK', 'Tm_NUPACK', 'dG_37_NUPACK']] = np.array(nupack_param.values.tolist())

# add mismatch 3mer
scaffolds = ['GC','CGCG','GATC']
scaffolds = get_rc(scaffolds)
mismatches_3mer = make_3mer_mismatches(scaffolds, add_loop=True, filter_3_mismatches=True)
df_3mer = pd.DataFrame(mismatches_3mer).melt()
df_3mer.columns = ['scaffold', 'RefSeq']
df_3mer['SEQID'] = ['MMTM%d'%i for i in range(len(df_3mer))]
df_3mer['Series'] = 'MisMatches'
df_3mer['ConstructType'] = '3mer'
df_3mer['topScaffold'] = df_3mer.scaffold.apply(lambda x: get_top_bottom_scaffold(x, 'top'))
df_3mer['bottomScaffold'] = df_3mer.scaffold.apply(lambda x: get_top_bottom_scaffold(x, 'bottom'))
df_3mer['TargetStruct'] = df_3mer.apply(get_target_struct, axis=1)
nupack_param = df_3mer.progress_apply(lambda row: get_nupack_dH_dS_Tm_dG_37(row.RefSeq, get_target_struct(row)), axis=1)
df_3mer[['dH_NUPACK', 'dS_NUPACK', 'Tm_NUPACK', 'dG_37_NUPACK']] = np.array(nupack_param.values.tolist())
df_3mer['dG_37_ensemble_NUPACK'] = df_3mer.RefSeq.progress_apply(lambda seq: get_seq_ensemble_dG(seq, 37))

# concat 2 parts and write to file
annotation_all = pd.concat([annotation_updated, df_3mer.drop(columns='scaffold')])
annotation_all.to_csv('./data/annotation/NNNlib2b_annotation_20220316.tsv', sep='\t', index=False)