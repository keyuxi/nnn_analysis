import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json

from util import *

def read_santalucia_df(santalucia_file):
    santa_lucia = pd.read_csv(santalucia_file, sep='\t')
    santa_lucia['motif'] = santa_lucia['motif_paper'].apply(convert_santalucia_motif_representation)
    santa_lucia.drop(columns='motif_paper', inplace=True)
    
    return santa_lucia

def read_fitted_variant(filename, filter=True, annotation=None):
    """
    Args:
        annotation - df, if given, merge onto fitted variant file
    """
    df = pd.read_csv(filename, sep='\t').set_index('SEQID')
    if 'chisquared_all_clusters' in df.columns:
        df.rename(columns={'chisquared_all_clusters': 'chisq'}, inplace=True)
    df.rename(columns={s: s.replace('_final', '') for s in df.columns if s.endswith('_final')}, inplace=True)

    # Add dG and chi2
    if not 'dG_37' in df.columns:
        df = add_dG_chi2_test(df)

    # Add standard error columns for params
    sqrt_n = np.sqrt(df['n_clusters'])
    for c in df.columns:
        if (c.endswith('_std') and (not c.endswith('_norm_std'))):
            df[c.replace('_std', '_se')] = df[c] / sqrt_n

    # Filter variants
    if filter:
        variant_filter = "n_clusters > 5 & dG_37_se < 0.2 & Tm_se < 2.5 & dH_se < 2.5 & RMSE < 0.5"
        df = filter_variant_table(df, variant_filter)

    if annotation is not None:
        df = df.join(annotation, how='left')

    return df


def read_annotation(annotation_file, mastertable_file):
    """
    Returns:
        annotation - df indexed on 'SEQID' with construct class information
    """
    annotation = pd.read_csv(annotation_file, sep='\t').set_index('SEQID')
    annotation.rename(columns={c: c+'_NUPACK' for c in ['dH', 'dS', 'Tm', 'dG_37C']}, inplace=True)
    annotation.rename(columns={'dG_37C_NUPACK': 'dG_37_NUPACK'}, inplace=True)

    mastertable = pd.read_csv(mastertable_file, sep='\t')
    annotation = annotation.reset_index().merge(mastertable[['Series', 'ConstructClass']], on='Series', how='left').set_index('SEQID')

