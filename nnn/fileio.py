import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json

from . import util

def read_santalucia_df(santalucia_file):
    santa_lucia = pd.read_csv(santalucia_file, sep='\t')
    santa_lucia['motif'] = santa_lucia['motif_paper'].apply(util.convert_santalucia_motif_representation)
    santa_lucia.drop(columns='motif_paper', inplace=True)
    
    return santa_lucia

def read_fitted_variant(filename, filter=True, annotation=None,
                        add_chisq_test=True, sodium=0.075):
    """
    Overwrites salt correction in the annotation df with sodium conc
    Args:
        annotation - df, if given, merge onto fitted variant file
        filter - Bool
    """
    df = pd.read_csv(filename, sep='\t').set_index('SEQID')
    if 'chisquared_all_clusters' in df.columns:
        df.rename(columns={'chisquared_all_clusters': 'chisq'}, inplace=True)
    df.rename(columns={s: s.replace('_final', '') for s in df.columns if s.endswith('_final')}, inplace=True)

    # Add dG and chi2 for old versions
    if not 'dG_37' in df.columns:
        df = util.add_dG_37(df)
    if add_chisq_test:
        pass
        # df = util.add_chisq_test(df)

    # Change all temperatures into celsius to avoid headaches later
    for col in ['Tm', 'Tm_lb', 'Tm_ub']:
        df[col] -= 273.15

    # Disambiguate standard error columns for params
    for c in df.columns:
        if (c.endswith('_std') and (not c.endswith('_norm_std'))):
            df.rename(columns={c: c.replace('_std', '_se')}, inplace=True)

    # Filter variants
    if filter:
        variant_filter = "n_clusters > 5 & dG_37_se < 2 & Tm_se < 25 & dH_se < 25 & RMSE < 0.5"
        df = util.filter_variant_table(df, variant_filter)

    # Optionally join annotation
    if annotation is not None:
        df = df.join(annotation, how='left')
        df['GC'] = df.RefSeq.apply(util.get_GC_content)
        df['Tm_NUPACK_salt_corrected'] = df.apply(lambda row: util.get_Na_adjusted_Tm(Tm=row.Tm_NUPACK, dH=row.dH_NUPACK, GC=row.GC, Na=sodium), axis=1)
        df['dG_37_NUPACK_salt_corrected'] = df.apply(lambda row: util.get_dG(dH=row.dH_NUPACK, Tm=row.Tm_NUPACK_salt_corrected, celsius=37), axis=1)
        
        for param in ['dH', 'dS']:
            df[param+'_NUPACK_salt_corrected'] = df[param+'_NUPACK']

    return df


def read_annotation(annotation_file, mastertable_file=None, sodium=None):
    """
    Older version required giving mastertable and merging. 
    Latest version simply reads in the annotation file.
    Returns:
        annotation - df indexed on 'SEQID' with construct class information
    """
    annotation = pd.read_csv(annotation_file, sep='\t').set_index('SEQID')

    # Rename for deprecated versions, does not affect new version
    annotation.rename(columns={c: c+'_NUPACK' for c in ['dH', 'dS', 'Tm', 'dG_37C']}, inplace=True)
    annotation.rename(columns={'dG_37C_NUPACK': 'dG_37_NUPACK'}, inplace=True)

    if mastertable_file is not None:
        mastertable = pd.read_csv(mastertable_file, sep='\t')
        annotation = annotation.reset_index().merge(mastertable[['Series', 'ConstructClass']], on='Series', how='left').set_index('SEQID')

    if sodium is not None:
        annotation['GC'] = annotation.RefSeq.apply(util.get_GC_content)
        annotation['Tm_NUPACK_salt_corrected'] = annotation.apply(lambda row: util.get_Na_adjusted_Tm(Tm=row.Tm_NUPACK, dH=row.dH_NUPACK, GC=row.GC, Na=sodium), axis=1)
        annotation['dG_37_NUPACK_salt_corrected'] = annotation.apply(lambda row: util.get_dG(dH=row.dH_NUPACK, Tm=row.Tm_NUPACK_salt_corrected, celsius=37), axis=1)
        
        for param in ['dH', 'dS']:
            annotation[param+'_NUPACK_salt_corrected'] = annotation[param+'_NUPACK']


    return annotation


def read_melt_file(melt_file):
    """
    Args:
        melt_file - str
    Returns:
        long-form dataframe, index not set
    """
    df = pd.read_csv(melt_file, header=1)
    melt = pd.DataFrame(data=df.values[:,:2], columns=['Temperature_C', 'Abs'])
    melt['ramp'] = 'melt'
    anneal = pd.DataFrame(data=df.values[:,2:4], columns=['Temperature_C', 'Abs'])
    anneal['ramp'] = 'anneal'
    
    return pd.concat((melt, anneal), axis=0)