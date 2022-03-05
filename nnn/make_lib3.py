#!/usr/bin/env python

from numpy.core.fromnumeric import var
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')
import os
import numpy as np
# from arnie.pfunc import pfunc
# from arnie.free_energy import free_energy
# from arnie.bpps import bpps
# from arnie.mfe import mfe
# import arnie.utils as utils
from decimal import Decimal
import operator as op
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

import make_lib2 as lib2

# hard-coded loop
LOOP = 'GAAA'

##### HELPER FUNCTIONS #####
def rcompliment(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', '-':'-'}
    return "".join(complement.get(base, base) for base in reversed(seq))

def is_WC(pair):
    """
    If a pair is WC
    Args:
        pair - str, len=2
    Returns:
        bool
    """
    WC_pairs = ['AT', 'TA', 'CG', 'GC']
    return pair in WC_pairs

def is_stack_legal(var_stack):
    """
    Whether a stack satysfies hard constraints:
        WC pair at start and end;
        len < 18
    Args:
        var_stack - List[str], each element is a pair
    Returns:
        is_legal - bool
    """
    is_legal = None

    pass

    return is_legal

def add_scaffold_2_stacks(stacks, scaffold):
    """
    Insert a list of variable stacks to one scaffold
    Args:
        stacks - List[List[str]]
        scaffold - str
    Returns:
        seqs - List[str]
    """
    full_scaffold = scaffold + rcompliment(scaffold)
    split = np.round(len(scaffold) / 4).astype(int)
    sca_start, sca_mid, sca_end = full_scaffold[0:split+1], full_scaffold[split+1:-split-1], full_scaffold[-split-1:]
    seqs = [(sca_start + ''.join([s[0] for s in var_stack]) + sca_mid + ''.join([s[1] for s in reversed(var_stack)]) + sca_end) \
        for var_stack in stacks]
    return seqs

def add_loop_2_seqs(loop, seqs):
    """
    Add one loop sequence to a list of sequences.
    Args:
        loop - str
        seqs - List[str]
    Returns:
        loop_seqs - List[str]
    """
    split = int(len(seqs[0]) / 2)
    loop_seqs = [s[0:split] + loop + s[split:] for s in seqs]
    return loop_seqs

def write_lib_2_txt(filename, var_lib, separate_scaffold=False):
    """
    Args:
        filename - str. If `separate_scaffold` is True, add '_scaffold-{scaffold}'
            to the end of the filename.
        var_lib - Dict[str: List[str]], where the keys are scaffolds and the values 
            the lists of sequences.
        separate_scaffold - bool. If true, write sequences with different scaffolds 
            to separate files
    Returns:
        None
    """
    if separate_scaffold:
        for scaffold in var_lib.keys():
            sub_filename = os.path.splitext(filename)[0] + '_scaffold-' + scaffold + \
                os.path.splitext(filename)[1]
            with open(sub_filename, 'w') as fh:
                fh.write('\n'.join(var_lib[scaffold]))
                print('Wrote %d sequences to %s' % (len(var_lib[scaffold]), sub_filename))
    else:
        with open(filename, 'w') as fh:
            for scaffold in var_lib.keys():
                fh.write('\n'.join(var_lib[scaffold]))
                print('Wrote %d sequences to %s' % (len(var_lib[scaffold]), filename))

        

##### FOR MORE COMPLICATED DESIGNS #####
def get_feature_matrix(var_stacks, features):
    """
    Get the #sequence by #feature matrix for stacks.
    Args:
        var_stacks - 
    """
    pass

##### DEFINE BASIC BLOCKS #####

def def_pairs():
    """
    Reutrns a list of all possible nt pairs, both WC and mismatch
    Returns:
        pairs - List[str]
    """
    nts = ['A', 'T', 'C', 'G']
    pairs = []

    for x in nts:
        for y in nts:
            pairs.append(x + y)

    return pairs

def def_3mer_mismatches(filter_3_mismatches=False):
    """
    Enumerates all the k=3 stacks of mismatches, including WCs.
    Each stack is represented by pairs from the bottom of the stem up.
    Args:
        filter_3_mismatches - bool. If True, filter out the sequences where all 3 are mismatches.
    Returns:
        var_stack - List[List[str]], variable area, where every element is a list of pairs.
    """
    pairs = def_pairs()
    var_stack = []

    if not filter_3_mismatches:
        for p1 in pairs:
            for p2 in pairs:
                for p3 in pairs:
                    var_stack.append([p1, p2, p3])
    else:
        for p1 in pairs:
            for p2 in pairs:
                for p3 in pairs:
                    if not ((not is_WC(p1)) and (not is_WC(p2)) and (not is_WC(p3))):
                        var_stack.append([p1, p2, p3])

    return var_stack

##### MAKE VARIANTS #####

def make_3mer_mismatches(scaffolds, add_loop=True, filter_3_mismatches=False):
    """
    Make the 5' to 3' sequence with both the constant scafold and the variable area.
    Args:
        scaffolds - List[str], where each str is a scaffold written in 5' to 3' to concat and test
    Returns:
        var_lib - Dict[str: List[str]], where the keys are scaffolds and the values the lists of 
            5'->3' concatenated sequences 
    """
    var_lib = defaultdict()
    var_stack = def_3mer_mismatches(filter_3_mismatches)

    for scaffold in scaffolds:
        seqs = add_scaffold_2_stacks(var_stack, scaffold)

        if add_loop:
            seqs = add_loop_2_seqs(LOOP, seqs)

        var_lib[scaffold] = seqs

    return var_lib


if __name__ == '__main__':
     
    #define the scaffolds you want to test
    scaffolds = ['GC','CGCG','GATC']
    scaffolds = lib2.get_rc(scaffolds)

    maxlength = 40
    # PKs = lib2.read_PK_file('short_pseudoknots.txt')
    
    #make all the variants for each scaffold
    mismatch_all_3mer_lib = make_3mer_mismatches(scaffolds, add_loop=True)
    write_lib_2_txt(r'../out/mismatch_all_3mer.txt', mismatch_all_3mer_lib, separate_scaffold=True)
    
    #filter out the 3 mismatches ones
    mismatch_1and2_3mer_lib = make_3mer_mismatches(scaffolds, add_loop=True, filter_3_mismatches=True)
    write_lib_2_txt(r'../out/mismatch_1and2_3mer.txt', mismatch_1and2_3mer_lib, separate_scaffold=True)