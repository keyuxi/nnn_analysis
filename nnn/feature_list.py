from inspect import stack
from itertools import count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder
from ipynb.draw import draw_struct
from nnn import util

##### Helper Functions #####
def sort_stack(stack):
    return '_'.join(sorted(stack.split('_')))
    
    
def sort_stack_full(stack, sep='+'):
    seq, struct = stack.split('_')
    return sep.join(sorted(seq.split(sep))) + '_' + struct
    
    
def plot_elements(feats, ax=None):
    """
    Args:
        feats - A list of features with struct
    """
    if ax is None:
        fig, ax = plt.subplots(1, len(feats), figsize=(1.8*len(feats),1.8))
        
    for i, feat in enumerate(feats):
        seq, struct = feat.split('_')
        seq = seq.replace('x', 'N').replace('y', 'N').replace('+', ' ')
        # print(seq, struct)
        draw_struct(seq, struct, ax=ax[i])
    
    
##### Feature Extractors #####

def get_stack_feature_list(row, stack_size=1):

    pad = min(stack_size - 1, 1)
    # print('pad:', pad)
    seq = 'x'*pad+row['RefSeq']+'y'*pad
    struct = '('*pad+row['TargetStruct']+')'*pad # has one more stack at the end

    loops = LoopExtruder(seq, struct, neighbor_bps=stack_size-1)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    return loops_cleaned+stacks_cleaned

def get_mismatch_stack_feature_list(row, stack_size=1):
    """
    Assumes the mistmatch pair to be paired when extracting features
    """
    pad = min(stack_size - 1, 1)
    seq = 'x'*pad+row['RefSeq']+'y'*pad
    struct = '('*pad + util.get_symmetric_struct(len(row.RefSeq), 4) + ')'*pad # has one more stack at the end
    loops = LoopExtruder(seq, struct, neighbor_bps=stack_size-1)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)

    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks]

    return loops_cleaned + stacks_cleaned


def get_stack_feature_list_A(row, stack_size=1):
    """
    Considers the 'A's at the end of the construct.
    Not recommended.
    """
    seq = 'A'*(stack_size-1)+row['RefSeq']+'A'*(stack_size-1)
    struct = '('*(stack_size-1)+row['TargetStruct']+')'*(stack_size-1) # has one more stack at the end
    
    loops = LoopExtruder(seq, struct, neighbor_bps=stack_size-1)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    return loops_cleaned+stacks_cleaned


def get_stack_feature_list_simple_loop(row, stack_size=2, loop_base_size=0,
                                       fit_intercept=False, symmetry=False):
    """
    Args:
        loop_base_size - int, #stacks at the base of the loop to consider
        symmetry - bool, if set to True, view 2 symmetric motifs as the same
    """
        
    pad = stack_size - 1
    seq = 'x'*pad+row['RefSeq']+'y'*pad
    struct = '('*pad+row['TargetStruct']+')'*pad # has one more stack at the end

    loops = LoopExtruder(seq, struct, neighbor_bps=loop_base_size)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    if symmetry:
        stacks_cleaned = [sort_stack(x) for x in stacks_cleaned]
    
    if fit_intercept:
        return loops_cleaned + stacks_cleaned + ['intercept']
    else:
        return loops_cleaned + stacks_cleaned


def get_hairpin_loop_feature_list(row, stack_size=2, loop_base_size=0, 
                                  fit_intercept=False, symmetry=False,
                                  count_scaffold_list=None):
    """
    Extract features for hairpin loops.
    Use scaffold as a feature instead of stem WC stacks if `count_scaffold_list` is set
    Only count scaffold that is in `count_scaffold_list`
    """
    seq = row['RefSeq']
    struct = row['TargetStruct']
    scaffold = row['scaffold']
    
    loops = LoopExtruder(seq, struct, neighbor_bps=loop_base_size)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    if symmetry:
        stacks_cleaned = [sort_stack(x) for x in stacks_cleaned]
    feature_list = loops_cleaned
    
    if count_scaffold_list is not None:
        if scaffold in count_scaffold_list:
            feature_list += [scaffold]
        
    if fit_intercept:
        feature_list += ['intercept']
        
    return feature_list
    
    
def get_feature_list(row, stack_size:int=2, sep_base_stack:bool=False,
                     fit_intercept:bool=False, symmetry:bool=False, ignore_base_stack:bool=False):
    """
    Keep dot bracket in the feature to account for bulges and mismatches etc.
    Args:
        loop_base_size - int, #stacks at the base of the loop to consider. Fixed to 1
        sep_base_stack - bool, whether to separate base stack of hairpin loops to save parameters
        symmetry - bool, if set to True, view 2 symmetric motifs as the same
    """
    def clean(x):
        return x.replace(' ','+').replace(',', '_')
        
    pattern = re.compile(r'^\([.]+\)')
    loop_base_size = 1
    pad = stack_size - 1
    seq = 'x'*pad+row['RefSeq']+'y'*pad
    struct = '('*pad+row['TargetStruct']+')'*pad # has one more stack at the end

    loops = LoopExtruder(seq, struct, neighbor_bps=loop_base_size)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    loops_cleaned = [clean(x) for x in loops]
    stacks_cleaned = [clean(x) for x in stacks]
    
    if sep_base_stack:
        for loop in loops_cleaned:
            seq, struct = loop.split('_')
            if pattern.match(struct):
                hairpin_loop = LoopExtruder(seq, struct, neighbor_bps=0)[0]
                hairpin_stack = StackExtruder(seq, struct, stack_size=1)[0]
                loops_cleaned.remove(loop)
                
                if len(seq) <= 6:
                    loops_cleaned.append(clean(hairpin_loop))
                else:
                    loops_cleaned.append('NNNNN_.....')
                                   
                if not ignore_base_stack:
                    loops_cleaned.append(clean(hairpin_stack))
                    
            elif struct == '(..(+)..)':
                mm = f'{seq[1:3]}+{seq[6:8]}_..+..'
                mm_stack = f'{seq[0]}+{seq[3]}+{seq[5]}+{seq[8]}_(+(+)+)'
                loops_cleaned.remove(loop)
                loops_cleaned.append(mm)
                loops_cleaned.append(mm_stack)
                
    if symmetry:
        stacks = [sort_stack_full(x, sep='+') for x in stacks_cleaned]
        
    if row['Series'] == 'VARloop':
        for loop in loops_cleaned:
            seq, struct = loop.split('_')
            if pattern.match(struct):
                loops_cleaned.remove(loop)
                loops_cleaned.append('NNNNN_.....')
                
    
    if fit_intercept:
        return loops_cleaned + stacks_cleaned + ['intercept']
    else:
        return loops_cleaned + stacks_cleaned
        
def get_stem_nn_feature_list(row):
    dup_row = util.get_duplex_row(row)
    seq = dup_row.RefSeq
    # struct = dup_row.TargetStruct
    stem_len = int((len(seq) - 4.0) / 2.0)
    seq_pad = f'x{seq[:stem_len]}y+x{seq[-stem_len:]}y'
    nn_list = []
    for flag in range(stem_len + 1):
        if flag == 0:
            nn = seq_pad[:2] + '+' + seq_pad[-2:] + '_((+))'
        elif flag == stem_len:
            nn = seq_pad[-flag-2:-flag] + '+' + seq_pad[flag:flag+2] + '_((+))'
        else:
            nn = seq_pad[flag:flag+2] + '+' + seq_pad[-flag-2:-flag] + '_((+))'
        nn_list.append(nn)
        
    return nn_list