from inspect import stack
from itertools import count
import numpy as np
import pandas as pd

from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder
from nnn import util

##### Helper Functions #####
def sort_stack(stack):
    return '_'.join(sorted(stack.split('_')))

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
    