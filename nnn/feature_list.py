import numpy as np
import pandas as pd

from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder
from nnn import util


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
    seq = 'A'*(stack_size-1)+row['RefSeq']+'A'*(stack_size-1)
    struct = '('*(stack_size-1)+row['TargetStruct']+')'*(stack_size-1) # has one more stack at the end
    
    loops = LoopExtruder(seq, struct, neighbor_bps=stack_size-1)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    return loops_cleaned+stacks_cleaned


def get_stack_feature_list_simple_loop(row, stack_size=1):
    pad = stack_size - 1
    seq = 'x'*pad+row['RefSeq']+'y'*pad
    struct = '('*pad+row['TargetStruct']+')'*pad # has one more stack at the end

    loops = LoopExtruder(seq, struct, neighbor_bps=0)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    return loops_cleaned+stacks_cleaned
