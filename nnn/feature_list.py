import numpy as np
import pandas as pd

from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder



def get_stack_feature_list(row, stack_size=1):
    seq = 'x'*stack_size+row['RefSeq']+'y'*stack_size
    struct = '('*stack_size+row['TargetStruct']+')'*stack_size # has one more stack at the end

    loops = LoopExtruder(seq, struct, neighbor_bps=stack_size-1)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    return loops_cleaned+stacks_cleaned


def get_stack_feature_list_A(row, stack_size=1):
    seq = 'A'*(stack_size-1)+row['RefSeq']+'A'*(stack_size-1)
    struct = '('*(stack_size-1)+row['TargetStruct']+')'*(stack_size-1) # has one more stack at the end
    
    loops = LoopExtruder(seq, struct, neighbor_bps=stack_size-1)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    return loops_cleaned+stacks_cleaned


def get_stack_feature_list_simple_loop(row, stack_size=1):
    seq = 'x'*stack_size+row['RefSeq']+'y'*stack_size
    struct = '('*stack_size+row['TargetStruct']+')'*stack_size # has one more stack at the end

    loops = LoopExtruder(seq, struct, neighbor_bps=0)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    return loops_cleaned+stacks_cleaned
