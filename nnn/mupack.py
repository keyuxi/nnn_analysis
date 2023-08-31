import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple
from datetime import datetime
import os

from . import fileio, util, plotting

param_name_dict = {'dH':'dH', 'dG':'dG_37'}

""" Helper Functions """
def add_2_dict_val(mydict, value):
    for key in mydict:
        mydict[key] += value
    return mydict
    
def get_dict_mean_val(mydict):
    return np.nanmean(list(mydict.values()))

def center_new_param(old_dict, new_dict):
    new_mean = get_dict_mean_val(old_dict)
    new_dict = add_2_dict_val(new_dict, -1 * get_dict_mean_val(new_dict))
    new_dict = add_2_dict_val(new_dict, new_mean)
    return new_dict
    
""" Formatting Functions """

def update_template_dict(template_dict, coef_dict):
    """ 
    Overwrite tempate_dict with values in coef_dict
    Up to 2 levels deep as in the parameter file
    """
    ignored_keys = ['name', 'type', 'material', 'references', 'time_generated']
    new_dict = template_dict.copy()
    for param in template_dict:
        if not param in ignored_keys:
            if isinstance(template_dict[param], List) and param in coef_dict:
                new_dict[param] = coef_dict[param]
            elif isinstance(template_dict[param], dict):
                for key in template_dict[param]:
                    if key in coef_dict[param] and not (key in ignored_keys):                    
                        if isinstance(coef_dict[param][key], dict):
                            for p,v in coef_dict[param][key].items():
                                new_dict[param][key][p] = v
                        else:
                            new_dict[param][key] = coef_dict[param][key]
                
    return new_dict

        

def coef_df_2_dict(coef_df, template_dict=None):
    """
    Convert lr.coef_df to a NUPACK style dictionary
    without modifying the contents
    """
    coef_dict = defaultdict(dict)
    for _,row in coef_df.iterrows():
        pclass, pname = row.name.split('#')
        coef_dict[pclass][pname] = row.values[0]
          
    # Convet numerical keys & values to lists
    for key, sub_dict in coef_dict.items():
        if isinstance(sub_dict, dict):
            inds = list(sub_dict.keys())
            if inds[0].isdigit():
                inds = [int(x) for x in inds]
                if template_dict is None:
                    new_values = np.zeros(np.max(inds))
                else:
                    new_values = template_dict[key].copy()
                    
                for ind, value in sub_dict.items():
                    new_values[int(ind)-1] = value
                coef_dict[key] = list(new_values)
    
    # Overwrite tempate_dict
    if not template_dict is None:
        new_dict = update_template_dict(template_dict, coef_dict)
    else:
        new_dict = coef_dict
            
    return new_dict
    
    
def coef_dict_2_df(coef_dict):
    """
    NUPACK style dict to lr.coef_df style
    Args:
        coef_dict - param_set_dict['dH'] level
    """
    flat_coef_dict = defaultdict()
    for pclass in coef_dict:
        if not pclass in ['name', 'type', 'material', 'references', 'time_generated']:
            if isinstance(coef_dict[pclass], list):
                for i,value in enumerate(coef_dict[pclass]):
                    coef_name = '%s#%d' % (pclass, i+1)
                    flat_coef_dict[coef_name] = value
            elif isinstance(coef_dict[pclass], float):
                flat_coef_dict[pclass] = coef_dict[pclass]
            else:
                for pname in coef_dict[pclass]:
                    try:
                        coef_name = pclass + '#' + pname
                        flat_coef_dict[coef_name] = coef_dict[pclass][pname]
                    except:
                        print(coef_name)
        else:
            pass
            
    return pd.DataFrame(index=['value'], data=flat_coef_dict).T
    
    
def get_fixed_params(param_set_template_file:str, fixed_pclass:List[str],
                     features_not_fixed:List[str]=None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Gets the params in `param_set_template_file` that starts with a str in `fixed_pclass`.
    Returns:
        fixed_coef_df - dataframe, contains the values for the fixed parameters
    """
    param_set_dict = fileio.read_json(param_set_template_file)
    
    ori_coef_df = pd.concat((coef_dict_2_df(param_set_dict['dH']), coef_dict_2_df(param_set_dict['dG'])), axis=1)
    ori_coef_df.columns = ['dH', 'dG']

    fixed_coef_df = ori_coef_df.loc[[x for x in ori_coef_df.index if x.split('#')[0] in fixed_pclass]]
    if features_not_fixed is not None:
        fixed_coef_df.drop(labels=features_not_fixed, inplace=True)
        
    fixed_coef_df.fillna(0, inplace=True)
    fixed_feature_names = fixed_coef_df.index.tolist()
    
    return fixed_coef_df, fixed_feature_names
    
""" Convert parameters between different models """

def get_hairpin_seq_df(lr:util.LinearRegressionSVD, param:str, loop_len:int=3) -> pd.DataFrame:
    """
    Converts `feature_list.get_feature_list()` style hairpin parameters 
    to nupack hairpin_triloop or hairpin_tetraloop style parameters
    """
    loop_mid_param = lr.coef_df.loc[[x for x in lr.coef_df.index if x.endswith('_'+'.'*loop_len)]]
    hairpinmm_param = lr.coef_df.loc[[x for x in lr.coef_df.index if (x.endswith('_(.+.)') and (not x.startswith('x')))]]

    full_loop_list = []
    param_list = []
    for loop_mid in loop_mid_param.index:
        loop_mid_seq = loop_mid.split('_')[0]
        for hairpinmm in hairpinmm_param.index:
            # hairpinmm - 'NN+NN_(.+.)'
            nt1, nt2 = hairpinmm[1], hairpinmm[3]
            hairpinmm_seq = hairpinmm.split('_')[0]
            if nt1 == loop_mid_seq[0] and nt2 == loop_mid_seq[-1]:
                full_triloop = hairpinmm_seq[0] + loop_mid_seq + hairpinmm_seq[-1]
                full_loop_list.append(full_triloop)
                param_list.append(loop_mid_param.loc[loop_mid][0] + hairpinmm_param.loc[hairpinmm][0])

    loop_df = pd.DataFrame(index = full_loop_list, data=param_list, columns=[param])
    return loop_df
    
    
def get_hairpin_mismatch(lr:util.LinearRegressionSVD):
    """
    Formats `feature_list.get_feature_list()` style hairpin_mismatch `NN+NN_(.+.)`
    to NUPACK style
    Args:
        
    Returns:
        Dict, equivalent to `hairpin_dict['dH']['hairpin_mismatch']`
    """
    def convert_mm_name(x):
        seq = x.split('_')[0]
        return seq[-2:] + seq[:2]
        
    hairpinmm_param = lr.coef_df.loc[[x for x in lr.coef_df.index if (x.endswith('_(.+.)') and (not x.startswith('x')))]]
    hairpinmm_param.index = [convert_mm_name(x) for x in hairpinmm_param.index]
    param = hairpinmm_param.keys()[0]
    return hairpinmm_param.to_dict()[param]
    
    
def get_adjusted_triloop_terminal_penalty(hairpin_triloop_dict, terminal_penalty_dict):
    """
    Called by `lr_dict_2_nupack_json()`
    """
    for key,value in hairpin_triloop_dict.items():
        closing_pair = key[0] + key[-1]
        terminal_penalty = terminal_penalty_dict[closing_pair]
        hairpin_triloop_dict[key] = value - terminal_penalty
    
    return hairpin_triloop_dict
    
           
    
def lr_dict_2_nupack_json(lr_dict:util.LinearRegressionSVD, template_file:str, out_file:str, 
                          lr_step:str='full', adjust_triloop_terminal_penalty:bool=True,
                          extract_hairpin_mismatch:bool=False, comment=''):
    """
    Formats and saves the parameters from the final regression object to 
    NUPACK json file
    Args:
        lr_dict - Dict, keys 'dH' and 'dG', 
            values are instances of the LinearRegressionSVD() class
        lr_step - str, {'hairpin', 'full'}. If hairpin, only update the hairpin seq params
        adjust_trilop_terminal_penalty - bool, only used when lr_step = 'hairpin', adjust the 
            terminal penalty off the hairpin_triloop parameters
        extract_hairpin_mismatch - bool, if True, fit hairpin_mismatch parameters from triloop 
            and tetraloop parameters
    """
    param_name_dict = {'dH':'dH', 'dG':'dG_37'}
    
    ori_param_set_dict = fileio.read_json(template_file)
    param_set_dict = defaultdict()
    
    if lr_step == 'full':
        for p in param_name_dict:
            param_set_dict[p] = coef_df_2_dict(lr_dict[p].coef_df, template_dict=ori_param_set_dict[p])

        param_set_dict = update_template_dict(ori_param_set_dict, param_set_dict)
        
        ### Populate the loopup tables ###
        for p in param_name_dict:
            for n1,n2 in [(1,1), (1,2), (2,2)]:
                interior_name = 'interior_%d_%d' % (n1, n2)
                for seq in param_set_dict[p][interior_name]:
                    seq1, seq2 = seq[:n1+2], seq[n1+2:]
                    mm1 = seq2[-2] + seq2[-1] + seq1[0] + seq1[1]
                    mm2 = seq1[-2] + seq1[-1] + seq2[0] + seq2[1]
                    new_value = param_set_dict[p]['interior_size'][n1+n2-1] + \
                        param_set_dict[p]['interior_mismatch'][mm1] + \
                        param_set_dict[p]['interior_mismatch'][mm2]
                    param_set_dict[p][interior_name][seq] = new_value
                    
    elif lr_step == 'hairpin':
        hairpin_dict = {'dH': dict(hairpin_triloop=None, hairpin_tetraloop=None),
                        'dG': dict(hairpin_triloop=None, hairpin_tetraloop=None)}
        
        for p in param_name_dict:
            if extract_hairpin_mismatch:
                mm_dict = get_hairpin_mismatch(lr_dict[p])
                # hairpin_dict[p]['hairpin_mismatch'] = center_new_param(ori_param_set_dict[p]['hairpin_mismatch'], new_dict=mm_dict)

            hairpin_dict[p]['hairpin_tetraloop'] = get_hairpin_seq_df(lr_dict[p], p, loop_len=4).to_dict()[p]
            hairpin_dict[p]['hairpin_triloop'] = get_hairpin_seq_df(lr_dict[p], p, loop_len=3).to_dict()[p]
            
            if adjust_triloop_terminal_penalty:
                hairpin_dict[p]['hairpin_triloop'] = get_adjusted_triloop_terminal_penalty(
                    hairpin_dict[p]['hairpin_triloop'], ori_param_set_dict[p]['terminal_penalty'])
                
            # for hairpin_p in ['hairpin_triloop', 'hairpin_tetraloop']:
            #     hairpin_dict[p][hairpin_p] = center_new_param(ori_param_set_dict[p][hairpin_p], 
            #                                                         hairpin_dict[p][hairpin_p])
                              
        param_set_dict = update_template_dict(ori_param_set_dict, hairpin_dict)
    
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%dT%H:%M:%S")
    param_set_dict['time_generated'] = current_time
    param_set_dict['comment'] = comment
    param_set_dict['name'] = os.path.split(out_file)[1].replace('.json', '')
    
    fileio.write_json(param_set_dict, out_file)


""" Plotting functions """

def plot_mupack_nupack(data, x_suffix, param, lim, color_by_density=False):
    fig, ax = plt.subplots(1,2,figsize=(4,2))
    kwargs = dict(show_cbar=False, lim=lim, color_by_density=color_by_density)
    plotting.plot_colored_scatter_comparison(data=data, x=param+x_suffix, y=param+'_MUPACK', 
                                             ax=ax[0], **kwargs)
    plotting.plot_colored_scatter_comparison(data=data, x=param+x_suffix, y=param+'_NUPACK_salt_corrected', 
                                             ax=ax[1], **kwargs)
    mae = defaultdict()
    mae['new'] = util.mae(data[param+x_suffix], data[param+'_MUPACK'])
    mae['original'] = util.mae(data[param+x_suffix], data[param+'_NUPACK_salt_corrected'])
    ax[0].set_title('MAE = %.2f' % mae['new'])
    ax[1].set_title('MAE = %.2f' % mae['original'])
    plt.tight_layout()