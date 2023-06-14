import numpy as np
import pandas as pd

from nnn import util


def get_model_prediction(df=None, seq_list=None, struct_list=None, sodium:float=0.083,
                         model:str='nupack', model_param_file:str='./models/dna-nnn-full.json', 
                         append_df_suffix=None):
    """
    Predicts thermodynamic parameters from input sequences and (optionally) structures
    Input is either df or list of strings
    Args:
        df - dataframe. ignores seq_list and struct_list if given. Uses column names
            `RefSeq` and `TargetStruct`
        seq_list, struct_list - List[str]
        sodium - float, in M
        model - str, {'nupack', 'linear_regression', 'knn', 'gnn', 'unet'}
            all models other than unet requires structures
        model_param_file - str, points to the file to use for the model
        append_df_suffix - If given and there is an input df, append the prediction to the 
            df and return. Otherwise, only return the prediction values.
    Returns:
        If given a dataframe, returns a dataframe, either appended to the input or
        just the prediction. If given lists, returns result_df
    """
    # pnames = ['dH', 'Tm', 'dG_37', 'dS']
    # result_df = pd.DataFrame(columns=pnames)
    
    if model in {'nupack', 'linear_regression', 'knn', 'gnn'}:
        # requires structures
        # == format input ==
        if df is not None:
            seq_list = df.RefSeq
            struct_list = df.TargetStruct
        else:
            assert len(seq_list) == struct_list
            
        # run model prediction
        result_df = globals()[f'run_{model}'](seq_list, struct_list, sodium, model_param_file)
            
    elif model == 'unet':
        # TODO
        return result_df
    else:
        raise "model has to be one of {'nupack', 'linear_regression', 'knn', 'gnn', 'unet'}"
        return None
    
    # == format output ==
    if (df is not None) and (append_df_suffix is not None):
        # TODO append results to df
        result_df.index = df.index.copy()
        df = df.join(result_df, rsuffix=append_df_suffix)
        return df
    else:
        return result_df
        
        
        
def run_nupack(seq_list, struct_list, sodium, model_param_file):
    pnames = ['dH', 'Tm', 'dG_37', 'dS']
    result_df = pd.DataFrame(index=np.arange(len(seq_list)), columns=pnames)
    
    for i in range(len(seq_list)):
        seq, struct = seq_list[i], struct_list[i]
        result_df.iloc[i,:] = util.get_nupack_dH_dS_Tm_dG_37(
            seq, struct, sodium=sodium, 
            return_dict=True, param_set=model_param_file)
    
    return result_df