
import numpy as np
import pandas as pd
from datetime import datetime

from nnn import simulation


np.random.seed(42)

if __name__ == "__main__":
    #### Settings ####
    annotation_file = '/home/groups/wjg/kyx/array_analysis/data/reference/NNNlib2b_annotation_20220418.tsv'
    subset = True
    T = np.arange(5, 97.5, 2.5)
    sodium = 0.075
    n_jobs = 6
    series_file = '/scratch/groups/wjg/kyx/NNNlib2b_Nov11/data/series_simulated/nupack_75mM_probability_2kT_WCsubset.CPseries.pkl'
    
    
    ####---- Simulation ----####
    
    print(datetime.now(), 'Loading file')
    annotation_df = pd.read_table(annotation_file)
    if subset:
        annotation_df = annotation_df.query('Series == "WatsonCrick"').sample(1000)
        
    print(datetime.now(), 'File loaded\n\nSimulating curves')
    curves = simulation.simulate_CPseries(annotation_df, T=T, sodium=sodium, n_jobs=n_jobs, dG_gap_kT=8.0)

    df = pd.DataFrame(curves, index=annotation_df.SEQID, columns=T)
    df.to_pickle(series_file)
    print(datetime.now(), 'Simulation results saved to file.')
    
    
    ####---- Fitting ----####
    
    fit_file = series_file.replace('.CPseries.pkl', '.CPvariant.pkl')
    
    print(datetime.now(), 'Loading file')
    df = pd.read_pickle(series_file)
        
    print(datetime.now(), 'File loaded. Fitting simulated curves')
    fit_df = simulation.fit_simulated_nupack_series(df, n_jobs=n_jobs)

    fit_df.to_pickle(fit_file)
    print(datetime.now(), 'Fitted CPvariant results saved to file.')