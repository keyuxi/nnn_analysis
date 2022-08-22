
import numpy as np
import pandas as pd
from datetime import datetime

from nnn import simulation


np.random.seed(42)

if __name__ == "__main__":
    subset = False
    n_jobs = 4
    series_file = '/scratch/groups/wjg/kyx/NNNlib2b_Nov11/data/series_simulated/nupack_75mM_probability_WCsubset.CPseries.pkl'
    fit_file = series_file.replace('.CPseries.pkl', '.CPvariant.pkl')
    
    print(datetime.now(), 'Loading file')
    series_df = pd.read_pickle(series_file)
    if subset:
        series_df = series_df.sample(25)
        
    print(datetime.now(), 'File loaded. Fitting simulated curves')
    df = simulation.fit_simulated_nupack_series(series_df, n_jobs=n_jobs)

    df.to_pickle(fit_file)
    print(datetime.now(), 'Results saved to file.')