
import numpy as np
import pandas as pd
import datetime

from nnn import simulation


np.random.seed(42)

if __name__ == "__main__":
    annotation_file = '/home/groups/wjg/kyx/array_analysis/data/reference/NNNlib2b_annotation_20220418.tsv'
    subset = False
    T = np.arange(5, 97.5, 2.5)
    sodium = 0.150
    n_jobs = 12
    series_file = '/scratch/groups/wjg/kyx/NNNlib2b_Nov11/data/series_simulated/nupack_150mM.CPseries.pkl'
    
    print(datetime.datetime.now(), 'Loading file')
    annotation_df = pd.read_table(annotation_file)
    if subset:
        annotation_df = annotation_df.sample(30)
        
    print(datetime.datetime.now(), 'File loaded\n\nSimulating curves')
    curves = simulation.simulate_CPseries(annotation_df, T=T, sodium=sodium, n_jobs=n_jobs)

    df = pd.DataFrame(curves, index=annotation_df.SEQID, columns=T)
    df.to_pickle(series_file)
    print(datetime.datetime.now(), 'Results saved to file.')