import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import seaborn as sns
import colorcet as cc
import json, os, pickle
from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder
from ipynb.draw import draw_struct
import nupack
import sklearn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from nnn import util, fileio, processing, plotting, simulation, dG_fit, uv
import nnn.motif_fit as mf
from nnn.arraydata import ArrayData

# suppress warnings 
import warnings
warnings.filterwarnings("ignore")

from nnn.uv import *

sample_sheet_file = './data/uv_melt/ECLTables/ECLSampleSheet230317.csv'
datadir="./data/uv_melt/ECLExportAuto"
result_file='./data/uv_melt/uvmelt_230317.csv'
agg_result_file='./data/uv_melt/uvmelt_agg_230317.csv'

result_df = fit_all_manual_blank(datadir, sample_sheet_file, result_file=result_file)