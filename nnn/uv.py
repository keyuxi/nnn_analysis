"""
Functions for UV melting analysis
"""
import numpy as np
import pandas as pd
from scipy import stats
import os, json
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from lmfit import minimize, Minimizer, Parameters, report_fit
from scipy.interpolate import interp1d
from scipy import signal
from .util import *

import warnings
warnings.filterwarnings("ignore")

def lookup_sample_df(df, df_ref, key):
    # looks up `key` in `df_ref`
    return df.apply(lambda row: df_ref.query("curve_date == '%s' & curve_num == '%s'" % (row['curve_date'], row['curve_num']))[key].values[0], axis=1)

def read_curve(fn):
    curve = pd.read_csv(fn, header=None)
    curve.columns = ['celsius', 'absorbance']
    curve.sort_values(by='celsius', inplace=True)
    return curve

def parse_curve_name(fn):
    curve_date = fn.split('/')[-2].split('_')[0]
    curve_num = fn.split('/')[-1].split('_')[0]
    curve_name = fn.split('/')[-1].split('_')[1].split('.csv')[0]
    curve_str = f'{curve_date}_{curve_num}_{curve_name}'
    return dict(curve_date=curve_date, curve_num=curve_num, 
                curve_name=curve_name, curve_str=curve_str)

def combine_results(out, result):
    result_dict = {}
    for p in ('dH', 'Tm', 'fmax', 'fmin', 'slope'):
        result_dict[p+'_fit'] = out.params[p].value
        result_dict[p+'_fit_std'] = out.params[p].stderr
    result_dict.update(result)
    result_dict['rmse_fit'] = np.sqrt(np.mean(np.square(out.residual)))
    return result_dict

def save_fig(filename, fig=None):

    figdir, _ = os.path.split(filename)
    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    if fig is None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        
def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

### Directly fit the curves ###
def curve_model(x, dH, Tm, fmin, fmax, slope):
    # define the function
    return fmin + slope * x + ((fmax - fmin)/(1 + np.exp(dH /0.0019872 * ((Tm + 273.15)**(-1) - (x + 273.15)**(-1)))))

def residual(pars, x, data):
    dH, Tm, fmax, fmin, slope = pars['dH'], pars['Tm'], pars['fmax'], pars['fmin'], pars['slope']
    model = curve_model(x, dH, Tm, fmin, fmax, slope)
    return model - data

def fit_param_direct(curve, celsius_min=5, celsius_max=95, smooth=True, plot_title=''):
    pfit = Parameters()
    data_max = np.max(curve.absorbance)
    data_min = np.min(curve.absorbance)
    pfit.add(name='dH', value=-20)
    pfit.add(name='Tm', value=(celsius_max + celsius_min) * 0.5)
    pfit.add(name='fmax', value=2*data_max, min=data_min, max=20*data_max)
    pfit.add(name='fmin', value=0.2*data_min, min=0, max=data_max)
    pfit.add(name='slope', value = 1e-5, max=5.0, min=-5.0)

    curve_used = curve.query(f'celsius >= {celsius_min} & celsius <= {celsius_max}')
    if smooth:
        curve_used.loc[:,'absorbance'] = signal.savgol_filter(curve_used.loc[:,'absorbance'], 9, 3)

    out = minimize(residual, pfit, args=(curve_used.celsius,), 
                   kws={'data': curve_used.absorbance})
    best_fit = curve_used.absorbance + out.residual
    #report_fit(out.params)
    
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(curve.celsius, curve.absorbance, '+', c='purple')
    ax.plot(curve_used.celsius, best_fit, 'orange', linewidth=2.5)
    ax.set_xlabel('temperature (°C)')
    ax.set_ylabel(r'absorbance')
    ax.set_title(plot_title)
    sns.despine()
    
    return out


### Use the d_absorbance method ###
def fit_param_d_absorbance(curve, out, celsius_min=5, celsius_max=95, smooth=True, plot_title=''):

    curve_used = curve.query(f'celsius >= {celsius_min} & celsius <= {celsius_max}').sort_values(by='celsius')
    x = np.arange(curve_used.celsius.iloc[0], curve_used.celsius.iloc[-1], 0.1)
    signal_used = signal.savgol_filter(curve_used.absorbance, 9, 3)
    signal_used = (signal_used - out.params['fmin']) / (out.params['fmax'] - out.params['fmin'])
    f = interp1d(curve_used.celsius, 
                 signal_used, 
                 kind='cubic')
    d_p_unfold = np.diff(f(x)) * 10
    if smooth:
        d_p_unfold = signal.savgol_filter(d_p_unfold, 100, 3)
    peaks = signal.find_peaks(d_p_unfold)[0]
    ind_max = np.argmax(d_p_unfold[peaks])
    peak = peaks[ind_max]
    
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(x[:-1], d_p_unfold, 'k')
    ax.axvline(x=x[peak], ls='--', c='gray')
    ax.set_xlabel('temperature (°C)')
    ax.set_ylabel(r"$p_{unfold}'$")
    ax.set_title(plot_title)
    sns.despine()
    
    result = {}
    result['Tm_diff'] = x[peak]
    result['dH_diff'] = - d_p_unfold[peak] * 4 * 0.0019872 * (result['Tm_diff'] + 273.15)**2
    result['dS_diff'] = result['dH_diff'] / (result['Tm_diff'] + 273.15)
    result['dG_37_diff'] = get_dG(result['dH_diff'], result['Tm_diff'], 37)
    return result

### Master Function ###
def fit_curve(fn, figdir='', **kwargs):
    try:
        curve = read_curve(fn)
        curve_name = parse_curve_name(fn)
        print(curve_name['curve_str'])
        out = fit_param_direct(curve, plot_title=curve_name['curve_str'], **kwargs)
        save_fig(os.path.join(figdir, curve_name['curve_date'], f"{curve_name['curve_num']}_{curve_name['curve_name']}_direct_fit.png"))
        result = fit_param_d_absorbance(curve, out, plot_title=curve_name['curve_str'], **kwargs)
        save_fig(os.path.join(figdir, curve_name['curve_date'], f"{curve_name['curve_num']}_{curve_name['curve_name']}_d_p_unfold.png"))
        result_dict = combine_results(out, result)
        result_dict.update(kwargs)
        result_dict.update(curve_name)
        print('\tDone!')
        return result_dict
    except:
        print("Trouble with", fn)
        return dict()
    
    
### Main ###
def fit_all():
    datadir = '/mnt/d/data/nnn/ECLExport'
    data_list = [fn for fn in absolute_file_paths(datadir) if fn.endswith('.csv')]
    sample_df = pd.read_csv('/mnt/d/data/nnn/UVMeltingSampleSheet.csv', index_col=0)
    
    result_columns = ['curve_date', 'curve_num', 'curve_name',
                  'dH_fit', 'dH_fit_std', 'Tm_fit', 'Tm_fit_std', 
                  'fmax_fit', 'fmax_fit_std', 'fmin_fit', 'fmin_fit_std', 
                  'slope_fit', 'slope_fit_std', 'rmse_fit',
                  'Tm', 'dH', 'dS', 'dG_37', 
                  'celsius_min', 'celsius_max']

    result_df = pd.DataFrame(index=[parse_curve_name(x)['curve_str'] for x in data_list], columns=result_columns)

    for fn in data_list:
        curve_name = parse_curve_name(fn)
        row = sample_df.query("curve_date == '%s' & curve_num == '%s'" % (curve_name['curve_date'], curve_name['curve_num']))
        
        if row.shape[0] == 0:
            print('Cannot find %s in the sample sheet' % curve_name['curve_str'])
        else:
            result_dict = fit_curve(fn, figdir='/mnt/d/data/nnn/fig', 
                                    celsius_min=row.at[row.index[0],'celsius_min'],
                                    celsius_max=row.at[row.index[0],'celsius_max'])

            result_df.loc[curve_name['curve_str'], :] = result_dict
    result_df.to_csv('/mnt/d/data/nnn/uvmelt.csv')