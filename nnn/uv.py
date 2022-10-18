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

import warnings
warnings.filterwarnings("ignore")

def absolute_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def get_dG(dH, Tm, celsius):
    return dH * (1 - (273.15 + celsius) / (273.15 + Tm))

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
def residual(pars, x, data):
    # define the function
    dH, Tm, fmax, fmin, slope = pars['dH'], pars['Tm'], pars['fmax'], pars['fmin'], pars['slope']
    model = fmin + slope * x + ((fmax - fmin)/(1 + np.exp(dH /0.0019872 * ((Tm + 273.15)**(-1) - (x + 273.15)**(-1)))))
    return model - data

def fit_param_direct(curve, celsius_min=5, celsius_max=95, smooth=True):
    pfit = Parameters()
    data_max = np.max(curve.absorbance)
    data_min = np.min(curve.absorbance)
    pfit.add(name='dH', value=-20)
    pfit.add(name='Tm', value=35)
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
    sns.despine()
    
    return out


### Use the d_absorbance method ###
def fit_param_d_absorbance(curve, out, celsius_min=5, celsius_max=95, smooth=True):

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
    sns.despine()
    
    result = {}
    result['Tm'] = x[peak]
    result['dH'] = - d_p_unfold[peak] * 4 * 0.0019872 * (result['Tm'] + 273.15)**2
    result['dS'] = result['dH'] / (result['Tm'] + 273.15)
    result['dG_37'] = get_dG(result['dH'], result['Tm'], 37)
    return result

### Master Function ###
def fit_curve(fn, figdir='', **kwargs):
    try:
        curve = read_curve(fn)
        curve_name = parse_curve_name(fn)
        print(curve_name['curve_str'])
        out = fit_param_direct(curve, **kwargs)
        save_fig(os.path.join(figdir, curve_name['curve_date'], f"{curve_name['curve_num']}_{curve_name['curve_name']}_direct_fit.png"))
        result = fit_param_d_absorbance(curve, out, **kwargs)
        save_fig(os.path.join(figdir, curve_name['curve_date'], f"{curve_name['curve_num']}_{curve_name['curve_name']}_d_p_unfold.png"))
        result_dict = combine_results(out, result)
        result_dict.update(kwargs)
        result_dict.update(curve_name)
        return result_dict
    except:
        print("Trouble with", fn)
        return dict()