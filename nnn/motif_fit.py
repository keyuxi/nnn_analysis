import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')
from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder

from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from util import *
import feature_list


def search_GU_pairs(seq):
    GU_counter=0
    for i, char in enumerate(seq[:7]):

        if char=='G':
            if seq[-1-i]=='T':
                GU_counter+=1
        elif char=='T':
            if seq[-1-i]=='G':
                GU_counter+=1
    return GU_counter


def get_model_metric(y, y_err, preds, n_feature):
    n = len(y)
    m = {}
    rss = np.sum(np.square(y - preds))
    m['bic'] = n*np.log(rss/n) + n_feature*np.log(n) 
    m['aic'] = n*np.log(rss/n) + 2*n_feature
    m['rmse'] = np.sqrt(np.mean(np.square(y - preds)))
    m['chi2'] = np.sqrt(np.mean(np.square(y - preds)/np.square(y_err)))

    return m

def get_motif_se(X, res):
    """
    Calculate heteroskedasticity robust standard errors for fitted parameters
    """
    XT_X_1 = np.linalg.inv(X.T @ X)
    var_coef = XT_X_1 @ X.T @ np.diag(res**2) @ X @ XT_X_1
    return np.sqrt(np.diag(var_coef))

def get_feature_count_matrix(df, feature_method='get_stack_feature_list', stack_size=2):
        """
        Args:
            df - (n_variant, N) df
        Returns:
            feats - a (n_variant, n_feature) feature df
            n_feature - int
        """
        df['feature_list'] = df.apply(lambda row: getattr(feature_list, feature_method)(row, stack_size=stack_size), axis=1)
        
        cv = CountVectorizer()
        feats = pd.DataFrame.sparse.from_spmatrix(cv.fit_transform([' '.join(x) for x in df['feature_list']]),
                        index=df.index, columns=[x.upper() for x in cv.get_feature_names()])
        
        # Remove features that every construct contains
        for k in feats.keys():
            if len(feats[k].unique())==1:
                feats = feats.drop(columns=[k])
                
        n_feature = len(feats.keys())

        return feats, n_feature


def fit_linear_motifs(df, feature_method='get_stack_feature_list',
                      param='dG_37', err='_se_corrected', lim=None,
                      stack_sizes=[1,2,3], fit_intercept=False):

    fig, ax = plt.subplots(1, 3, figsize=(9,4), sharey=True)

    N_SPLITS = 5
    y = df[param]
    y_err = df[param + err]

    titles = {1:'N_N features (stacks only)', 2:'NN_NN (Nearest neighbor)', 3:'N(3)_N(3)', 4:'N(4)_N(4)', 5:'N(5)_N(5)'}

    coef_dfs=[]
    for i, stack_size in enumerate(stack_sizes):
        
        feats, n_feature = get_feature_count_matrix(df, feature_method=feature_method, stack_size=stack_size)
        
        #Perform linear regression fit
        mdl = Ridge(fit_intercept=fit_intercept)

        X = feats.values

        results = cross_validate(mdl, X, y, cv=N_SPLITS, return_estimator=True)
        coef_df = pd.DataFrame()
        for x in results['estimator']:
            for j in range(len(feats.columns)):
                coef_df = coef_df.append({'motif': feats.columns[j], param: x.coef_[j]}, ignore_index=True)
        coef_dfs.append(coef_df)

        preds = cross_val_predict(mdl, X, y, cv=N_SPLITS)
        
        df['tmp_pred'] = preds
        m = get_model_metric(y, y_err, preds, n_feature)


        N = X.shape[0]
        p = X.shape[1] + 1  # plus one because LinearRegression adds an intercept term

        X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
        X_with_intercept[:, 0] = 1
        X_with_intercept[:, 1:p] = X

        beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y.values
        
        residuals = y.values - preds
        residual_sum_of_squares = residuals.T @ residuals
        sigma_squared_hat = residual_sum_of_squares / (N - p)
        var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
        for p_ in range(p):
            standard_error = var_beta_hat[p_, p_] ** 0.5
            print(f"SE(beta_hat[{p_}]): {standard_error}")

        plt.subplot(1,3,stack_size)
        #errorbar(y, preds, xerr=y_err, fmt='.', alpha=0.1,zorder=0, color='k')
        hue_order = ['WC_5ntstem', 'WC_6ntstem', 'WC_7ntstem']
        sns.scatterplot(x=param, y='tmp_pred', data=df, hue='ConstructType', hue_order=hue_order, linewidth=0, s=10, alpha=0.6, palette='plasma')
        plt.xlabel('Fit '+param)
        plt.ylabel('CV-test-split predicted '+param)
        plt.title("%s, %d features\n RMSE: %.2f, $\chi^2$: %.2f \n BIC: %.2f" % (titles[stack_size], n_feature, m['rmse'], m['chi2'], m['bic']))
        
        if lim is not None:
            plt.plot(lim,lim,'--',color='grey',zorder=0)
            plt.xlim(lim)
            plt.ylim(lim)
        if i!=0: plt.legend([],frameon=False)

        plt.tight_layout()

    return coef_dfs, feats, preds, results


def compare_fit_with_santalucia(df, santa_lucia, params=['dH', 'dS', 'dG_37']):

    for param in params:
        coef_dfs, _, _ = fit_linear_motifs(df, param=param, err='_se', stack_sizes=[1,2,3])
        fit_param = pd.DataFrame(coef_dfs[1].groupby('motif').apply(np.nanmean), columns=[param]).join(
             pd.DataFrame(coef_dfs[1].groupby('motif').apply(lambda x: np.nanstd(x)/np.sqrt(5)), columns=[param+'_cv_se']))
        santa_lucia = santa_lucia.merge(fit_param, on='motif', how='inner', suffixes=('_SantaLucia', '_MANIfold'))

    return santa_lucia