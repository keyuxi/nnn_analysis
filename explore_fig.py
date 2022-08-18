"""
Run-able versions of important parts of the jupyter notebooks
"""
# import stuffs
exec(open("./nnn/ipynb_init.py").read())

def plot_triloop_linear_model(param='dG_37'):
    annotation_file = './data/annotation/NNNlib2b_annotation_20220519.tsv'
    replicate_df = pd.read_table('./data/nnnlib2b_replicates.tsv')

    arraydata = ArrayData(replicate_df=replicate_df.iloc[:4,:],
                        annotation_file=annotation_file)
    
    myfilter = "dH_err_rel < 0.2 & Tm_err_abs < 2 & redchi < 1.5 & n_inlier > 10"
    two_state_df = arraydata.filter_two_state(min_rep_pass=2, force_recalculate=True, myfilter=myfilter)
    
    arr = arraydata.data.join(arraydata.annotation)
    arr['scaffold'] =arr.apply(lambda row: f'{row.bottomScaffold}_{row.topScaffold}', axis=1)
    
    # First get the WC model lr_sym
    df = arr.query('Series == "WatsonCrick" & two_state == True')
    df = df.query("dG_37_se < .3 & Tm < 55 & Tm > 30")

    feats = mf.get_feature_count_matrix(df, feature_method='get_stack_feature_list_simple_loop', 
                                        fit_intercept=True, symmetry=True)
    y = df[param]
    y_err = df[param+'_se']
    X_train, X_test, y_train, y_test, yerr_train, yerr_test = sklearn.model_selection.train_test_split(feats.values, y.values, y_err.values, 
                                                                                                    test_size=0.25, random_state=42, 
                                                                                                    stratify=df.ConstructType)

    lr_sym = util.LinearRegressionSVD()
    lr_sym.fit(X_train, y_train, yerr_train, feature_names=feats.columns.tolist())
    
    # get the triloop data to play with
    df = arr.query('Series == "TRIloop" & two_state == 1')
    df = df.query("dG_37_se < .3 & Tm < 55 & Tm > 30")
    y = df[param]
    y_err = df[param+'_se']
    
    # make a model for vanilla SantaLucia parameters
    # X_train and y_train are only for fitting the intercept
    loop_dG_37 = 3.5
    feats = mf.get_feature_count_matrix(df, feature_method='get_stack_feature_list_simple_loop', 
                                        loop_base_size=0, fit_intercept=True, symmetry=True)
    X_train_intercept, X_test_intercept, y_train, _, yerr_train, yerr_test, _, indices_test = sklearn.model_selection.train_test_split(feats.values, y.values, y_err.values, np.arange(len(y)),
                                                                                                    test_size=0.25, random_state=42, 
                                                                                                    stratify=df.Series)
    santalucia_new = pd.read_table('./data/literature/SantaLucia_flipped.csv')
    santalucia_loop = pd.DataFrame(index=feats.columns).join(santalucia_new.set_index('motif')[[param]])
    santalucia_loop.fillna(loop_dG_37, inplace=True)
    sl_hairpin = util.LinearRegressionSVD(param=param)
    sl_hairpin.set_coef(feature_names=feats.columns, coef_df=santalucia_loop, index_col='index')
    sl_hairpin.fit_intercept_only(X_train_intercept, y_train)
    
    # Fit a triloop model from data
    # Fix the WC parameters from fitted
    # not using the intercept fitted from the previous model
    feats = mf.get_feature_count_matrix(df, feature_method='get_stack_feature_list_simple_loop', 
                                    loop_base_size=0, fit_intercept=False, symmetry=True)
    X_train, X_test, y_train, y_test, yerr_train, yerr_test, indices_train, indices_test = sklearn.model_selection.train_test_split(feats.values, y.values, y_err.values, np.arange(len(y)),
                                                                                                   test_size=0.25, random_state=42, stratify=df.Series)    
    coef_df = lr_sym.coef_df.iloc[:-1,:]
    coef_se_df = lr_sym.coef_se_df.iloc[:-1,:]

    hairpin_model = util.LinearRegressionSVD(param=param)
    hairpin_model.fit_with_some_coef_fixed(X_train, y_train, yerr_train, 
                                        feature_names=feats.columns.tolist(), fixed_feature_names=coef_df.index,
                                        coef_df=coef_df, coef_se_df=coef_se_df, debug=True)

    
    # Fix the WC parameters from SantaLucia
    hairpin_model_sl = util.LinearRegressionSVD(param=param)
    hairpin_model_sl.fit_with_some_coef_fixed(X_train, y_train, yerr_train, 
                                        feature_names=feats.columns.tolist(), fixed_feature_names=santalucia_new.motif.tolist(),
                                        coef_df=santalucia_new.set_index('motif')[[param]], debug=True)

    # Plot the results on 2-state test set
    fig, ax = plt.subplots(1,4,figsize=(16,4))
    if param == 'dG_37':
        lim = [-3, 1.5]
    elif param == 'dH':
        lim = [-50, 0]
        # lim = [-60, -10]
    elif param == 'dS':
        lim = [-.2,0]
        
    plotting.plot_truth_predict(hairpin_model, X_test, y_test, yerr_test, ax=ax[0], lim=lim, title='MANifold')
    plotting.plot_truth_predict(hairpin_model_sl, X_test, y_test, yerr_test, ax=ax[1], lim=lim, title='MANifold hairpin param + SantaLucia WC param')
    plotting.plot_truth_predict(sl_hairpin, X_test_intercept, y_test, yerr_test, ax=ax[2], lim=lim, title='SantaLucia')

    df_test = df.iloc[indices_test,:]
    plt.errorbar(x=df_test[param], y=df.iloc[indices_test,:][param+'_NUPACK_salt_corrected'] + 1.1, fmt='k.', alpha=.3)
    ax[3].set_xlim(lim)
    ax[3].set_ylim(lim)
    ax[3].plot(lim, lim, '--', c='gray')
    ax[3].set_title(r'$\bf{Adjusted\ NUPACK}$' + '\n(SantaLucia with stacking and adjustments)\n' + '$R^2$ = %.2f, corr = %.2f' % 
                    (r2_score(df_test[param], df_test[param+'_NUPACK_salt_corrected']),
                    pearsonr(df_test[param], df_test[param+'_NUPACK_salt_corrected'])[0]))
    ax[3].set_xlabel('measurement (kcal/mol)')
    util.save_fig('./fig/motif_fit/triloop_models_test_set_%s.pdf'%param)