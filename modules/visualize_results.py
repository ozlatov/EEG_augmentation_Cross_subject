import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import os

def get_result_plot_fnames(mode, clf_name, filters, nct, ds, n_subjects, N, lr, batch_size, weight_decay, shrinkage, optimizer):
    fname1 = 'nct='+str(nct)+'_ds='+str(ds)+'/'+mode+'/'
    if filters:
        raw19_or_filtered = 'filtered'
    else:
        raw19_or_filtered = 'raw19'
    if clf_name == 'nn' or clf_name == 'dnn':
        if mode == 'few_subjects':
            fname2 = clf_name+'/'+raw19_or_filtered+'_'+str(n_subjects)+'_subjects_N='+str(N)+'/'+optimizer+'_'+'lr='+str(lr)+'_bs='+str(batch_size)+'_wd='+str(weight_decay)+".p"
            plot_name = 'n='+str(n_subjects)+'_'+'k='+str(nct)+'_'+clf_name+'_'+raw19_or_filtered+'_'+str(n_subjects)+'_subjects_N='+str(10)+'_'+optimizer+'_'+'lr='+str(lr)+'_bs='+str(batch_size)+'_wd='+str(weight_decay)+".pdf"
        elif mode == 'few_trials':
            fname2 = clf_name+'/'+raw19_or_filtered+'_N='+str(N)+'/'+optimizer+'_'+'lr='+str(lr)+'_bs='+str(batch_size)+'_wd='+str(weight_decay)+".p"
            plot_name = 'k='+str(nct)+'_'+clf_name+'_'+raw19_or_filtered+'_N='+str(10)+'_'+optimizer+'_'+'lr='+str(lr)+'_bs='+str(batch_size)+'_wd='+str(weight_decay)+".pdf"
           
    if clf_name == 'lda':
        if mode == 'few_subjects':
            fname2 = clf_name+'/'+raw19_or_filtered+'_'+str(n_subjects)+'_subjects_N='+str(N)
            plot_name = 'n='+str(n_subjects)+'_'+'k='+str(nct)+'_'+clf_name+'_'+raw19_or_filtered+'_'+str(n_subjects)+'_subjects_N='+str(10)
        elif mode == 'few_trials':
            fname2 = clf_name+'/'+raw19_or_filtered+'_N='+str(N)
            plot_name = 'k='+str(nct)+'_'+clf_name+'_'+raw19_or_filtered+'_N='+str(10)
        if shrinkage:
            fname2 = fname2 + '_shrinkage'
            plot_name = plot_name + '_shrinkage'
        plot_name = plot_name +'.pdf'
            
    result_fname = 'results/'+fname1+ fname2
    result_fname = result_fname 
    plot_fname = 'new_plots/'+fname1+plot_name
    if not os.path.isdir('new_plots/'+fname1):
        os.makedirs('new_plots/'+fname1)
        print("The new directory for plots is created!")
    return result_fname, plot_fname

def visualize_results(mode, clf_name, filters, nct, ds, n_subjects, N, lr, batch_size, weight_decay, shrinkage, optimizer):

    result_fname, plot_fname = get_result_plot_fnames(mode, clf_name, filters, nct, ds, n_subjects, N, lr, batch_size, weight_decay, shrinkage, optimizer)
    
    file = open(result_fname,'rb')
    results = pickle.load(file)
    original_results = np.asarray(results['original'])
    augmented_results = np.asarray(results['augmented'])
    n_subs = original_results.shape[0]
    idx_sort = original_results.argsort()
    stacked_results = np.hstack((original_results[idx_sort],augmented_results[idx_sort]))
    
    if clf_name == 'lda':
        plot_title = 'LDA'
    if clf_name == 'nn':
        plot_title = 'ShallowNN'
    if clf_name == 'dnn':
        plot_title = 'DeepNN'

    N=10
    sns.set_style("darkgrid")
    acc = dict([['no', np.hstack((range(n_subs), range(n_subs)))],
                ['acc', stacked_results],
                ['label', np.hstack([['N=1' for _ in range(n_subs)], ['N={0:d}'.format(N) for _ in range(n_subs)]])]])
    #print(acc['acc'].shape)
    #print(acc['label'].shape)
    #print(acc)
    plt.figure(figsize=(6, 3.5))
    fig, (ax, ax2) = plt.subplots(1, 2)
    plt.sca(ax)
    sns.scatterplot(data=acc, x='no', y='acc', hue='label')
    ax.set_position([0.1, 0.15, 0.6, 0.75])
    ax.xaxis.grid(False)
    ax.legend(loc='lower right')
    plt.xticks(np.arange(n_subs))
    plt.xlabel('participant  [sorted index]')
    plt.ylim(50, 100)
    plt.yticks(np.arange(50, 100, 5))
    plt.ylabel("accuracy  [%]")
    plt.title(plot_title)
    ax2.set_position([0.75, 0.15, 0.2, 0.75])
    plt.sca(ax2)
    sns.boxplot(data=acc, x='label', y='acc', hue='label')
    plt.ylim(50, 100)
    plt.yticks(np.arange(50, 100, 5))
    ax2.set_yticklabels([])
    ax2.legend_ = None
    plt.ylabel('')
    plt.xlabel('')
    #plt.show()
    plt.savefig(plot_fname, bbox_inches='tight')
    print('saving visualized results')
