import torch
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import random
import seaborn as sns
import os
import itertools
import pickle
import numpy as np
from sys import argv

import modules.augmentation as aug
import modules.leadfields as lf
import modules.filters as flt
import modules.create_datasets as cd
import modules.classification as clf
import modules.classify as cfy
import modules.visualize_results as vis

n_calibration_trials = 100                  # how many originally measured trials per subject to use, <=150
N = 10                                      # N times more data after augmentation
n_train_subj = 5                            # in case of 'few_subjects' mode, number of subjects to use for data augmentation, False otherwise


optimizer = 'sgd'
lr = 0.005 
batch_size = 32
weight_decay = 0.1
epochs = 100
                     
#set these parameters------------------------------------------------------------------------------------------------------------------
classifier = 'dnn'                          # 'lda', 'nn' or 'dnn'
mode = 'few_subjects'                       # 'few_subjects' or 'few_trials'
filters = True                              # True or False

#LDA parameters
shrinkage = False                           # True or False

#--------------------------------------------------------------------------------------------------------------------------------------
#augmentation
names_used, common_clab, leadfields, gridnorms, gridpos = lf.load_leadfields()  # load subjects names and leadfields
N_all = 21                                                                      # how many times augment the data results in N_all times bigger dataset)   
dipole_shift = 1e-3 

dirname = 'generated_data/nct='+str(n_calibration_trials)+'_ds='+str(dipole_shift)+'/'
if not os.path.isdir(dirname):
    os.makedirs(dirname)
    print("The new directory is created!")
    aug.augment_data(dirname, names_used, N_all, n_calibration_trials,
                     dipole_shift, common_clab, leadfields, gridnorms, gridpos)
    flt.bp_laplacian_filter(dirname, names_used, common_clab)
    flt.raw_19_channel_data(dirname, names_used, common_clab)

#select all the data needed for the experiments: original calibration and generated trials for all subjects
if filters:
    dirname = dirname+'filtered_'
else:
    dirname = dirname+'raw19_'
X_train_all, y_train_all = cd.create_train_dataset(dirname, names_used)

#--------------------------------------------------------------------------------------------------------------------------------------
#directory for saving classification results
res_dir = 'results/nct='+str(n_calibration_trials)+'_ds='+str(dipole_shift)+'/'+mode+'/'+classifier+'/'
if not os.path.isdir(res_dir):
    os.makedirs(res_dir)

#classification
if classifier == 'lda':
    cfy.lda_classification(mode, n_train_subj, N, n_calibration_trials, X_train_all, y_train_all,
                   shrinkage, N_all, names_used, common_clab, filters, res_dir)                     #dirname = res_dir - directory to save the results
elif classifier == 'nn' or classifier == 'dnn':
    cfy.nn_classification(classifier, mode, n_train_subj, N, n_calibration_trials, X_train_all, y_train_all,
                          N_all, names_used, common_clab, filters, res_dir, lr, batch_size, weight_decay, epochs, optimizer)
#visualize the results    
vis.visualize_results(mode, classifier, filters, n_calibration_trials, dipole_shift,
                      n_train_subj, N, lr, batch_size, weight_decay, shrinkage, optimizer) #plot results



    

