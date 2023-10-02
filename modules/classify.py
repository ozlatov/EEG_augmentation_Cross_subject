import modules.create_datasets as cd
import modules.classification as clf
from modules.shallow_network import ShallowNetwork
from modules.deep_network import DeepNetwork
import modules.train_test_functions as tt

import numpy as np
import pickle
import os

import torch
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import random

def lda_classification(mode, n_train_subj, N, n_calibration_trials, X_train_all, y_train_all,
                       shrinkage, N_all, names_used, common_clab, filters, dirname):
    '''  
    perform multisubject leave-one-out LDA (shrinkege if True) classification
    for the original and generated data from X_train_all and y_train_all
    use only N times bigger dataset than original (incrrased by augmentation), N <= N_all
    filters = True to apply bp and Laplacian filters
    
    save original and augmented classification results for each subject
    '''
    n_subj = len(names_used)
    lda_acc_orig = []
    lda_acc = []
    
    for i, name_id in enumerate(names_used):

        X_train, y_train = cd.select_aug_train(mode, n_train_subj, i, n_calibration_trials, N, X_train_all, y_train_all, N_all, n_subj)
        X_test, y_test = cd.create_test_dataset(name_id, filters, common_clab)
        fv_train, fv_test = np.log(np.var(X_train.T,axis=0)), np.log(np.var(X_test.T,axis=0))
        if shrinkage:
            w, b = clf.train_LDAshrink(fv_train, y_train)
        else:
            w, b = clf.train_LDA(fv_train, y_train)
        out = w.T.dot(fv_test) - b
        lda_acc.append(100-clf.loss_weighted_error(out, y_test))

        X_train, y_train = cd.select_orig_train(mode, n_train_subj, i, n_calibration_trials, X_train_all, y_train_all, N_all, n_subj)
        X_test, y_test = cd.create_test_dataset(name_id, filters, common_clab)
        fv_train, fv_test = np.log(np.var(X_train.T,axis=0)), np.log(np.var(X_test.T,axis=0))
        if shrinkage:
            w, b = clf.train_LDAshrink(fv_train, y_train)
        else:
            w, b = clf.train_LDA(fv_train, y_train)
        out = w.T.dot(fv_test) - b
        lda_acc_orig.append(100-clf.loss_weighted_error(out, y_test))

        if i in [1,5,12,16]:
            print(str(np.round(100*i/18, 0))+'% is done')

    results ={
            "original": lda_acc_orig,
            "augmented": lda_acc
            }
    if filters:
        dirname = dirname + 'filtered_'
    else:
        dirname = dirname + 'raw19_'
    if mode == 'few_subjects':
        dirname = dirname + str(n_train_subj) + '_subjects_'
    fname = dirname + 'N='+str(N)
    if shrinkage:
        fname = fname + '_shrinkage'
    file = open(fname, 'wb')
    pickle.dump(results, file)
    file.close()

def nn_classification(classifier, mode, n_train_subj, N, n_calibration_trials, X_train_all, y_train_all,
                      N_all, names_used, common_clab, filters, dirname, lr, batch_size, weight_decay, epochs, optimizer_name):
    '''
    perform multisubject leave-one-out NN classification
    for the original and generated data from X_train_all and y_train_all
    use only N times bigger dataset than original (incriesed by augmentation), N <= N_all
    filters = True to apply bp and Laplacian filters

    mode = 'few_subjects' if less then 17 subjects are used or 'few_trials'
    n_train_subj is needed only for this mode
    
    save original and augmented classification results for each subject
    '''
    if filters:
        dirname = dirname + 'filtered_'
    else:
        dirname = dirname + 'raw19_'
    if mode == 'few_subjects':
        dirname = dirname + str(n_train_subj) + '_subjects_'
    dirname = dirname + 'N='+str(N)+'/'
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    dirname = dirname+optimizer_name+'_'

    #directory for detailed results
    subdir = dirname+'lr='+str(lr)+'_bs='+str(batch_size)+'_wd='+str(weight_decay)+'/'
    if not os.path.isdir(subdir):
        os.makedirs(subdir)

    n_subj = len(names_used)
    nn_acc_orig = []
    nn_acc = []
    
    for i, name_id in enumerate(names_used):

        #Augmented
        X_train, y_train = cd.select_aug_train(mode, n_train_subj, i, n_calibration_trials, N, X_train_all, y_train_all, N_all, n_subj)
        X_test, y_test = cd.create_test_dataset(name_id, filters, common_clab)

        tensor_x_train, tensor_y_train = torch.Tensor(X_train), torch.LongTensor(y_train)  # transform to torch tensor
        tensor_x_test, tensor_y_test  = torch.Tensor(X_test), torch.LongTensor(y_test)
        training_data = TensorDataset(tensor_x_train, tensor_y_train ) # create train datset
        test_data = TensorDataset(tensor_x_test, tensor_y_test) # create test dataset
        train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(device))
        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it

        seed = 20200220  # random seed to make results reproducible
        random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        if classifier == 'nn':
            model = ShallowNetwork().to(device)
        if classifier == 'dnn':
            model = DeepNetwork().to(device)
        loss_fn = nn.NLLLoss()

        if optimizer_name == 'adam':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) #ADAMw!
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) #SGD!

        data_accuracy = []
        data_loss = []
        train_accuracy = []
        train_loss = []
        for t in range(epochs):
            tt.train(train_dataloader, model, loss_fn, optimizer, device=device)
            tt.test(test_dataloader, model, loss_fn, device=device, data_accuracy=data_accuracy, data_loss=data_loss)
            tt.test(train_dataloader, model, loss_fn, device=device, data_accuracy=train_accuracy, data_loss=train_loss)
        nn_acc.append(data_accuracy)

        #Original
        X_train, y_train = cd.select_orig_train(mode, n_train_subj, i, n_calibration_trials, X_train_all, y_train_all, N_all, n_subj)
        X_test, y_test = cd.create_test_dataset(name_id, filters, common_clab)

        tensor_x_train, tensor_y_train = torch.Tensor(X_train), torch.LongTensor(y_train)  # transform to torch tensor
        tensor_x_test, tensor_y_test  = torch.Tensor(X_test), torch.LongTensor(y_test)
        training_data = TensorDataset(tensor_x_train, tensor_y_train ) # create train datset
        test_data = TensorDataset(tensor_x_test, tensor_y_test) # create test dataset
        train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(device))
        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it

        seed = 20200220  # random seed to make results reproducible
        random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


        if classifier == 'nn':
            model = ShallowNetwork().to(device)
        if classifier == 'dnn':
            model = DeepNetwork().to(device)
        loss_fn = nn.NLLLoss()

        if optimizer_name == 'adam':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) #ADAMw!weight decay is 0.01 by default if not specified!
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) #SGD!

        data_accuracy_orig = []
        data_loss_orig = []
        train_accuracy_orig = []
        train_loss_orig = []
        for t in range(epochs):
            tt.train(train_dataloader, model, loss_fn, optimizer, device=device)
            tt.test(test_dataloader, model, loss_fn, device=device, data_accuracy=data_accuracy_orig, data_loss=data_loss_orig)
            tt.test(train_dataloader, model, loss_fn, device=device, data_accuracy=train_accuracy_orig, data_loss=train_loss_orig)
        nn_acc_orig.append(data_accuracy_orig)


        
        sub_results = {'data_accuracy': data_accuracy, #these are lists, with the lengths = epochs, so we have the results after each epoch to track learning procedure
                    'data_loss': data_loss,
                    'train_accuracy': train_accuracy,
                    'train_loss': train_loss,
                    'data_accuracy_orig': data_accuracy_orig,
                    'data_loss_orig': data_loss_orig,
                    'train_accuracy_orig': train_accuracy_orig,
                    'train_loss_orig': train_loss_orig}
        pickle.dump(sub_results, open(subdir+"subject_"+str(i)+".p", "wb" ))


        
    results ={
            "original": nn_acc_orig,
            "augmented": nn_acc
            }

    fname = dirname+'lr='+str(lr)+'_bs='+str(batch_size)+'_wd='+str(weight_decay)+".p"

    file = open(fname, 'wb')
    pickle.dump(results, file)
    file.close()
