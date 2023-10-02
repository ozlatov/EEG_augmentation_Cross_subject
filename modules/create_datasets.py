import numpy as np
import scipy as sp
import modules.my_modules as mm
import pickle


def select_aug_train(mode, n_train_subj, sub_i, n_calibration_trials, N, X_train_all, y_train_all, N_all, n_subj):
    '''
    from all generated data
    select the data for train for particular sub_number and particular N,
    where N shows how many times bigger the dataset becomes after augmentation (N <= N_all)
    
    mode = 'few_subjects' (for mode with <18 subjects) or 'few_trials' for 18 subjects with less trials
    n_train_subj is needed only for this mode
    '''
    mask = np.zeros(n_calibration_trials*N_all).astype(np.bool)
    mask[0:n_calibration_trials*N] = True
    mask = np.tile(mask,n_subj)
    mask[sub_i*n_calibration_trials*N_all:(sub_i+1)*n_calibration_trials*N_all] = False

    # if 'few_subjects' mode then select randomly only n_train_subj subjects for train:
    if mode == 'few_subjects':
        sub_candidates = np.delete(np.asarray(range(18)),sub_i)
        np.random.seed(44*sub_i)
        train_subjects = np.sort(np.random.choice(sub_candidates, n_train_subj, replace=False))
        for k in range(18):
            if k not in train_subjects:
                mask[k*n_calibration_trials*N_all:(k+1)*n_calibration_trials*N_all] = False
                
    X_train, y_train = X_train_all[mask], y_train_all[mask]
    return X_train, y_train

def select_orig_train(mode, n_train_subj, sub_i, n_calibration_trials, X_train_all, y_train_all, N_all, n_subj):
    '''
    from all generated data
    select the original data for train for particular sub_number,
    the resulted data contains only original trials, n_calibration_trials is the number of trials

    
    mode = 'few_subjects' (for mode with <18 subjects) or 'few_trials' for 18 subjects with less trials
    n_train_subj is needed only for this mode
    '''
    mask = np.zeros(n_calibration_trials*N_all).astype(np.bool)
    mask[0:n_calibration_trials] = True
    mask = np.tile(mask,n_subj)
    mask[sub_i*n_calibration_trials*N_all:(sub_i+1)*n_calibration_trials*N_all] = False

   # if 'few_subjects' mode then select randomly only n_train_subj subjects for train:
    if mode == 'few_subjects':
        sub_candidates = np.delete(np.asarray(range(18)),sub_i)
        np.random.seed(44*sub_i)
        train_subjects = np.sort(np.random.choice(sub_candidates, n_train_subj, replace=False))
        for k in range(18):
            if k not in train_subjects:
                mask[k*n_calibration_trials*N_all:(k+1)*n_calibration_trials*N_all] = False

    X_train, y_train = X_train_all[mask], y_train_all[mask]
    return X_train, y_train


def create_train_dataset(dirname, names_used):
    '''
    for each subject in names_used select trials for left and right hand classes
    epoch the data and concatenate (include all original and generated trials for each subject)

    X    resulted data:   n_calibration_trials*N*n_subjects x n_channels x trial_length
    y    resulted_labels: n_calibration_trials*N*n_subjects
    '''
    ns = 0
    for name_id in names_used:
        fs = 100
        ival = [1000, 4500]      #[1000, 4500]
        class_0 = 0    #left
        class_1 = 1    #right
        class_rest = 2 #foot

        fname = dirname+name_id
        file = open(fname, 'rb')
        data = pickle.load(file)
        file.close()
        cnt = data['cnt']
        mrk_class = data['mrk_class']
        mrk_pos = data['mrk_pos']
        mrk_pos_samples = np.int64(np.floor(mrk_pos*fs/1000))[0] 

        classes = 0*mrk_class[class_0]+1*mrk_class[class_1]+2*mrk_class[class_rest]
        trials_used = np.where(classes!=2)[0]
        class_mrk_binary = classes[trials_used]

        epo, epo_t = mm.makeepochs(cnt, fs, mrk_pos_samples, ival)
        epo = epo.T
        #print(epo.shape)

        if ns==0:
            X = epo[trials_used, :,:]
            y = class_mrk_binary
        else:
            X = np.concatenate([X,epo[trials_used, :,:]])
            y = np.concatenate([y,class_mrk_binary])
        #print(X.shape,y.shape)
        ns = ns+1

    dataset_Xy ={

        "X": X,    # n_calibration_trials*N*n_subjects x n_channels x trial_length
        "y": y     # n_calibration_trials*N*n_subjects
        }

    #fname = 'generated_data/dataset_Xy'
    #file = open(fname, 'wb')
    #pickle.dump(dataset_Xy, file)
    #file.close()
    return dataset_Xy["X"], dataset_Xy["y"]

def create_test_dataset(name_id, filters, common_clab):
    class_0 = 0    #left
    class_1 = 1    #right
    class_rest = 2 #foot
    
    #load data
    subject_name = 'imag_arrow'+name_id #subject_name = 'imag_arrowVPjj'
    fname = 'data/'+subject_name
    cnt, fs, clab, mnt, mrk_pos, mrk_class, mrk_className = mm.load_mat_data(fname)
    cnt = cnt.T

    # keep only used channels
    # find the data channels to the ones shared with the lead fields (order of channels as in original cnt)
    common_clab_cnt_i=[clab.index(x) for x in common_clab]
    # also eliminate channels for mnt & make a mask to get rid of nan values for plotting
    mnt = mnt[common_clab_cnt_i, :]
    nanmask=np.any(~np.isnan(mnt),1)
    cnt = cnt[common_clab_cnt_i, :]
    
    if filters == True:
        fs = np.array([[100]])
        band = [8., 13.]         #[10.5, 13] in ex. sheet for bandpass filter
        Wn = band / fs[0][0]* 2
        b, a = sp.signal.butter(5, Wn, btype='bandpass')

        #filter data   
        cnt_flt = sp.signal.lfilter(b, a, cnt) 
        clab = list(common_clab)
        # Laplace 19
        c3 = mm.proc_spatialFilter(cnt_flt, clab, 'C3', ['C1','C5','FC3','CP3'])
        c4 = mm.proc_spatialFilter(cnt_flt, clab, 'C4', ['C2','C6','FC4','CP4'])
        fc3 = mm.proc_spatialFilter(cnt_flt, clab, 'FC3', ['C3','FC1','FC5','F3'])
        fc4 = mm.proc_spatialFilter(cnt_flt, clab, 'FC4', ['C4','FC2','FC6','F4'])
        cp3 = mm.proc_spatialFilter(cnt_flt, clab, 'CP3', ['C3','CP1','CP5','P3'])
        cp4 = mm.proc_spatialFilter(cnt_flt, clab, 'CP4', ['C4','CP2','CP6','P4'])
        c1 = mm.proc_spatialFilter(cnt_flt, clab, 'C1', ['C3','Cz','FC1','CP1'])
        c2 = mm.proc_spatialFilter(cnt_flt, clab, 'C2', ['C4','Cz','FC2','CP2']) 

        cz = mm.proc_spatialFilter(cnt_flt, clab, 'Cz', ['C2','C1','FCz','CPz'])
        fc1 = mm.proc_spatialFilter(cnt_flt, clab, 'FC1', ['C1','FC3','FCz','F1'])
        fcz = mm.proc_spatialFilter(cnt_flt, clab, 'FCz', ['Cz','FC2','FC1','Fz'])
        fc2 = mm.proc_spatialFilter(cnt_flt, clab, 'FC2', ['C2','FCz','FC4','F2'])
        cp1 = mm.proc_spatialFilter(cnt_flt, clab, 'CP1', ['C1','CP3','CPz','P1'])
        cpz = mm.proc_spatialFilter(cnt_flt, clab, 'CPz', ['Pz','Cz','CP1','CP2'])
        cp2 = mm.proc_spatialFilter(cnt_flt, clab, 'CP2', ['CP4','CPz','C2','P2'])

        c5 = mm.proc_spatialFilter(cnt_flt, clab, 'C5', ['C3','T7','CP5','FC5'])
        c6 = mm.proc_spatialFilter(cnt_flt, clab, 'C6', ['C4','T8','CP6','FC6'])

        fz = mm.proc_spatialFilter(cnt_flt, clab, 'Fz', ['Fpz','F3','F4','Cz'])
        pz = mm.proc_spatialFilter(cnt_flt, clab, 'Pz', ['POz','P1','P2','CPz'])
        clab_flt = ['C3 lap', 'C4 lap', 'FC3 lap', 'FC4 lap', 'CP3 lap', 'CP4 lap', 'C1 lap', 'C2 lap', 'Cz lap', 'FC1 lap', 'FCz lap', 'FC2 lap', 'CP1 lap', 'CPz lap', 'CP2 lap', 'C5 lap','C6 lap', 'Fz lap', 'Pz lap']
        cnt = np.concatenate((c3,c4,fc3,fc4,cp3,cp4,c1,c2,cz,fc1,fcz,fc2,cp1,cpz,cp2,c5,c6,fz,pz))
    else:
        print('Using raw data')
        clab = list(common_clab)
        # Raw 19
        c3 = mm.get_channel_cnt(cnt, clab, 'C3')
        c4 = mm.get_channel_cnt(cnt, clab, 'C4')
        fc3 = mm.get_channel_cnt(cnt, clab, 'FC3')
        fc4 = mm.get_channel_cnt(cnt, clab, 'FC4')
        cp3 = mm.get_channel_cnt(cnt, clab, 'CP3')
        cp4 = mm.get_channel_cnt(cnt, clab, 'CP4')
        c1 = mm.get_channel_cnt(cnt, clab, 'C1')
        c2 = mm.get_channel_cnt(cnt, clab, 'C2')

        cz = mm.get_channel_cnt(cnt, clab, 'Cz')
        fc1 = mm.get_channel_cnt(cnt, clab, 'FC1')
        fcz = mm.get_channel_cnt(cnt, clab, 'FCz')
        fc2 = mm.get_channel_cnt(cnt, clab, 'FC2')
        cp1 = mm.get_channel_cnt(cnt, clab, 'CP1')
        cpz = mm.get_channel_cnt(cnt, clab, 'CPz')
        cp2 = mm.get_channel_cnt(cnt, clab, 'CP2')

        c5 = mm.get_channel_cnt(cnt, clab, 'C5')
        c6 = mm.get_channel_cnt(cnt, clab, 'C6')
        fz = mm.get_channel_cnt(cnt, clab, 'Fz')
        pz = mm.get_channel_cnt(cnt, clab, 'Pz')

        cnt = np.concatenate((c3,c4,fc3,fc4,cp3,cp4,c1,c2,cz,fc1,fcz,fc2,cp1,cpz,cp2,c5,c6,fz,pz))
        
    mrk_pos_samples = np.int64(np.floor(mrk_pos*fs/1000))[0] 

    classes = 0*mrk_class[class_0]+1*mrk_class[class_1]+2*mrk_class[class_rest]
    trials_used = np.where(classes!=2)[0]
    class_mrk_binary = classes[trials_used]

    fs = 100
    ival = [1000, 4500]      #[1000, 4500]
    epo, epo_t = mm.makeepochs(cnt, fs, mrk_pos_samples, ival)
    epo = epo.T
    #print(epo.shape)

    X = epo[trials_used, :,:]
    y = class_mrk_binary

    dataset_Xy ={

        "X": X,    
        "y": y
        }

    return dataset_Xy["X"], dataset_Xy["y"]
