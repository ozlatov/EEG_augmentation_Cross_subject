import modules.my_modules as mm
import numpy as np
from numpy import linalg as la
import pickle

def augment_data(dirname, names_used, N, n_calibration_trials, dipole_shift, common_clab, leadfields, gridnorms, gridpos):
        '''
        Augment the data and save in the corresponding directory
        Arguments:
        dirname: directory to save data
        names_used: a list with subject names used for augmentation
        N: N times more data after augmentation
        n_calibration_trials: how many originally measured trials per subject to use, <=150
        dipole_shift: the distance of the dipole shift
        common_clab: list of channels used
        leadfields:  ndarray for grid points; L(:,i,j) is the potential
        of a unit dipole at point i in direction j
        gridnorms: gridnorms (not used anymore)
        gridpos: ndarray denoting locations of grid points
        '''
    #SSD parameters
    c_band = [8, 13]        # central band for SSD; [8, 13]
    l_band = [5, 8]         # left band for SSD; [5, 8]
    r_band = [13, 16]       # right band for SSD; [13, 16] 
    subj_id = 80            # whose head model to use for source localization; 80

    ns = 0
    for name_id in names_used:
        ns=ns+1
        if ns in [1,10]:
            print(ns, name_id)
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

        #keep n_calibration_trials of 0 and 1 class
        trial_count = mrk_class[0,:].cumsum()+mrk_class[1,:].cumsum()
        cut_trials = np.where(trial_count==n_calibration_trials+1)[0][0] 
        cnt_train = cnt[:,0:int(mrk_pos[0][cut_trials]/10)-1] #cut the data to keep n_calibration_trials of 0 and 1 class
        
        #generate data
        # SSD
        W_SSD, A_SSD, D_SSD, Y_SSD = mm.train_SSD(cnt_train, c_band, l_band, r_band, 4, fs) #it doesn't "see" cnt_test
        # filter train and test data with SSD; number of components for filtering corresponds to number of dipoles we will search for
        n_ssd = A_SSD.shape[1] #total number of SSD components

        nChans = len(common_clab)
        trial_length = cnt_train.shape[1] #length of original cnt_train data
        generated_data_result = np.zeros((nChans, (N-1)*trial_length))
        comp_strength = []
        norm_factors = []
        for comp_id in range(n_ssd):    #n_ssd
            #components from A_SSD to use:
            from_id = n_ssd-comp_id-1
            to_id = n_ssd-comp_id
            mask_change = np.zeros(A_SSD.shape[0]).astype(np.bool)
            mask_change[from_id:to_id] = True
            cnt_filtered = A_SSD[:, mask_change] @ W_SSD[:, mask_change].T @ cnt_train

            #MUSIC
            V = leadfields[:, :, :, subj_id] # could test different subjects or see which dipoles most commonly chosen etc.
            V = V[:,:,:, np.newaxis]
            grid = gridpos[:, :, subj_id].T
            grid = grid[:, np.newaxis, :]
            patt = A_SSD[:, (from_id):(from_id+1)] # it's the same as A_SSD[:, mask_change]
            s, vmax, imax, dip_mom, dip_loc = mm.music(patt, V, grid)
            #print("source localized")

            # Find where to shift dipoles (new dipoles ids in lead field)
            shifted_dipoles = mm.new_dipoles(gridpos[:,int(imax),subj_id], gridpos[:,:,subj_id].transpose(), N, dipole_shift)

            #Augmented patterns
            augmented_patterns = np.zeros((leadfields.shape[0], N-1))
            norm_factor = la.norm(A_SSD[:, mask_change].T) #norm of the original SSD pattern is used for normalization
            norm_factors.append(norm_factor)
            #normalize originally localized pattern for baseline 
            wrong_signs = np.sum(np.multiply(patt,vmax[:,np.newaxis])<0)
            if wrong_signs > n_ssd/2:
                vmax = -vmax
            vmax = vmax * norm_factor / la.norm(vmax)
            #for every shifted dipole i and for every ssd localized dipole j
            for i in range(N-1):
                pattern = leadfields[:, shifted_dipoles[i], :, subj_id].dot(dip_mom[0]) # was leadfields[:, int(imax[j]), :, i]
                wrong_signs = np.sum(np.multiply(patt,pattern[:,np.newaxis])<0)
                if wrong_signs > n_ssd/2:
                    pattern = -pattern #reverse the sign of the pattern if the majority of coefficients have different sign
                pattern = pattern * norm_factor / la.norm(pattern) #normalize the pattern
                augmented_patterns[:, i] = patt[:,0]*np.clip(pattern/vmax, 0.7,1.5) #apply change to original ssd pattern, clip to avoid outliers

            #Generate new data
            generated_data = np.outer(augmented_patterns[:,0:(N-1)], Y_SSD[-(comp_id+1), :][:,np.newaxis])
            nChans = len(common_clab)
            generated_data = generated_data.reshape(nChans, (N-1)*trial_length)
            #generated_data = generated_data + np.tile(cnt_train_keep, N-1) # NEW
            comp_strength.append(la.norm(generated_data))
            generated_data_result = generated_data_result + generated_data

        generated_data = generated_data_result
        new_data = np.concatenate((cnt_train, generated_data), axis=1) #concatenate original data with generated data
        mrk_class_new = np.tile(mrk_class[:,0:cut_trials], N) #new matk_pos and new_mrk_class
        mrk_pos_new = mrk_pos[:,0:cut_trials]
        for i in range(N-1):
            mrk_pos_new = np.array([np.concatenate((mrk_pos_new, mrk_pos[:,0:cut_trials]+(i+1)*Y_SSD[-(comp_id+1), :].shape[0]*1000/fs), axis=None)]) 


        generated_data ={
            "cnt": new_data,
            "mrk_class": mrk_class_new,
            "mrk_pos": mrk_pos_new
        }

        fname = dirname+name_id
        file = open(fname, 'wb')
        pickle.dump(generated_data, file)
        file.close()

