import numpy as np
import scipy as sp
import modules.my_modules as mm
import pickle

def bp_laplacian_filter(dirname, names_used, common_clab):
    fs = np.array([[100]])
    band = [8., 13.]         #[10.5, 13] in ex. sheet for bandpass filter
    ival = [1000, 4500]      #[1000, 4500]

    ns = 0
    for name_id in names_used:
        ns=ns+1
        if ns in [1,10]:
            print(ns, name_id)

        fname = dirname+name_id
        file = open(fname, 'rb')
        data = pickle.load(file)

        Wn = band / fs[0][0]* 2
        b, a = sp.signal.butter(5, Wn, btype='bandpass')

        #filter data   
        cnt_flt = sp.signal.lfilter(b, a, data['cnt']) 

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
        cnt_flt_laplace19 = np.concatenate((c3,c4,fc3,fc4,cp3,cp4,c1,c2,cz,fc1,fcz,fc2,cp1,cpz,cp2,c5,c6,fz,pz))

        mrk_class = data["mrk_class"]
        mrk_pos = data["mrk_pos"]
        filtered_data ={
            "cnt": cnt_flt_laplace19,
            "mrk_class": mrk_class,
            "mrk_pos": mrk_pos
        }

        fname = dirname+'filtered_'+name_id
        file = open(fname, 'wb')
        pickle.dump(filtered_data, file)
        file.close()

def raw_19_channel_data(dirname, names_used, common_clab):
    fs = np.array([[100]])
    
    ns = 0
    for name_id in names_used:
        ns=ns+1
        if ns in [1,10]:
            print(ns, name_id)

        fname = dirname+name_id
        file = open(fname, 'rb')
        data = pickle.load(file)
        
        clab = list(common_clab)
        # Raw 19
        c3 = mm.get_channel_cnt(data['cnt'], clab, 'C3')
        c4 = mm.get_channel_cnt(data['cnt'], clab, 'C4')
        fc3 = mm.get_channel_cnt(data['cnt'], clab, 'FC3')
        fc4 = mm.get_channel_cnt(data['cnt'], clab, 'FC4')
        cp3 = mm.get_channel_cnt(data['cnt'], clab, 'CP3')
        cp4 = mm.get_channel_cnt(data['cnt'], clab, 'CP4')
        c1 = mm.get_channel_cnt(data['cnt'], clab, 'C1')
        c2 = mm.get_channel_cnt(data['cnt'], clab, 'C2')

        cz = mm.get_channel_cnt(data['cnt'], clab, 'Cz')
        fc1 = mm.get_channel_cnt(data['cnt'], clab, 'FC1')
        fcz = mm.get_channel_cnt(data['cnt'], clab, 'FCz')
        fc2 = mm.get_channel_cnt(data['cnt'], clab, 'FC2')
        cp1 = mm.get_channel_cnt(data['cnt'], clab, 'CP1')
        cpz = mm.get_channel_cnt(data['cnt'], clab, 'CPz')
        cp2 = mm.get_channel_cnt(data['cnt'], clab, 'CP2')

        c5 = mm.get_channel_cnt(data['cnt'], clab, 'C5')
        c6 = mm.get_channel_cnt(data['cnt'], clab, 'C6')
        fz = mm.get_channel_cnt(data['cnt'], clab, 'Fz')
        pz = mm.get_channel_cnt(data['cnt'], clab, 'Pz')

        cnt_19 = np.concatenate((c3,c4,fc3,fc4,cp3,cp4,c1,c2,cz,fc1,fcz,fc2,cp1,cpz,cp2,c5,c6,fz,pz))

        mrk_class = data["mrk_class"]
        mrk_pos = data["mrk_pos"]
        raw_19_data ={
            "cnt": cnt_19,
            "mrk_class": mrk_class,
            "mrk_pos": mrk_pos
        }

        fname = dirname+'raw19_'+name_id
        file = open(fname, 'wb')
        pickle.dump(raw_19_data, file)
        file.close()

