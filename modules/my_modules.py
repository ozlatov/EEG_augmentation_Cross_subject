import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from os.path import dirname, join as pjoin
import scipy as sp
import scipy.io
import scipy.io as sio
import scipy.interpolate
import scipy.signal

#my_toolbox

def load_mat_data(fname):
    '''
    Synopsis:
        cnt, fs, clab, mnt, mrk_pos, mrk_class, mrk_className = load_mat_data(fname)
    Arguments:
        fname: name of the mat file
    Returns:\
        cnt:    a 2D array of multi-channel timeseries (channels x samples), (unit [uV] - not sure)
        fs:   sampling frequency [Hz]
        clab: a 1D array of channel names  (channels)
        mnt:  a 2D array of channel coordinates (channels x 2)   
              The electrode montage "mnt" holds the information of the 
              2D projected positions of the channels, i.e. electrodes, 
              on the scalp - seem from the top with nose up.
        mrk_pos:   a 1D array of marker positions (in samples)
        mrk_class: a 1D array that assigns markers to classes (0, 1)
        mrk_className: a list that assigns class names to classes
    '''
    md = sio.loadmat(fname)
    mrk_pos=md['mrk'][0][0][0]*10 #*10 to have the same values
    mnt = np.concatenate((md['mnt'][0][0][0], md['mnt'][0][0][1]), axis=1)# a bit different values
    mrk_class = md['mrk'][0][0][3]
    if mrk_class.shape[0]==1:
        mrk_class = md['mrk'][0][0][5]
    mrk_className = md['mrk'][0][0][4]
    clab = md['nfo'][0][0][1][0]  # clab = md['nfo'][0][0][0][0]
    if clab.shape[0]==1:
        clab = md['nfo'][0][0][0][0]
    #print(clab.shape)
    clab = [x[0] for x in clab]
    fs = md['fs_orig']
    cnt = md['ch1']
    for ch in range(len(clab)-1):
        key='ch'+str(ch+2)
        cnt = np.concatenate((cnt,md[key]), axis=1)
    
    return cnt, fs, clab, mnt, mrk_pos, mrk_class, mrk_className

def closest_nodes(node, nodes):
    '''
    Returns ids of closest nodes to the given node and corresponding distances
    '''
    dist = np.sum((nodes - node)**2, axis=1)
    return np.argsort(dist, axis = 0), np.sort(dist)

def new_dipoles(node, nodes, N, distance = 10e-5):
    
    '''
    Return N dipoles from inpun dipoles list (nodes), that are the closest to node, but at least 
    at distance 'distance' from it 
    
    node - 3D coordinates of the node (array (3,))
    nodes - array of k nodes 3D coordinates (array (k, 3))
    
    '''
    closest_dipoles, distances = closest_nodes(node,nodes)
    n_from = int(np.sum(distance > distances))
    n_to = n_from + N
    dip_list = closest_dipoles[n_from:n_to]
    return dip_list

#bci_minitoolbox

def scalpmap(mnt, v, clim='minmax', cb_label=''): 
    '''
    Usage:
        scalpmap(mnt, v, clim='minmax', cb_label='')
    Parameters:
        mnt: a 2D array of channel coordinates (channels x 2)
        v:   a 1D vector (channels)
        clim: limits of color code, either
          'minmax' to use the minimum and maximum of the data
          'sym' to make limits symmetrical around zero, or
          a two element vector giving specific values
        cb_label: label for the colorbar
    '''    
    # interpolate between channels
    xi, yi = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    xi, yi = np.meshgrid(xi, yi)
    rbf = sp.interpolate.Rbf(mnt[:,0], mnt[:,1], v, function='linear')
    zi = rbf(xi, yi)
        
    # mask area outside of the scalp  
    a, b, n, r = 50, 50, 100, 50
    mask_y, mask_x = np.ogrid[-a:n-a, -b:n-b]
    mask = mask_x*mask_x + mask_y*mask_y >= r*r    
    zi[mask] = np.nan

    if clim=='minmax':
        vmin = v.min()
        vmax = v.max()
    elif clim=='sym':
        vmin = -np.absolute(v).max()
        vmax = np.absolute(v).max()
    else:
        vmin = clim[0]
        vmax = clim[1]
    
    plt.imshow(zi, vmin=vmin, vmax=vmax, origin='lower', extent=[-1, 1, -1, 1], cmap='jet')
    plt.colorbar(shrink=.5, label=cb_label)
    plt.scatter(mnt[:,0], mnt[:,1], c='k', marker='+', vmin=vmin, vmax=vmax)
    plt.axis('off')


def makeepochs(X, fs, mrk_pos, ival):
    '''
    Usage:
        makeepochs(X, fs, mrk_pos, ival)
    Parameters:
        X: 2D array of multi-channel timeseries (channels x samples) 
        fs: sampling frequency [Hz]
        mrk_pos: marker positions [sa]
        ival: a two element vector giving the time interval relative to markers (in ms)
    Returns:
        epo: a 3D array of segmented signals (samples x channels x epochs)
        epo_t: a 1D array of time points of epochs relative to marker (in ms)
    '''
    time = np.array([range(np.int(np.floor(ival[0]*fs/1000)), 
                           np.int(np.ceil(ival[1]*fs/1000))+1)])
    T = time.shape[1]
    nEvents = len(mrk_pos)
    nChans = X.shape[0]
    idx = (time.T+np.array([mrk_pos])).reshape(1, T*nEvents)    
    epo = X[:,idx].T.reshape(T, nEvents, nChans)
    epo = np.transpose(epo, (0,2,1))
    epo_t = np.linspace(ival[0], ival[1], T)
    return epo, epo_t


def baseline(epo, epo_t, ref_ival):
    '''
    Usage:
        epo = baseline(epo, epo_t, ref_ival)
    Parameters:
        epo: a 3D array of segmented signals, see makeepochs
        epo_t: a 1D array of time points of epochs relative to marker (in ms)
        ref_ival: a two element vector specifying the time interval for which the baseline is calculated [ms]
    '''
    idxref = (ref_ival[0] <= epo_t) & (epo_t <= ref_ival[1])
    eporef = np.mean(epo[idxref, :, :], axis=0, keepdims=True)
    epo = epo - eporef
    return epo

#misc_functions (SSD)
def train_SSD(cnt, center_band, flank1_band, flank2_band, filterorder, fs):
    ''' Usage: W, A, D, Y = train_SSD(cnt, center_band, flank1_band, flank2_band, filterorder, fs) '''
    
    b_c, a_c = sp.signal.butter(filterorder, np.array(center_band)*2/fs[0], btype='bandpass')
    cnt_filt_c = sp.signal.lfilter(b_c, a_c, cnt, axis=1)
    S1 = np.cov(cnt_filt_c)
    del cnt_filt_c
    b_f1, a_f1 = sp.signal.butter(filterorder, np.array(flank1_band)*2/fs[0], btype='bandpass')
    cnt_filt_f1 = sp.signal.lfilter(b_f1, a_f1, cnt, axis=1)
    S_f1 = np.cov(cnt_filt_f1)
    del cnt_filt_f1
    b_f2, a_f2 = sp.signal.butter(filterorder, np.array(flank2_band)*2/fs[0], btype='bandpass')
    cnt_filt_f2 = sp.signal.lfilter(b_f2, a_f2, cnt, axis=1)
    S_f2 = np.cov(cnt_filt_f2)
    del cnt_filt_f2
    
    S2 = (S_f1+S_f2)/2

    D, W = sp.linalg.eigh(a=S1, b=S2)
    Y = np.dot(W.T, cnt)
    S_x = np.cov(cnt)
    S_y = np.cov(Y)
    A = np.dot(np.dot(S_x, W), np.linalg.inv(S_y))
    
    return W, A, D, Y

def plot_PSD3class(cnt, fs, mrk_pos, mrk_class, ival):
    epo, _ = bci.makeepochs(cnt, fs, mrk_pos, ival)
    print(epo.shape)
    X1 = epo[:, 0, mrk_class[0,:]==1]
    X2 = epo[:, 0, mrk_class[1,:]==1]
    X3 = epo[:, 0, mrk_class[2,:]==1]
    f1, X1psd = sp.signal.welch(X1.flatten('F'), fs=fs)
    f2, X2psd = sp.signal.welch(X2.flatten('F'), fs=fs)
    f3, X3psd = sp.signal.welch(X3.flatten('F'), fs=fs)
    plt.semilogy(f1.T, X1psd)
    plt.semilogy(f2.T, X2psd)
    plt.semilogy(f3.T, X3psd)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [$uV^2$/Hz]')

def get_channel_cnt(cnt, clab, chan):
    '''
    Usage:
        cnt_sf = get_channel_cnt(cnt, clab, chan)
    Parameters:
        cnt:       a 2D array of multi-channel timeseries (size: channels x samples),
        clab:      a 1D array of channel names  (size: channels)
        chan:      channel
    Returns:
        cnt_sf:    timeseries of channel (size: 1 x samples)
    '''
    cidx= clab.index(chan)
    cnt_sf = cnt[[cidx],:]
    return cnt_sf

def proc_spatialFilter(cnt, clab, chan, neighbors='*'):
    '''
    Usage:
        cnt_sf = proc_spatialFilter(cnt, clab, chan, neighbors='*')
    Parameters:
        cnt:       a 2D array of multi-channel timeseries (size: channels x samples),
        clab:      a 1D array of channel names  (size: channels)
        chan:      channel of center location
        neighbors: labels of channels that are to be subtracted 
    Returns:
        cnt_sf:    timeseries of spatially filtered channel (size: 1 x samples)
    Examples:
        cnt_c4_bip = proc_spatialFilter(cnt, clab, 'C4', 'CP4')
        cnt_c4_lap = proc_spatialFilter(cnt, clab, 'C4', ['C2','C6','FC4','CP4'])
        cnt_c4_car = proc_spatialFilter(cnt, clab, 'C4', '*')
    '''
    cidx= clab.index(chan)
    if isinstance(neighbors, list):
        nidx = [clab.index(cc) for cc in neighbors]
    elif neighbors == '*':
        nidx = range(len(clab))   # Common Average Reference (CAR)
    else:
        nidx = [clab.index(neighbors)]
    cnt_sf = cnt[[cidx],:] - np.mean(cnt[nidx,:], axis=0)
    return cnt_sf

def my_interpolation(cnt, clab, chan, neighbors='*'):
    '''
    Usage:
        cnt_sf = my_interpolatio(cnt, clab, chan, neighbors='*')
    Parameters:
        cnt:       a 2D array of multi-channel timeseries (size: channels x samples),
        clab:      a 1D array of channel names  (size: channels)
        chan:      channel of center location
        neighbors: labels of channels that are to be subtracted 
    Returns:
        cnt_sf:    timeseries of interpolated channel (size: 1 x samples)
    Examples:
        cnt_c4_lap = my_interpolation(cnt, clab, 'C4', ['C2','C6','FC4','CP4'])
    '''
    cidx= clab.index(chan)
    if isinstance(neighbors, list):
        nidx = [clab.index(cc) for cc in neighbors]
    elif neighbors == '*':
        nidx = range(len(clab))   # Common Average Reference (CAR)
    else:
        nidx = [clab.index(neighbors)]
    cnt_sf = np.mean(cnt[nidx,:], axis=0)
    return cnt_sf

def plot_SSD_components (W_SSD, A_SSD, D_SSD, Y_SSD, mnt, nanmask, fs, selected_chans, figsize=[16,8]):
    mnt[0]=[0,1]  #???
    mnt[-2]=[0,1.1]
    mnt[-1]=[0,1.2]

    plt.figure(figsize=figsize) 
    maxamp=abs(A_SSD[nanmask,selected_chans[0]:]).max()
    for ki in range(4):
        plt.subplot(2, 4, ki+1)
        #bci.scalpmap(mnt[nanmask,:], W_SSD[nanmask,selected_chans[ki]], clim='sym', cb_label='[a.u.]')
        #plt.title('filter - ev = %.2f' % D_SSD[selected_chans[ki]])
        #plt.subplot(2, 4, ki+5)
        #bci.scalpmap(mnt[nanmask,:], A_SSD[nanmask,selected_chans[ki]], clim=(-maxamp,maxamp), cb_label='[a.u.]')
        bci.scalpmap(mnt[nanmask,:], A_SSD[nanmask,selected_chans[ki]], clim='sym', cb_label='[a.u.]')
        plt.title('patt_ev = %.2f' % D_SSD[selected_chans[ki]])   
    plt.show()
    
#MUSIC

def music(patt, L, grid):
    """Performs a MUSIC-scan for a given pattern in the leadfield

    Parameters
    ----------
    patt : (N,M) ndarray for N channels; 
        each column in patt represents a spatial pattern;
        (only the span(patt) matters; mixing of the patterns has no effect)
    L : (N,M,3) ndarray for m grid points; L(:,i,j) is the potential
        of a unit dipole at point i in direction j
    grid : (M,3) ndarray denoting locations of grid points
    
    Returns
    -------
    s : (M, K) 
        s(i,k) indicates fit-quality (from 0 (worst) to 1 (best)) at grid-point
        i of k.th dipole (i.e. preceeding k-1 dipoles are projected out 
        at each location); the first column is the 'usual' MUSIC-scan
    vmax : (N, ) ndarray
        the field of the best dipole
    imax : int
        denotes grid-index of best dipole
    dip_mom : (3, ) ndarray
        the moment of the  best dipole
    dip_loc : (3, ) ndarray
        the location of the  best dipole
               
    """
    nchan, nx = (patt.shape, 1)    
    dims=np.ones(4,dtype=int)
    dims[:len(L.shape)]=L.shape
    nchan, ng, ndum, nsubj =    dims
    
    data = patt /np.linalg.norm(patt)
    nd = np.minimum(nx,ndum)
    
    [s,vmax,imax]=calc_spacecorr_all(L,data,nd)
    
    dip_mom=vmax2dipmom(L,imax,vmax)
    dip_loc = grid[imax,:]
    
    return s, vmax, imax, dip_mom, dip_loc

def calc_spacecorr_all(V, data, nd):

    dims=np.ones(4,dtype=int)
    dims[:len(V.shape)]=V.shape
    nchan, ng, ndum, nsubj =    dims                            # [nchan,ng,ndum]=size(V);
    s = np.zeros((ng, nd))                                      # s=zeros(ng,nd);

    for i in range(ng):                                         # for i=1:ng;
        Vortholoc = sp.linalg.orth(np.squeeze(V[:, i, :]))      # Vortholoc=orth(squeeze(V(:,i,:)));
        s[i, :] = calc_spacecorr(Vortholoc, data, nd)           # s(i,:)=calc_spacecorr(Vortholoc,data,nd);

    imax = np.argmax(s[:, 0])                                   # [smax,imax]=max(s(:,1));
    Vortholoc = sp.linalg.orth(np.squeeze(V[:, imax, :]))       # Vortholoc=orth(squeeze(V(:,imax,:)));
    vmax , sbest= calc_bestdir(Vortholoc, data)                        # vmax=calc_bestdir(Vortholoc,data);
    return s, vmax, imax

def calc_spacecorr(Vloc, data_pats, nd):

    A = np.matmul(data_pats.T, Vloc)                                      # A=data_pats'*Vloc;
    s = np.sqrt(np.abs(np.linalg.eigvals(np.outer(A, A))[0]))                    # s=sd(1:nd);

    return s

def vmax2dipmom(V, imax_all, vmax_all):
    if np.isscalar(imax_all):
        ns = 1 
        vmax_all=np.expand_dims(vmax_all,1)
    else: 
        ns =len(imax_all)
        
    dips_mom_all = np.zeros((ns, 3))                                # dips_mom_all = zeros(ns, 3);

    for i in range(ns):                                             # for i=1:ns
        Vloc = np.squeeze(V[:, imax_all, :])                     # Vloc = squeeze(V(:, imax_all(i),:));

        v = vmax_all[:,i]                                         # v = vmax_all(:, i);
        dip = np.linalg.inv(Vloc.T @ Vloc) @ Vloc.T @ v             # dip = inv(Vloc'*Vloc)*Vloc' * v;
        dips_mom_all[i, :] = dip.T / np.linalg.norm(dip)            # dips_mom_all(i,:)=dip'/norm(dip);

    return dips_mom_all

def calc_bestdir(Vloc, data_pats, proj_pats={}):

    if len(proj_pats) == 0:                                                 # if nargin==2
        A = data_pats.T @ Vloc                                              # A=data_pats'*Vloc;
        u, s, v = sp.linalg.svd(A)                                          # [u s, v]=svd(A);
        vmax = Vloc @ v[:, 0]                                               # vmax=Vloc*v(:,1);
        vmax = vmax / np.linalg.norm(vmax)                                  # vmax=vmax/norm(vmax);
        s = s[0]                                                            # s=s(1,1);
    else:
        n, m = Vloc.shape                                                   # [n m]=size(Vloc);
        a = proj_pats.T @ Vloc
        V_proj = sp.linalg.orth(Vloc - proj_pats @ a)                       # V_proj=orth(Vloc-proj_pats*(proj_pats'*Vloc));
        A = data_pats.T @ V_proj                                            # A=data_pats'*V_proj;
        u, s, v = sp.linalg.svd(A)                                          # [u, s v]=svd(A);
        BB = Vloc.T @ proj_pats                                             # BB=(Vloc'*proj_pats);
        Q = np.linalg.inv(np.identity(m) - BB @ BB.T + np.sqrt(np.finfo(float).eps))   # Q=inv(eye(m)-BB*BB'+sqrt(eps));
        vmax = Vloc @ (Q @ Vloc.T @ (V_proj @ v[:, 0]))                     # vmax=Vloc*(Q*Vloc'*(V_proj*v(:,1)));
        vmax = vmax / np.linalg.norm(vmax)                                             # vmax=vmax/norm(vmax);
        s = s[0, 0]                                                         # s=s(1,1);

    return vmax, s


