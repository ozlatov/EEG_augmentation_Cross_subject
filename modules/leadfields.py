import numpy as np
import pickle
import modules.my_modules as mm

def load_leadfields():
    '''
    load head model files
    '''
    #only left-right classes
    names_used = pickle.load( open( "leadfields/left_right_names", "rb" ) ) #"left_right_names" (based on multisubj)

    file = open("leadfields/common_clab", 'rb')
    common_clab = pickle.load(file)
    common_clab = np.asarray(common_clab)
    file.close()

    # get the channel labels for the lead fields
    clab_leads = np.load('leadfields/clab.npy')
    clab_leadfields = list(clab_leads[0])
    #print("Number of channels in leadfields:", len(clab_leadfields))

    # make a mask to exclude dipoles outside of the head
    allin = np.load('leadfields/AllIn.npy')
    dipoles_in = np.where(allin[0]==1)

    #load leadfields for MUSIC   # L : (N,M,3) ndarray for m grid points; L(:,i,j) is the potential of a unit dipole at point i in direction j
    gridpos = np.load('leadfields/gridpos.npy') #3 dimentions x 3990 sources x 318 subjects; positions of voxels(places for dipoles)
    leadfields = np.load('leadfields/leadfielddatabase.npy') #92 channels x 3990 sources x 3 dimentions x 318 subjects
    gridpos = gridpos[:, dipoles_in[0], :]
    leadfields = leadfields[:, dipoles_in[0], :, :]  # unsure why this gives an error if both dimensions are trimmed at once

    #load gridnorms
    gridnorms = np.load('leadfields/gridnorms.npy') #3D orientation of dipoles when they are orthogonal to the brain surface
    gridnorms = gridnorms[:, dipoles_in[0], :]
    #print("Shape of gridnorms:", gridnorms.shape)

    common_clab_leads = [clab_leadfields.index(x) for x in common_clab]  # the indices of the common labels in the lead field clab
    #print("Number of common channels in data and leadfields:", len(common_clab))

    leadfields = leadfields[common_clab_leads,:,:,:]
    #print("Shape of leadfields:")
    #print(gridpos.shape)
    #print(leadfields.shape)
    
    return names_used, common_clab, leadfields, gridnorms, gridpos
