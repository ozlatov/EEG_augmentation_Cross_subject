import numpy as np


def cov_shrink(X):
    '''
    Estimate covariance of given data using shrinkage estimator.
    
    Synopsis:
        C= cov_shrink(X)
    Argument:
        X: data matrix (features x samples)
    Output:
        C: estimated covariance matrix
    '''
    Xc = X - np.mean(X, axis=1, keepdims=True)
    d, n = Xc.shape
    Cemp= Xc.dot(Xc.T)/(n-1)
    
    sumVarCij = 0
    for ii in range(d):
        for jj in range(d):
            varCij = np.var(Xc[ii,:]*Xc[jj,:])
            sumVarCij += varCij

    nu = np.mean(np.diag(Cemp))
    gamma = n/(n-1.)**2 * sumVarCij / sum(sum((Cemp-nu*np.eye(d,d))**2))
    S= gamma*nu*np.eye(d,d) + (1-gamma)*Cemp
    return S


def train_LDAshrink(X, y):
    '''
    Synopsis:
        w, b= train_LDAshrink(X, y)
    Arguments:
        X: data matrix (features X samples)
        y: labels with values 0 and 1 (1 x samples)
    Output:
        w: LDA weight vector
        b: bias term
    '''
    mu1 = np.mean(X[:, y==0], axis=1)
    mu2 = np.mean(X[:, y==1], axis=1)
    # pool centered features to estimate covariance on samples of both classes at once
    Xpool = np.concatenate((X[:, y==0]-mu1[:,np.newaxis], X[:, y==1]-mu2[:,np.newaxis]), axis=1)
    C = cov_shrink(Xpool)
    w = np.linalg.pinv(C).dot(mu2-mu1)
    b = w.T.dot((mu1 + mu2) / 2.)
    return w, b


def train_LDA(X, y):
    '''
    Synopsis:
        w, b= train_LDA(X, y)
    Arguments:
        X: data matrix (features X samples)
        y: labels with values 0 and 1 (1 x samples)
    Output:
        w: LDA weight vector
        b: bias term
    '''
    mu1 = np.mean(X[:, y == 0], axis=1)
    mu2 = np.mean(X[:, y == 1], axis=1)

    ## Three ways to get an estimate of the covariance
    # -- 1. Simply average class-covariance matrices
    #C1 = np.cov(X[:, y==0])
    #C2 = np.cov(X[:, y==1])
    #C = (C1 + C2) / 2
    # -- 2. Weighted average of class-covariance matrices
    #C1 = np.cov(X[:, y==0])
    #C2 = np.cov(X[:, y==1])
    #N1= np.sum(y==0)           # this would be the weighted average  
    #N2= np.sum(y==1)
    #C= (N1-1)/(N1+N2-1)*C1 + (N2-1)/(N1+N2-1)*C2
    # -- 3. Center features classwise to estimate covariance on all samples at once
    Xpool = np.concatenate((X[:, y==0]-mu1[:,np.newaxis], X[:, y==1]-mu2[:,np.newaxis]), axis=1)
    C = np.cov(Xpool)

    w = np.linalg.pinv(C).dot(mu2-mu1)
    b = w.T.dot((mu1 + mu2) / 2)
    return w, b


def crossvalidation(classifier_fcn, X, y, folds=10, verbose=False):
    '''
    Synopsis:
        loss_te, loss_tr= crossvalidation(classifier_fcn, X, y, folds=10, verbose=False)
    Arguments:
        classifier_fcn: handle to function that trains classifier as output w, b
        X:              data matrix (features X samples)
        y:              labels with values 0 and 1 (1 x samples)
        folds:         number of folds
        verbose:        print validation results or not
    Output:
        loss_te: value of loss function averaged across test data
        loss_tr: value of loss function averaged across training data
    '''
    nDim, nSamples = X.shape
    inter = np.round(np.linspace(0, nSamples, num=folds + 1)).astype(int)
    perm = np.random.permutation(nSamples)
    errTr = np.zeros([folds, 1])
    errTe = np.zeros([folds, 1])

    for ff in range(folds):
        idxTe = perm[inter[ff]:inter[ff + 1] + 1]
        idxTr = np.setdiff1d(range(nSamples), idxTe)
        w, b = classifier_fcn(X[:, idxTr], y[idxTr])
        out = w.T.dot(X) - b
        errTe[ff] = loss_weighted_error(out[idxTe], y[idxTe])
        errTr[ff] = loss_weighted_error(out[idxTr], y[idxTr])

    if verbose:
        print('{:5.1f} +/-{:4.1f}  (training:{:5.1f} +/-{:4.1f})  [using {}]'.format(errTe.mean(), errTe.std(),
                                                                                     errTr.mean(), errTr.std(), 
                                                                                     classifier_fcn.__name__))
    return np.mean(errTe), np.mean(errTr)


def loss_weighted_error(out, y):
    '''
    Synopsis:
        loss= loss_weighted_error( out, y )
    Arguments:
        out:  output of the classifier
        y:    true class labels
    Output:
        loss: weighted error
    '''
    loss = 50 * (np.mean(out[y == 0] >= 0) + np.mean(out[y == 1] < 0))
    return loss
