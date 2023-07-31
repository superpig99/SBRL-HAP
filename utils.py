# utils.py
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

SQRT_CONST = 1e-10

FLAGS = tf.app.flags.FLAGS

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

def lindisc(X,p,t):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,axis=0)
    mean_treated = tf.reduce_mean(Xt,axis=0)

    c = tf.square(2*p-1)*0.25
    f = tf.sign(p-0.5)

    mmd = tf.reduce_sum(tf.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + safe_sqrt(c + mmd)

    return mmd

def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,axis=0)
    mean_treated = tf.reduce_mean(Xt,axis=0)

    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def pop_dist(X,t):
    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.cast(tf.shape(Xc)[0], tf.float32)
    nt = tf.cast(tf.shape(Xt)[0], tf.float32)

    ''' Compute distance matrix'''
    M = pdist2(Xt,Xc)
    return M

def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.cast(tf.shape(Xc)[0], tf.float32)
    nt = tf.cast(tf.shape(Xt)[0], tf.float32)

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))],axis=0)
    Mt = tf.concat([M,row],axis=0)
    Mt = tf.concat([Mt,col],axis=1)

    ''' Compute marginal vectors '''
    a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))],axis=0)
    b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))],axis=0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D, Mlam

def vars_from_scopes(scopes):
    '''
    Parameters list from the variable_scope
    :param scopes: tf.variable_scope
    :return: Trainable parameters
    '''
    current_scope = tf.get_variable_scope().name
    if current_scope != '':
        scopes = [current_scope + '/' + scope for scope in scopes]
    var = []
    for scope in scopes:
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            var.append(v)
    print('current_scope:', current_scope, 'var:',var)
    return var