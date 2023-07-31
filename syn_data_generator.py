import numpy as np
import random
import scipy.special
import os
from scipy.stats import bernoulli


def get_multivariate_normal_params(m, dep, seed=0):
    np.random.seed(seed)

    if dep:
        mu = np.random.normal(size=m) / 10.
        ''' sample random positive semi-definite matrix for cov '''
        temp = np.random.uniform(size=(m, m))
        temp = .5 * (np.transpose(temp) + temp)
        sig = (temp + m * np.eye(m)) / 100.

    else:
        mu = np.zeros(m)
        sig = np.eye(m)

    return mu, sig

def get_latent(m, seed, n, dep):
    L = np.array((n * [[]]))
    if m != 0:
        mu, sig = get_multivariate_normal_params(m, dep, seed)
        L = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
    return L

def generate_XTY(run_dict, size=0):
    ''' Setting... '''
    mA = run_dict["mA"]  # Dimensions of instrumental variables
    mB = run_dict["mB"]  # Dimensions of confounding variables
    mC = run_dict["mC"]  # Dimensions of adjustment variables
    mD = run_dict["mD"]  # Dimensions of irrelevant variables
    sc = run_dict["sc"]  # 1
    sh = run_dict["sh"]  # 0
    init_seed = run_dict["init_seed"] # Fixed random seed
    
    # Dataset size
    if size==0:
        size = run_dict["size"]

    # Randomly generated coefficients
    random_coef = run_dict["random_coef"]

    # fixed coef
    coef_t_AB = run_dict["coef_t_AB"]
    coef_y1_BC = run_dict["coef_y1_BC"]
    coef_y2_BC = run_dict["coef_y2_BC"]

    # Fixed coefficient value
    use_one = run_dict["use_one"]

    # harder datasets
    dep = 0  # overwright; dep=0 generates harder datasets

    # Big Dataset size for sample
    n_trn = size * 100

    # Coefficient of random seed allocation
    seed_coef = 10

    # all dimension
    max_dim = mA + mB + mC + mD

    # Variables
    temp = get_latent(max_dim, seed_coef * init_seed + 4, n_trn, dep)

    # Divide I, C, A, D
    A = temp[:, 0:mA]
    B = temp[:, mA:mA + mB]
    C = temp[:, mA + mB:mA + mB + mC]
    D = temp[:, mA + mB + mC:mA + mB + mC + mD]

    # X: all data; AB: variable related T; BC: variable related Y
    x = np.concatenate([A, B, C, D], axis=1)
    AB = np.concatenate([A, B], axis=1)
    BC = np.concatenate([B, C], axis=1)

    # coef_t_AB： 1: Random normal generation; 2: fixed coefficient; 3: 1 coefficient
    np.random.seed(1 * seed_coef * init_seed)
    if random_coef == "True" or random_coef == "T":
        coefs_1 = np.random.normal(size=mA + mB)
    else:
        coefs_1 = np.array(coef_t_AB)
    if use_one == "True" or use_one == "T":
        coefs_1 = np.ones(shape=mA + mB)
    
    # generate t_binary
    z = np.dot(AB, coefs_1)
    if random_coef == "True" or random_coef == "T" or use_one == "True" or use_one == "T":   
        pass
    else:
        z = z / run_dict["coef_devide_1"]
    per = np.random.normal(size=n_trn)
    pi0_t1 = scipy.special.expit(sc * (z + sh + per))

    t = bernoulli.rvs(pi0_t1)
    
    # coef_y_BC： 1: Random normal generation; 2: fixed coefficient; 3: 1 coefficient
    np.random.seed(2 * seed_coef * init_seed)  # <--
    if random_coef == "True" or random_coef == "T":
        coefs_2 = np.random.normal(size=mB + mC)
    else:
        coefs_2 = np.array(coef_y1_BC)
    if use_one == "True" or use_one == "T":
        coefs_2 = np.ones(shape=mB + mC)
    if random_coef == "True" or random_coef == "T":
        coefs_3 = np.random.normal(size=mB + mC)
    else:
        coefs_3 = np.array(coef_y2_BC)
    if use_one == "True" or use_one == "T":
        coefs_3 = np.ones(shape=mB + mC)

    # base continuous mu_0, mu_1
    if random_coef == "True" or random_coef == "T" or use_one == "True" or use_one == "T":   
        mu_0 = np.dot(BC ** 1, coefs_2) / (mB + mC)
        mu_1 = np.dot(BC ** 2, coefs_3) / (mB + mC)
    else:
        mu_0 = np.dot(BC ** 1, coefs_2) / (mB + mC) / run_dict["coef_devide_2"]
        mu_1 = np.dot(BC ** 2, coefs_3) / (mB + mC) / run_dict["coef_devide_3"]
    
    print('x.shape:',x.shape,'t.shape:',t.shape,'mu0.shape:',mu_0.shape,'mu1.shape:',mu_1.shape)
    return {'x':x, 't':t, 'mu0':mu_0, 'mu1':mu_1, 'z':pi0_t1}

def correlation_sample(data, r, n, dim_v):
    nall = data['x'].shape[0]
    prob = np.ones(nall)

    ite = data['mu1']-data['mu0']

    if r!=0.0:
        for idv in range(dim_v):
            d = np.abs(data['x'][:, -idv - 1] - np.sign(r) * ite)
            prob = prob * np.power(np.abs(r), -10 * d)
    prob = prob / np.sum(prob)
    idx = np.random.choice(range(nall), n, p=prob)
    x = data['x'][idx, :]
    t = data['t'][idx]
    mu0 = data['mu0'][idx]
    mu1 = data['mu1'][idx]

    # continuous y
    y0_cont = mu0 + np.random.normal(loc=0., scale=.1, size=n)
    y1_cont = mu1 + np.random.normal(loc=0., scale=.1, size=n)

    yf_cont, ycf_cont = np.zeros(n), np.zeros(n)
    yf_cont[t>0], yf_cont[t<1] = y1_cont[t>0], y0_cont[t<1]
    ycf_cont[t>0], ycf_cont[t<1] = y0_cont[t>0], y1_cont[t<1]

    # binary y
    median_0 = np.median(mu0)
    median_1 = np.median(mu1)
    mu0[mu0 >= median_0] = 1.
    mu0[mu0 < median_0] = 0.
    mu1[mu1 < median_1] = 0.
    mu1[mu1 >= median_1] = 1.

    yf_bin, ycf_bin = np.zeros(n), np.zeros(n)
    yf_bin[t>0], yf_bin[t<1] = mu1[t>0], mu0[t<1]
    ycf_bin[t>0], ycf_bin[t<1] = mu0[t>0], mu1[t<1]

    # return
    biny_dict = {'x':x, 't':t, 'yf':yf_bin, 'ycf':ycf_bin, 'mu0':mu0, 'mu1':mu1}
    conty_dict = {'x':x, 't':t, 'yf':yf_cont, 'ycf':ycf_cont, 'mu0':y0_cont, 'mu1':y1_cont}

    return biny_dict, conty_dict

def run(run_dict):
    # save path
    outdir_conty = run_dict['outdir'] + 'syn_conty/'
    outdir_biny = run_dict['outdir'] + 'syn_biny/'
    if not os.path.exists(outdir_conty):
        os.mkdir(outdir_conty)
    if not os.path.exists(outdir_biny):
        os.mkdir(outdir_biny)
    
    # train & test size
    size = run_dict['size']
    size_train = int(run_dict['size'] * 0.9)
    size_test = int(run_dict['size'] * 0.1)
    
    # Number of repeated experiments
    num_exp = run_dict["num"]

    ''' Setting... '''
    mA = run_dict["mA"]  # Dimensions of instrumental variables
    mB = run_dict["mB"]  # Dimensions of confounding variables
    mC = run_dict["mC"]  # Dimensions of adjustment variables
    mD = run_dict["mD"]  # Dimensions of irrelevant variables
    # all dimension
    max_dim = mA + mB + mC + mD

    # generate X & T & Y0 & Y1
    raw_data = generate_XTY(run_dict, size)

    ''' bias rate '''
    br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0, 0.0]
    brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}

    # biased sampling
    for r in br:
        # dict for saving
        conty_train_data = {'x':np.zeros((size_train, max_dim, num_exp)), 't':np.zeros((size_train, num_exp)), \
                            'yf':np.zeros((size_train, num_exp)), 'ycf':np.zeros((size_train, num_exp)), 'mu0':np.zeros((size_train, num_exp)), 'mu1':np.zeros((size_train, num_exp))}
        conty_test_data = {'x':np.zeros((size_test, max_dim, num_exp)), 't':np.zeros((size_test, num_exp)), \
                            'yf':np.zeros((size_test, num_exp)), 'ycf':np.zeros((size_test, num_exp)), 'mu0':np.zeros((size_test, num_exp)), 'mu1':np.zeros((size_test, num_exp))}
        biny_train_data = {'x':np.zeros((size_train, max_dim, num_exp)), 't':np.zeros((size_train, num_exp)), \
                            'yf':np.zeros((size_train, num_exp)), 'ycf':np.zeros((size_train, num_exp)), 'mu0':np.zeros((size_train, num_exp)), 'mu1':np.zeros((size_train, num_exp))}
        biny_test_data = {'x':np.zeros((size_test, max_dim, num_exp)), 't':np.zeros((size_test, num_exp)), \
                            'yf':np.zeros((size_test, num_exp)), 'ycf':np.zeros((size_test, num_exp)), 'mu0':np.zeros((size_test, num_exp)), 'mu1':np.zeros((size_test, num_exp))}

        # generate selection probability
        biny_dict, conty_dict = correlation_sample(raw_data, r, size, mD)
        # repeat sampling
        for n_exp in range(num_exp):
            random.seed(n_exp*1000)
            I_test = random.sample(range(0, size), size_test)
            I = range(size); I_train = list(set(I)-set(I_test))
            for k in conty_dict.keys():
                if k=='x':
                    conty_train_data[k][:,:,n_exp] = conty_dict[k][I_train,:]
                    conty_test_data[k][:,:,n_exp] = conty_dict[k][I_test,:]
                    biny_train_data[k][:,:,n_exp] = biny_dict[k][I_train,:]
                    biny_test_data[k][:,:,n_exp] = biny_dict[k][I_test,:]
                else:
                    conty_train_data[k][:,n_exp] = conty_dict[k][I_train]
                    conty_test_data[k][:,n_exp] = conty_dict[k][I_test]
                    biny_train_data[k][:,n_exp] = biny_dict[k][I_train]
                    biny_test_data[k][:,n_exp] = biny_dict[k][I_test]

        # data saving
        np.savez(outdir_conty+'r%s.train.npz'%brdc[r], **conty_train_data)
        np.savez(outdir_conty+'r%s.test.npz'%brdc[r], **conty_test_data)
        np.savez(outdir_biny+'r%s.train.npz'%brdc[r], **biny_train_data)
        np.savez(outdir_biny+'r%s.test.npz'%brdc[r], **biny_test_data)

if __name__ == '__main__':
    # example: Syn_8_8_8_2 with size n=10000
    run_dict ={ 'outdir': "./Syn_8_8_8_2_10000/",
                'size': 10000,
                'num': 10,
                'mA': 8,
                'mB': 8,
                'mC': 8,
                'mD': 2,
                "sc": 1.0,
                "sh": 0.0,
                "init_seed": 4,
                "random_coef": "F",
                "coef_t_AB": [8,8,11,13,8,8,16,11,13,16,11,16,8,8,8,8],
                "coef_y1_BC": [11,8,8,16,13,8,8,16,13,16,11,11,8,8,8,8],
                "coef_y2_BC": [8,8,8,8,16,13,11,11,13,11,16,11,8,8,8,8],
                "coef_devide_1": 10,
                "coef_devide_2": 10,
                "coef_devide_3": 10,
                "use_one": "F",
               }

    run(run_dict)