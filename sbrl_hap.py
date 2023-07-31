import tensorflow.compat.v1 as tf
import numpy as np

from utils import *

NUM_ITERATIONS_PER_DECAY = 100

class sbrl_hap(object):
    """
    This file contains the class for SBRL-HAP.
    The network is implemented as a tensorflow graph. 
    It is built upon implementation of the counterfactual regression neural network.
    by Yuling Zhang et.al
    """
        
    def __init__(self, FLAGS, dims):
        self.FLAGS = FLAGS
        self.wd_loss = 0
        
        ''' Initialize input placeholders of cfr_net '''
        self.x  = tf.placeholder("float", shape=[None, dims[0]], name='x') # Features
        self.t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
        self.y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome
        self.ycf = tf.placeholder("float", shape=[None, 1], name='ycf')  # Outcome
        self.idx = tf.placeholder("int32", shape=[None, ], name='indicator')  # Indicates the corresponding index of the current input data
        self.n = tf.placeholder("float", name='sample_num') # number of samples
    
        ''' Parameter placeholders of cfr_net '''
        self.p_t = tf.placeholder("float", name='p_treated')
        self.do_in = tf.placeholder("float", name='dropout_in')
        self.do_out = tf.placeholder("float", name='dropout_out')
        
        ''' activation func '''
        if FLAGS.activation.lower() == 'elu':
            self.activation = tf.nn.elu
        else:
            self.activation = tf.nn.relu
        
        self.i0 = tf.cast(tf.where(self.t < 1)[:, 0], dtype=tf.int32)
        self.i1 = tf.cast(tf.where(self.t > 0)[:, 0], dtype=tf.int32)
        
        self.initializer = tf.keras.initializers.glorot_normal(seed=FLAGS.seed)
        
        self.build_graph(dims)
        self.IPM()
        self.calculate_loss()
        self.setup_train_ops()
        
    def build_graph(self, dims):
        ''' initial HISC loss '''
        self.loss_hsic = tf.Variable(tf.zeros(shape=(), dtype=tf.float32))
        
        ''' sample weights '''
        with tf.variable_scope('weight'):
            if self.FLAGS.reweight_sample:
                sample_weight = tf.get_variable(name='tain_sample_weight', shape=[dims[1], 1], initializer=tf.constant_initializer(1))
                self.sample_weight = tf.gather(sample_weight, self.idx)
                
                self.sample_weight_0 = tf.gather(self.sample_weight, self.i0)
                self.sample_weight_1 = tf.gather(self.sample_weight, self.i1)
                
                self.sample_weight_sfmx = tf.nn.softmax(self.sample_weight, axis=0) * self.n # normalization of sample weights
            else:
                w_t = self.t/(2*self.p_t)
                w_c = (1-self.t)/(2*(1-self.p_t))
                self.sample_weight = w_t + w_c
        
        ''' representation network '''
        with tf.variable_scope('representation'):
            self.rep_C, self.reps_C, self.w_C, self.b_C = self.representation(data_input=self.x,
                                                                              dim_in=dims[0],
                                                                              dim_out=self.FLAGS.rep_dim,
                                                                              layer=self.FLAGS.rep_layer,
                                                                              name='Phi_x')
        ''' two-head predictive nerworks '''
        with tf.variable_scope('outcome'):
            self.mu_Y, self.mu_YCF, self.mus_Y, self.w_muY, self.b_muY = self.output(data_input=self.rep_C,
                                                                                    dim_in=self.FLAGS.rep_dim,
                                                                                    dim_out=self.FLAGS.y_dim,
                                                                                    layer=self.FLAGS.y_layer,
                                                                                    name='Mu_ytx')
    
    def representation(self, data_input, dim_in, dim_out, layer, name):
        rep, weight, bias = [data_input], [], []

        dim = np.around(np.linspace(dim_in, dim_out, layer+1)).astype(int) # number of neurons
        
        if self.FLAGS.batch_norm:
            bn_biases, bn_scales = [], []
        
        if self.FLAGS.p_lambda>0 and self.FLAGS.rep_weight_decay:
            wd = 1.
        else:
            wd = 0.
        
        for i in range(0, layer):
            w, b = self.FC_layer(dim_in=dim[i], dim_out=dim[i+1], name='_{}_{}'.format(i,name), wd=wd)
            weight.append(w)
            bias.append(b)
            out = tf.add(tf.matmul(rep[i], weight[i], name='matmul_{}_{}'.format(i,name)), bias[i], name='add_{}_{}'.format(i,name))
            if self.FLAGS.batch_norm:
                batch_mean, batch_var = tf.nn.moments(out, [0])
                if self.FLAGS.normalization == 'bn_fixed':
                    out = tf.nn.batch_normalization(out, batch_mean, batch_var, 0, 1, 1e-3)
                else:
                    bn_biases.append(tf.Variable(tf.zeros([dim[i+1]]), name='bn_bias_{}_{}'.format(i,name)))
                    bn_scales.append(tf.Variable(tf.ones([dim[i+1]]), name='bn_scale_{}_{}'.format(i,name)))
                    out = tf.nn.batch_normalization(out, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)
            rep.append(tf.nn.dropout(self.activation(out), rate = 1- self.do_in))
            ''' Independence Regularizer for Z^o with gamma3 '''
            if self.FLAGS.p_gamma3>0 and i!=layer-1:
                self.loss_hsic = self.loss_hsic + self.FLAGS.p_gamma3*self.dependence_loss(rep[-1], None, self.sample_weight, None, name=name+'_hsic%d'%i)
        
        if self.FLAGS.normalization == 'divide':
            rep[-1] = rep[-1] / safe_sqrt(tf.reduce_sum(tf.square(rep[-1]), axis=1, keepdims=True))
        
        ''' Independence Regularizer for Z^r with gamma2 '''
        if self.FLAGS.p_gamma2>0:
            self.loss_hsic = self.loss_hsic + self.FLAGS.p_gamma2*self.dependence_loss(rep[-1], None, self.sample_weight, None, name=name+'_hsic%d'%i)
        
        return rep[-1], rep, weight, bias
    
    def FC_layer(self, dim_in, dim_out, name, wd=0):
        if self.FLAGS.var_from == 'get_variable':
            weight = tf.get_variable(name='weight' + name, shape=[dim_in, dim_out], initializer=self.initializer)
            bias = tf.get_variable(name='bias' + name, shape=[1, dim_out], initializer=tf.constant_initializer())
        else:
            weight = tf.Variable(tf.random.normal([dim_in, dim_out], stddev=self.FLAGS.weight_init / np.sqrt(dim_in)), name='weight' + name)
            bias = tf.Variable(tf.zeros([1, dim_out]), name='bias' + name)

        if wd>0:
            self.wd_loss += wd * tf.nn.l2_loss(weight)

        return weight, bias
    
    def output(self, data_input, dim_in, dim_out, layer, name, mode='mu'):
        if self.FLAGS.y_is_binary:
            mu_Y_0, mus_Y_0, w_muY_0, b_muY_0 = self.predict(data_input, dim_in, dim_out, layer, name+'0', 1., 2, mode) # 把所有数据都放进去预测
            mu_Y_1, mus_Y_1, w_muY_1, b_muY_1 = self.predict(data_input, dim_in, dim_out, layer, name+'1', 1., 2, mode)
        else:
            mu_Y_0, mus_Y_0, w_muY_0, b_muY_0 = self.predict(data_input, dim_in, dim_out, layer, name+'0', 1., 1, mode)
            mu_Y_1, mus_Y_1, w_muY_1, b_muY_1 = self.predict(data_input, dim_in, dim_out, layer, name+'1', 1., 1, mode)
        
        mu_YF_0 = tf.gather(mu_Y_0, self.i0) # YF of control group
        mu_YF_1 = tf.gather(mu_Y_1, self.i1) # YF of treated group
        
        mu_YCF_0 = tf.gather(mu_Y_0, self.i1) # YCF of control group
        mu_YCF_1 = tf.gather(mu_Y_1, self.i0) # YCF of control group
        
        mu_YF = tf.dynamic_stitch([self.i0, self.i1], [mu_YF_0, mu_YF_1])
        mu_YCF = tf.dynamic_stitch([self.i0, self.i1], [mu_YCF_1, mu_YCF_0])
        
        mus_Y = mus_Y_0 + mus_Y_1
        w_muY = w_muY_0 + w_muY_1
        b_muY = b_muY_0 + b_muY_1

        ''' Z^p '''
        self.cfeatures_0 = tf.gather(mus_Y_0[-2], self.i0) # control group
        self.cfeatures_1 = tf.gather(mus_Y_1[-2], self.i1) # treated group

        return mu_YF, mu_YCF, mus_Y, w_muY, b_muY
    
    def predict(self, data_input, dim_in, dim_out, layer, name, wd=0, class_num=1, mode='mu'):
        pred, weight, bias = [data_input], [], []
        
        dim = np.around(np.linspace(dim_in, dim_out, layer + 1)).astype(int)
        
        for i in range(0, layer):
            w, b = self.FC_layer(dim_in=dim[i], dim_out=dim[i + 1], name='_{}_{}'.format(i,name), wd=wd)
            weight.append(w)
            bias.append(b)
            out = tf.add(tf.matmul(pred[i], weight[i], name='matmul_{}_{}'.format(i, name)), bias[i], name='add_{}_{}'.format(i,name))
            pred.append(tf.nn.dropout(self.activation(out), rate = 1 - self.do_out))
            ''' Independence Regularizer for Z^o with gamma3 '''
            if self.FLAGS.p_gamma3>0 and i!=layer-1:
                if name[-1]=='0':
                    pred_hid0 = tf.gather(pred[-1], self.i0)
                    self.loss_hsic = self.loss_hsic + self.FLAGS.p_gamma3*self.dependence_loss(pred_hid0, None, self.sample_weight_0, None, name=name+'_hsic%d'%i)
                else:
                    pred_hid1 = tf.gather(pred[-1], self.i1)
                    self.loss_hsic = self.loss_hsic + self.FLAGS.p_gamma3*self.dependence_loss(pred_hid1, None, self.sample_weight_1, None, name=name+'_hsic%d'%i)

        ''' last layer of predictive network (Z^p_ti) '''
        w, b = self.FC_layer(dim_in=dim[-1], dim_out=class_num, name='_{}_{}'.format('pred',name), wd=wd)
        weight.append(w)
        bias.append(b)
        out = tf.add(tf.matmul(pred[-1], weight[-1], name='matmul_{}_{}'.format('pred',name)), bias[-1],name='add_{}_{}'.format('pred',name))
        pred.append(out)
        # pred.append(tf.nn.dropout(out, self.do_out))

        return pred[-1], pred, weight, bias
    
    def IPM(self):
        if self.FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5
        
        if self.FLAGS.p_alpha>0:
            weighted_rep = self.sample_weight_sfmx * self.rep_C
        else:
            weighted_rep = self.rep_C
        
        if self.FLAGS.imb_fun == 'mmd_lin':
            imb_dist = mmd2_lin(weighted_rep, self.t, p_ipm)
        elif self.FLAGS.imb_fun == 'wass':
            imb_dist, imb_mat = wasserstein(weighted_rep, self.t, p_ipm, lam=self.FLAGS.wass_lambda, its=self.FLAGS.wass_iterations, sq=False, backpropT=self.FLAGS.wass_bpt)
        elif self.FLAGS.imb_fun == 'wass2':
            imb_dist, imb_mat = wasserstein(weighted_rep, self.t, p_ipm, lam=self.FLAGS.wass_lambda, its=self.FLAGS.wass_iterations, sq=True, backpropT=self.FLAGS.wass_bpt)
        else:
            imb_dist = lindisc(weighted_rep, p_ipm, self.t)
        
        self.IPM_C = imb_dist
        
    def calculate_loss(self):
        ''' Regression Loss '''
        if self.FLAGS.y_is_binary:
            self.y_pred, _, self.loss_YF = self.log_loss(self.mu_Y, self.y_, True)
            self.ycf_pred, _, self.loss_YCF = self.log_loss(self.mu_YCF, self.ycf)
        else:
            self.y_pred, self.ycf_pred = self.mu_Y, self.mu_YCF
            self.loss_YF, _ = self.l2_loss(self.mu_Y, self.y_, True)
            self.loss_YCF, _ = self.l2_loss(self.mu_YCF, self.ycf)

        self.loss_pred = self.loss_YF # L_Y^w
        self.loss_reg = self.FLAGS.p_lambda * (1e-3 * self.wd_loss) # R_{l2}
        
        self.loss_main = self.loss_pred + self.loss_reg # L_Y^w
        
        ''' Independence Regularizer for Z^p with gamma1 '''
        if self.FLAGS.p_gamma1>0:
            self.loss_hsic = self.loss_hsic + self.FLAGS.p_gamma1*self.dependence_loss(self.cfeatures_0, self.sample_weight_0, name='hsic0')
            self.loss_hsic = self.loss_hsic + self.FLAGS.p_gamma1*self.dependence_loss(self.cfeatures_1, self.sample_weight_1, name='hsic1')
        
        ''' Balancing Regularizer for Phi '''
        self.loss_ipm = self.FLAGS.p_alpha * self.IPM_C
        
        ''' L_w '''
        if self.FLAGS.reweight_sample:
            self.loss_w = tf.square(tf.reduce_sum(self.sample_weight_0)/tf.reduce_sum(1.0 - self.t) - 1.0) + tf.square(tf.reduce_sum(self.sample_weight_1)/tf.reduce_sum(self.t) - 1.0) # (w-1)^2
            
            self.loss_reweight = self.loss_main + self.loss_ipm + self.FLAGS.p_beta*self.loss_w + self.loss_hsic
    
    def dependence_loss(self, cfeatures, cweights, name=''):
        all_weights = tf.nn.softmax(cweights, axis=0, name='all_weights_{}'.format(name))
        rff_features = self.random_fourier_features(cfeatures)
        
        loss = tf.Variable(tf.zeros(shape=(), dtype=tf.float32), name='loss_hsic_{}'.format(name))
        for i in range(rff_features.shape[-1]):
            rff_feature = rff_features[:,:,i]
            hsic_cov = self.cov(rff_feature, all_weights)
            cov_matrix = hsic_cov * hsic_cov
            loss += (tf.reduce_sum(cov_matrix) - tf.linalg.trace(cov_matrix))
        return loss
    
    def random_fourier_features(self, x, w=None, b=None, num_f=1, sum=True, sigma=None):
        if num_f is None:
            num_f = 1
        r = x.shape[1]
        x = tf.expand_dims(x, axis=-1)
        c = x.shape[2]
        if sigma is None or sigma == 0:
            sigma = 1
        if w is None:
            w = 1 / sigma * (tf.random.normal(shape=[num_f, c])) # 让每一维原始特征都映射出num_f维RFF特征
            b = 2 * np.pi * tf.random.uniform(shape=[1, r, num_f], maxval=1)
        
        Z = tf.sqrt(tf.constant(2.0 / num_f))
    
        mid = tf.matmul(x, tf.transpose(w))
        print(x, w, b, mid)
        mid = tf.add(mid, b)
        mid -= tf.reduce_min(mid, axis=1, keepdims=True)
        mid /= tf.reduce_max(mid, axis=1, keepdims=True)
        mid *= np.pi / 2.0
    
        if sum:
            Z = Z * (tf.cos(mid) + tf.sin(mid))
        else:
            Z = Z * tf.concat((tf.cos(mid), tf.sin(mid)), axis=-1)
        
        return Z
    
    def cov(self, x, w=None):
        if w is None:
            n = x.shape[0]
            cov = tf.matmul(tf.transpose(x), x) / n # return (r,r), E[X*X]
            e = tf.reshape(tf.reduce_mean(x, axis=0), [-1, 1]) # return (r,1), E[X]
            res = cov - tf.matmul(e, tf.transpose(e)) # return (r,r), E[X*X]-E[X]*E[X]
        else:
            w = tf.reshape(w, [-1, 1])
            cov = tf.matmul(tf.transpose(w * x), x)
            e = tf.reshape(tf.reduce_sum(w*x, axis=0), [-1, 1])
            res = cov - tf.matmul(e, tf.transpose(e))
    
        return res
    
    def log_loss(self, pred, label, sample = False):
        sigma = 0.995 / (1.0 + tf.exp(-pred)) + 0.0025
        pi_0 = tf.multiply(label, sigma) + tf.multiply(1.0 - label, 1.0 - sigma)

        labels = tf.concat((1 - label, label), axis=1)
        logits = pred

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        if sample and (self.FLAGS.reweight_sample or self.FLAGS.p_alpha>0):
            loss = tf.reduce_mean(self.sample_weight_sfmx * loss)
        else:
            loss = tf.reduce_mean(loss)

        return sigma[:,1:2], pi_0, loss
    
    def l2_loss(self, pred, out, sample = False):

        if sample and (self.FLAGS.reweight_sample or self.FLAGS.p_alpha>0):
            loss = tf.reduce_mean(self.sample_weight_sfmx * tf.square(pred - out))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(pred - out)))
        else:
            loss = tf.reduce_mean(tf.square(pred - out))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(pred - out)))

        return loss, pred_error
    
    def setup_train_ops(self):
        W_vars = vars_from_scopes(['weight'])
        R_vars = vars_from_scopes(['representation'])
        O_vars = vars_from_scopes(['outcome'])

        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(self.FLAGS.lrate, global_step, \
                NUM_ITERATIONS_PER_DECAY, self.FLAGS.lrate_decay, staircase=True)
        wlr = tf.train.exponential_decay(self.FLAGS.wlrate, global_step, \
                NUM_ITERATIONS_PER_DECAY, self.FLAGS.wlrate_decay, staircase=True)
        
        if self.FLAGS.optimizer == 'Adagrad':
            opt = tf.train.AdagradOptimizer(lr)
            wopt = tf.train.AdagradOptimizer(wlr)
        elif self.FLAGS.optimizer == 'GradientDescent':
            opt = tf.train.GradientDescentOptimizer(lr)
            wopt = tf.train.GradientDescentOptimizer(wlr)
        elif self.FLAGS.optimizer == 'Adam':
            opt = tf.train.AdamOptimizer(lr)
            wopt = tf.train.AdamOptimizer(wlr)
        else:
            opt = tf.train.RMSPropOptimizer(lr, self.FLAGS.rms_decay)
            wopt = tf.train.RMSPropOptimizer(wlr, self.FLAGS.rms_decay)

        self.train = opt.minimize(self.loss_main, var_list=R_vars+O_vars)
        if self.FLAGS.reweight_sample or self.FLAGS.p_alpha>0:
            self.train_reweight = wopt.minimize(self.loss_reweight, var_list=W_vars)
        

