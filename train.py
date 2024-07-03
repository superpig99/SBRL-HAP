from sbrl_hap import sbrl_hap
from utils import *

import tensorflow.compat.v1 as tf
import numpy as np
import random
import datetime
import traceback
import os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Params
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('reweight_sample', 1, "decorrelateion reweighting")
tf.app.flags.DEFINE_float('p_alpha', 0.05, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0001, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('p_beta', 1., """coef of loss_w. """)
tf.app.flags.DEFINE_float('p_gamma1', 1., """coef of pred_last_hsic. """)
tf.app.flags.DEFINE_float('p_gamma2', 1., """coef of rep_last_hsic. """)
tf.app.flags.DEFINE_float('p_gamma3', 0.1, """coef of else_hsic. """)
tf.app.flags.DEFINE_integer('rep_dim', 128, "The dimension of representation network")
tf.app.flags.DEFINE_integer('rep_layer', 3, "The number of representation network layers")
tf.app.flags.DEFINE_integer('y_dim', 64, "The dimension of outcome network")
tf.app.flags.DEFINE_integer('y_layer', 3, "The number of outcome network layers")
tf.app.flags.DEFINE_float('lrate', 0.00001, """Learning rate. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.90, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_float('wlrate', 0.0002, """Learning rate. """)
tf.app.flags.DEFINE_float('wlrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('batch_size', 0, "batch_size")
tf.app.flags.DEFINE_integer('n_experiments', 10, "num_experiments")
tf.app.flags.DEFINE_integer('iterations', 3000, """Number of iterations. """)
tf.app.flags.DEFINE_string('activation', 'elu', "Activation function")
tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_float('rms_decay', 0.3, """RMSProp decay. """)
tf.app.flags.DEFINE_string('var_from', '', "get_variable/Variable")
tf.app.flags.DEFINE_integer('use_p_correction', 0, "fix coef")
tf.app.flags.DEFINE_integer('batch_norm', 1, "batch normalization")
tf.app.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('imb_fun', 'mmd_lin', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 10., """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
tf.app.flags.DEFINE_string('outdir', '/ossfs/workspace/SBRL-HAP/results/syn_i8_c8_a8_d2_n10000_biny/rp25_train/res3', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '/ossfs/workspace/SBRL-HAP/data/Syn_8_8_8_2_10000/syn_biny/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'rp25.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', 'rp25.test.npz~rn30.test.npz~rn25.test.npz~rn15.test.npz~rn13.test.npz~rp13.test.npz~rp15.test.npz~rp30.test.npz', """Test data filename form. """)
tf.app.flags.DEFINE_integer('y_is_binary', 1, "The outcome is binary")
tf.app.flags.DEFINE_integer('output_delay', 50, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
tf.app.flags.DEFINE_integer('seed', 888, "seed")


def run(net, sess, i_exp, D_exp, D_exp_test, I_valid, logfile):
    ''' Index '''
    n = D_exp['x'].shape[0];n_test = D_exp_test[0]['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train, n_valid = len(I_train), len(I_valid)

    ''' Compute treatment probability'''
    p_treated = np.mean(D_exp['t'][I_train,:])

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' feed_dict for evaluation '''
    train_dict = {net.x: D_exp['x'], net.t: D_exp['t'], net.y_: D_exp['yf'], \
                  net.do_in: 1.0, net.do_out: 1.0, net.p_t: p_treated, net.idx: I, net.n: n}
    if D_exp['ycf'] is not None:
        train_cf_dict = {net.x: D_exp['x'], net.t: D_exp['t'], net.ycf: D_exp['ycf'], \
                         net.do_in: 1.0, net.do_out: 1.0, net.p_t: p_treated, net.idx: I, net.n: n}

    ''' Set up for storing '''
    preds_train = []
    preds_test = {di: [] for di in range(len(D_exp_test))}
    sample_weights = []

    objnan, w_objnan = False, False
    ''' train for each batch '''
    for i in range(FLAGS.iterations):
        ''' Fetch sample for train '''
        batch_size = FLAGS.batch_size if FLAGS.batch_size else n_train
        I_batch = random.sample(range(0, n_train), batch_size)
        x_batch = D_exp['x'][I_train,:][I_batch,:]
        t_batch = D_exp['t'][I_train,:][I_batch]
        y_batch = D_exp['yf'][I_train,:][I_batch]

        batch_dict = {net.x: x_batch, net.t: t_batch, net.y_: y_batch, \
                      net.do_in: FLAGS.dropout_out, net.do_out: FLAGS.dropout_out, net.p_t: p_treated, net.idx: I_batch, net.n: len(I_batch)}

        ''' optimization '''
        if not objnan:
            sess.run(net.train, feed_dict=batch_dict)
            if (FLAGS.reweight_sample or FLAGS.p_alpha>0) and not w_objnan:
                batch_cov_dict = {net.x: x_batch, net.t: t_batch, net.y_: y_batch, net.do_in: 1.0, net.do_out: 1.0, net.p_t: p_treated, net.idx: I_batch, net.n:len(I_batch)}
                sess.run(net.train_reweight, feed_dict=batch_cov_dict)

        ''' eval '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            batch_loss_hsic=None
            if FLAGS.reweight_sample:
                loss_rew, loss_hsic, ipm, loss_w = sess.run([net.loss_reweight, net.loss_hsic, net.IPM_C, net.loss_w], feed_dict=batch_cov_dict)
            loss_main, loss_pred, train_y_hat, train_ycf_hat = sess.run([net.loss_main, net.loss_pred, net.y_pred, net.ycf_pred], feed_dict=train_dict)
            loss_pred_ycf = 0
            if D_exp['ycf'] is not None:
                loss_pred_ycf = sess.run(net.loss_YCF, feed_dict=train_cf_dict)
            sample_weights.append(sess.run(net.sample_weight, feed_dict={net.idx:I_train}))
            ''' save results '''
            preds_train.append(np.concatenate((train_y_hat, train_ycf_hat),axis=1))

            log(logfile, 'Iter:{:<6d}{:>15s}:{:.3f}{:>15s}:{:.3f}{:>15s}:{:.3f}{:>15s}:{:.3f}{:>15s}:{:.3f}{:>15s}:{:.3f}{:>15s}:{:.3f}'.format( \
                i, 'loss_main',loss_main, 'loss_pred',loss_pred, 'loss_pred_ycf',loss_pred_ycf, 'loss_rew',loss_rew, 'loss_hsic', loss_hsic, 'ipm',ipm, 'loss_w',loss_w))

            for dt_i in range(len(D_exp_test)):
                test_dict = {net.x: D_exp_test[dt_i]['x'], net.t: D_exp_test[dt_i]['t'], net.y_: D_exp_test[dt_i]['yf'], \
                             net.do_in: 1.0, net.do_out: 1.0, net.p_t: p_treated, net.idx: range(n_test), net.n: n_test}
                test_y_hat, test_ycf_hat = sess.run([net.y_pred, net.ycf_pred], feed_dict=test_dict)

                ''' save results '''
                preds_test[dt_i].append(np.concatenate((test_y_hat, test_ycf_hat),axis=1))

    ''' output of hidden layers '''
    test_pred_hidden = []
    for di in range(len(D_exp_test)):
        pred_hidden = sess.run(net.mus_Y, feed_dict={net.x: D_exp_test[di]['x'], net.t: D_exp_test[di]['t'], net.do_in: 1.0, net.do_out: 1.0})
        test_pred_hidden.append(pred_hidden[-2])

    rep_hidden, weight_in, pred_hidden, weight_pred = sess.run([net.rep_C, net.w_C, net.mus_Y, net.w_muY], \
                                                               feed_dict={net.x: D_exp['x'], net.t: D_exp['t'], net.do_in: 1.0, net.do_out: 1.0})

    net_dict= {'rep_hidden': rep_hidden, 'weight_in': weight_in, 'pred_hidden': pred_hidden[-2], 'weight_pred': weight_pred, 'test_pred_hidden': test_pred_hidden}

    return preds_train, preds_test, sample_weights, net_dict


def train(outdir):
    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()

    net_dir = outdir+'net_weight'
    os.mkdir(net_dir)

    ''' Save parameters '''
    save_config(outdir+'config.txt')

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Load data '''
    datapath = FLAGS.datadir + FLAGS.dataform
    dataname_test = FLAGS.data_test.split('~')
    datapath_test = [FLAGS.datadir + fi for fi in dataname_test]
    npzfile_test = [outdir+dtf[:-9]+'_result.test' for dtf in dataname_test]
    log(logfile, 'Training data: ' + datapath)
    log(logfile, 'Test data:     ' + ' '.join(dataname_test))

    D = load_data(datapath)
    D_test = [load_data(test_data) for test_data in datapath_test]
    n_exp_test = D_test[0]['x'].shape[-1]
    dims = [D['dim'], D['n']]

    ''' Start Session '''
    sess = tf.Session()

    ''' Define model graph '''
    net = sbrl_hap(FLAGS, dims)

    ''' Set up for saving variables '''
    all_weights = []
    all_preds_train = []
    all_preds_test = {di: [] for di in range(len(D_test))}
    all_valid = []

    ''' train for each experiment '''
    for i_exp in range(1,FLAGS.n_experiments+1):
        log(logfile, '\nExperiment %d:'%i_exp)
        ''' get batch data '''
        D_exp = {'HAVE_TRUTH': D['HAVE_TRUTH']}
        D_exp['x']  = D['x'][:,:,i_exp-1]
        D_exp['t']  = D['t'][:,i_exp-1:i_exp]
        D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
        if D['HAVE_TRUTH']:
            D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
        else:
            D_exp['ycf'] = None

        D_exp_test_list = []
        for dt_i in range(len(D_test)):
            D_exp_test = {'HAVE_TRUTH': D_test[dt_i]['HAVE_TRUTH']}
            D_exp_test['x']  = D_test[dt_i]['x'][:,:,(i_exp-1)%n_exp_test]
            D_exp_test['t']  = D_test[dt_i]['t'][:,(i_exp-1)%n_exp_test].reshape(-1,1)
            D_exp_test['yf'] = D_test[dt_i]['yf'][:,(i_exp-1)%n_exp_test].reshape(-1,1)
            if D_test[dt_i]['HAVE_TRUTH']:
                D_exp_test['ycf'] = D_test[dt_i]['ycf'][:,(i_exp-1)%n_exp_test].reshape(-1,1)
            else:
                D_exp_test['ycf'] = None
            D_exp_test_list.append(D_exp_test)

        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        preds_train, preds_test, sample_weights, net_dict = run(net, sess, i_exp, D_exp, D_exp_test_list, I_valid, logfile)

        ''' save params of net '''
        np.save('%s/exp%d.npy'%(net_dir,i_exp), net_dict)
        log(logfile, 'Saving result to %s...\n' % outdir)

        ''' Collect all preds '''
        all_preds_train.append(preds_train)
        for dt_i in range(len(D_test)):
            all_preds_test[dt_i].append(preds_test[dt_i])

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        for dt_i in range(len(D_test)):
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test[dt_i],1,3),0,2)
            np.savez(npzfile_test[dt_i], pred=out_preds_test)

        ''' Save results and predictions '''
        all_weights.append(sample_weights)
        all_valid.append(I_valid)
        np.savez(npzfile, pred=out_preds_train, val=np.array(all_valid), weights = np.array(all_weights))


def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/results_'+timestamp+'/'
    os.mkdir(outdir)

    try:
        train(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    tf.app.run()
