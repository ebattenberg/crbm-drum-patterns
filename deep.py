import cPickle as pkl
import pdb
import time
import datetime

import numpy as np
import pylab as pl
import scipy.stats
import scipy.special
from scipy.special import gamma
from scipy.misc import factorial

import gDeep as gpu
reload(gpu)


class RBM(object):
    """Restricted Boltzmann Machine (CRBM) using numpy """
    def __init__(self, Nv, Nh, vis_unit='binary', vis_scale=0,
            bv = None,
            dtype='float32'):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        input:
        -----------------

        Nv: number of visible units

        Nh: number of hidden units

        vis_unit: type of visible unit {'binary','linear'}
                ('linear' = rectified linear unit)

        vis_scale: maximum output value for linear visible units
                (average std_dev is ~= 1 at this scale, 
                so pre-scale training data with this in mind)

        W: weight between current hidden and visible units (undirected)
            [Nv x Nh]

        bh: hidden bias

        bv: visible bias
        """

        if vis_unit not in ['binary','linear']:
            raise ValueError, 'Unknown visible unit type %s' % vis_unit
        if vis_unit == 'linear':
            if not vis_scale > 0:
                raise ValueError, 'Linear unit scale must be >= 0'
        if vis_unit == 'binary':
            vis_scale = 1.


        # W is initialized with `initial_W` which is uniformly sampled
        # from -4.*sqrt(6./(Nv+Nh)) and 4.*sqrt(6./(Nh+Nv))
        # the output of uniform if converted using asarray to dtype
        W = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv+Nh)),
            high  =  4*np.sqrt(6./(Nv+Nh)),
            size  =  (Nv, Nh)),
            dtype =  dtype)

        bh = np.zeros(Nh,dtype=dtype)

        if bv is None :
            bv = np.zeros(Nv,dtype=dtype)
        else:
            bv = bv.copy()

        # params -------------------------------------------
        self.dtype  = dtype 

        self.Nv     = Nv        # num visible units
        self.Nh     = Nh        # num hidden units

        self.vis_unit = vis_unit    # type of visible output unit
        self.vis_scale = vis_scale  # scale of linear output units

        self.W      = W     # vis<->hid weights
        self.bv     = bv    # vis bias
        self.bh     = bh    # hid bias

        self.W_update = np.zeros((Nv,Nh),dtype=dtype)
        self.bh_update = np.zeros((Nh,),dtype=dtype)
        self.bv_update = np.zeros((Nv,),dtype=dtype)

        self.params     = [ 'dtype',
                            'vis_unit','vis_scale',
                            'Nv','Nh',
                            'W','bh','bv']

    def save_params(self,filename=None):
        '''save parameters to file'''
        if filename is None:
            id = np.random.randint(100000)
            filename = 'RBM_%u.pkl' % id
            
        params_out = {}  
        for p in self.params:
            params_out[p] = vars(self)[p]
        fp = open(filename,'wb')
        pkl.dump(params_out,fp,protocol=-1)
        fp.close()

        print 'saved %s' % filename

    def load_params(self,filename):
        '''load parameters from file'''
        fp = open(filename,'rb')
        params_in = pkl.load(fp)
        fp.close()
        for key,value in params_in.iteritems():
            vars(self)[key] = value

        Nv,Nh = self.Nv,self.Nh
        dtype = self.dtype

        self.W_update = np.zeros((Nv,Nh),dtype=dtype)
        self.bh_update = np.zeros((Nh,),dtype=dtype)
        self.bv_update = np.zeros((Nv,),dtype=dtype)

    def return_params(self):
        '''
        return a formatted string containing scalar parameters
        '''
        output = 'Nv=%u, Nh=%u, vis_unit=%s, vis_scale=%0.2f' \
                % (self.Nv,self.Nh,self.vis_unit,self.vis_scale)
        return output

    def mean_field_h_given_v(self,v):
        '''compute mean-field reconstruction of P(h=1|v)'''
        prob =  sigmoid(self.bh + np.dot(v, self.W))
        return prob

    def mean_field_v_given_h(self,h):
        '''compute mean-field reconstruction of P(v|h)'''
        x = self.bv + np.dot(h, self.W.T)
        if self.vis_unit == 'binary':
            return sigmoid(x)
        elif self.vis_unit == 'linear':
            return log_1_plus_exp(x) - log_1_plus_exp(x-self.vis_scale)
        return prob

    def sample_h_given_v(self,v):
        '''compute samples from P(h|v)'''
        prob = self.mean_field_h_given_v(v)
        samples = (np.random.uniform(size=prob.shape) < prob).astype(self.dtype)

        return samples, prob

    def sample_v_given_h(self,h):
        '''compute samples from P(v|h)'''
        if self.vis_unit == 'binary':
            mean = self.mean_field_v_given_h(h)
            samples = (np.random.uniform(size=mean.shape) < mean).astype(self.dtype)
            return samples, mean

        elif self.vis_unit == 'linear':
            x = self.bv + np.dot(h, self.W.T)
            # variance of noise is sigmoid(x) - sigmoid(x - vis_scale)
            stddev = np.sqrt(sigmoid(x) - sigmoid(x - self.vis_scale)) 
            mean =  log_1_plus_exp(x) - log_1_plus_exp(x-self.vis_scale)
            noise = stddev * np.random.standard_normal(size=x.shape)
            samples = np.fmax(0,np.fmin(self.vis_scale, mean + noise))
            return samples, mean

    def cdk(self,K,v0_data,rate=0.001,momentum=0.0,weight_decay=0.001,noisy=0):
        '''
        compute K-step contrastive divergence update

        input:

        K - number of gibbs iterations (for cd-K)
        v0_data - training data [N x (Nv+Nl)]
        rate - learning rate
        momentum - learning momentum
        weight_decay - L2 regularizer
        noisy - 0 = use h0_mean, use visible means everywhere
                1 = use h0_samp, use visible means everywhere
                2 = use samples everywhere
        '''



        # collect gradient statistics
        h0_samp,h0_mean = self.sample_h_given_v(v0_data)
        hk_samp = h0_samp

        if noisy == 0:
            for k in xrange(K):  # vk_mean <--> hk_samp
                vk_mean = self.mean_field_v_given_h(hk_samp)
                hk_samp, hk_mean = self.sample_h_given_v(vk_mean)
            h0 = h0_mean 
            vk = vk_mean
            hk = hk_mean
        elif noisy == 1:
            for k in xrange(K):  # vk_mean <--> hk_samp 
                vk_mean = self.mean_field_v_given_h(hk_samp)
                hk_samp, hk_mean = self.sample_h_given_v(vk_mean)
            h0 = h0_samp # <--
            vk = vk_mean
            hk = hk_mean
        elif noisy == 2:
            for k in xrange(K): # vk_samp <--> hk_samp
                vk_samp, vk_mean = self.sample_v_given_h(hk_samp)
                hk_samp, hk_mean = self.sample_h_given_v(vk_samp)
            h0 = h0_samp
            vk = vk_samp # <--
            hk = hk_samp # <--


        W_grad,bv_grad,bh_grad = self.compute_gradients(v0_data,h0,vk,hk)

        if weight_decay > 0.0:
            W_grad += weight_decay * self.W


        if momentum > 0.0:
            self.W_update = momentum * self.W_update - rate*W_grad
            self.bh_update = momentum * self.bh_update - rate*bh_grad
            self.bv_update = momentum * self.bv_update - rate*bv_grad
        else:
            self.W_update = -rate*W_grad
            self.bh_update = -rate*bh_grad
            self.bv_update = -rate*bv_grad

        self.W = self.W + self.W_update
        self.bh = self.bh + self.bh_update
        self.bv = self.bv + self.bv_update

    def compute_gradients(self,v0,h0,vk,hk):
        N = v0.shape[0]
        N_inv = 1./N
        W_grad = N_inv * (np.dot(vk.T, hk) - np.dot(v0.T, h0))
        bv_grad = np.mean(vk - v0,axis=0)
        bh_grad = np.mean(hk - h0,axis=0)

        return W_grad,bv_grad,bh_grad

    def gibbs_samples(self,K,v0_data,noisy=0):
        '''
        compute a visible unit sample using Gibbs sampling

        input:
        K - number of complete Gibbs iterations
        v_input - seed value of visible units
        noisy - 0 = always use visible means and use hidden means to drive final sample
                1 = drive final sample with final hidden sample
                2 = use visible means for updates but use visible and hidden samples for final update
                3 = always use samples for both visible and hidden updates
                note: hidden samples are always used to drive visible reconstructions unless noted otherwise
        '''

        Nv = self.Nv


        h0_samp,h0_mean = self.sample_h_given_v(v0_data)

        hk_samp = h0_samp
        hk_mean = h0_mean
        if noisy < 3:
            for k in xrange(K-1): # hk_samp <--> vk_mean
                vk_mean = self.mean_field_v_given_h(hk_samp)
                hk_samp, hk_mean = self.sample_h_given_v(vk_mean)
        else:
            for k in xrange(K-1): # hk_samp <--> vk_samp
                vk_samp, vk_mean = self.sample_v_given_h(hk_samp)
                hk_samp, hk_mean = self.sample_h_given_v(vk_samp)

        if noisy == 0:  # hk_mean --> v_mean
            v_mean = self.mean_field_v_given_h(hk_mean)
            return v_mean
        elif noisy == 1: # hk_samp --> v_mean
            v_mean = self.mean_field_v_given_h(hk_samp)
            return v_mean
        elif noisy > 1:  # hk_samp --> v_samp
            v_samp, v_mean = self.sample_v_given_h(hk_samp)
            return v_samp

    def recon_error(self, v0_data,K=1,print_output=False):
        '''compute K-step reconstruction error'''

        vk_mean = self.gibbs_samples(K,v0_data,noisy=0)
        recon_error = np.mean(np.abs(v0_data - vk_mean))

        if print_output:
            output = '%30s %6.5f' % ('vis error:', recon_error/self.vis_scale)
            print output
            return output
        else: 
            return recon_error

    def update_stats(self):
        W_stats = [np.min(self.W),np.mean(np.abs(self.W)),np.max(self.W)]
        bh_stats = [np.min(self.bh),np.mean(np.abs(self.bh)),np.max(self.bh)]
        bv_stats = [np.min(self.bv),np.mean(np.abs(self.bv)),np.max(self.bv)]

        W_update_stats = [np.min(self.W_update), np.mean(np.abs(self.W_update)), np.max(self.W_update)]
        bh_update_stats = [np.min(self.bh_update), np.mean(np.abs(self.bh_update)), np.max(self.bh_update)]
        bv_update_stats = [np.min(self.bv_update), np.mean(np.abs(self.bv_update)), np.max(self.bv_update)]

        param_stats = dict(W=W_stats,bh=bh_stats,bv=bv_stats)
        update_stats = dict(W=W_update_stats,
                bh=bh_update_stats,bv=bv_update_stats)

        return [param_stats, update_stats]

class LRBM(RBM):
    '''
    Labeled Restricted Boltzmann Machine
    '''
    def __init__(self, Nv, Nh, Nl, vis_unit='binary', vis_scale=0,
            bv = None, 
            dtype='float32'):
        """
        input:
        -----------------

        (in addition to those defined in RBM class)

        Nl: number of label units (group of softmax units)
        """

        super(LRBM,self).__init__(Nv, Nh,vis_unit, vis_scale,
            bv, dtype='float32')


        # add label units to visible units

        # W is initialized with uniformly sampled data
        # from -4.*sqrt(6./(Nv+Nh)) and 4.*sqrt(6./(Nh+Nv))
        W = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv+Nl+Nh)),
            high  =  4*np.sqrt(6./(Nv+Nl+Nh)),
            size  =  (Nv+Nl, Nh)),
            dtype =  dtype)


        if bv is None :
            bv = np.zeros((Nv+Nl),dtype=dtype)
        else:
            bv = bv.copy()


        # new label-unit params -------------------------------------------
        self.Nl     = Nl        # num label units

        self.W      = W     # (vis+lab)<->hid weights
        self.bv     = bv    # vis bias

        self.W_update = np.zeros((Nv+Nl,Nh),dtype=dtype)
        self.bv_update = np.zeros((Nv+Nl,),dtype=dtype)

        self.params     += ['Nl']

    def load_params(self,filename):
        '''load parameters from file'''
        super(LRBM,self).load_params(filename)

        Nv,Nh,Nl,= self.Nv,self.Nh,self.Nl
        dtype = self.dtype
        self.W_update = np.zeros((Nv+Nl,Nh),dtype=dtype)
        self.bv_update = np.zeros((Nv+Nl,),dtype=dtype)

    def save_params(self,filename=None):
        '''save parameters to file'''
        if filename is None:
            id = np.random.randint(100000)
            filename = 'LRBM_%u.pkl' % id
        super(LRBM,self).save_params(filename)

    def return_params(self):
        '''
        return a formatted string containing scalar parameters
        '''

        output = super(LRBM,self).return_params()
        output = 'Nl=%u, ' % (self.Nl) + output
        return output

    def separate_vis_lab(self,x,axis=1):
        '''
        separate visible unit data from label unit data
        '''

        Nl = self.Nl

        if x.ndim == 1:
            axis = 0

        if axis == 0:
            x_lab = x[-Nl:]
            x_vis = x[:-Nl]
        elif axis == 1:
            x_lab = x[:,-Nl:]
            x_vis = x[:,:-Nl]

        return x_vis, x_lab

    def join_vis_lab(self,x_vis,x_lab,axis=1):
        '''
        join visible unit data to label unit data
        '''
        if x_vis.ndim == 1:
            axis = 0

        x = np.concatenate((x_vis,x_lab),axis=axis)

        return x

    def mean_field_v_given_h(self,h):
        '''compute mean-field reconstruction of P(v|h)'''
        x = self.bv + np.dot(h, self.W.T)
        x_vis, x_lab = self.separate_vis_lab(x)
        lab_mean = softmax(x_lab)

        if self.vis_unit == 'binary':
            vis_mean = sigmoid(x_vis)
        elif self.vis_unit == 'linear':
            vis_mean = log_1_plus_exp(x_vis) - log_1_plus_exp(x_vis-self.vis_scale)

        means = self.join_vis_lab(vis_mean,lab_mean)
        return means

    def sample_v_given_h(self,h):
        '''compute samples from P(v|h)'''
        if self.vis_unit == 'binary':
            means = self.mean_field_v_given_h(h)
            vis_mean,lab_mean = self.separate_vis_lab(means)
            vis_samp = (np.random.uniform(size=vis_mean.shape) < vis_mean).astype(self.dtype)

        elif self.vis_unit == 'linear':
            x = self.bv + np.dot(h, self.W.T)
            x_vis, x_lab = self.separate_vis_lab(x)
            # variance of noise is sigmoid(x_vis) - sigmoid(x_vis - vis_scale)
            vis_stddev = np.sqrt(sigmoid(x_vis) - sigmoid(x_vis - self.vis_scale)) 
            vis_mean =  log_1_plus_exp(x_vis) - log_1_plus_exp(x_vis-self.vis_scale)
            vis_noise = stddev * np.random.standard_normal(size=x.shape)
            vis_samp = np.fmax(0,np.fmin(self.vis_scale, vis_mean + vis_noise))

            lab_mean = softmax(x_lab)
            means = self.join_vis_lab(vis_mean,lab_mean)

        lab_samp = sample_categorical(lab_mean)
        samples = self.join_vis_lab(vis_samp,lab_samp)

        return samples, means

    def label_probabilities(self,v_input,output_h=False):
        '''
        compute the activation probability of each label unit given the visible units
        '''

        #compute free energy for each label configuration
        # F(v,c) = -sum(v*bv) - bl[c] - sum(log(1 + exp(z_c)))
        # where z_c = bh + dot(v,W) + r[c]  (r[c] are the weights for label c)
        # also,  v_input = [v,l],  where l are binary "one-hot" labels

        b_hid = self.bh
        b_vis, b_lab = self.separate_vis_lab(self.bv)

        v_vis, v_lab = self.separate_vis_lab(v_input)
        W_vis,W_lab = self.separate_vis_lab(self.W,axis=0)


        # the b_vis term cancels out in the softmax
        #F = -np.sum(v_vis*b_vis,axis=1) 
        #F = F.reshape((-1,1)) - b_lab
        F =  - b_lab

        z = b_hid + np.dot(v_vis,W_vis)
        z = z.reshape(z.shape + (1,))
        z = z + W_lab.T.reshape((1,) + W_lab.T.shape)

        hidden_terms = -np.sum(log_1_plus_exp(z), axis=1)
        F = F + hidden_terms

        pr = softmax(-F)

        # compute hidden probs for each label configuration
        # this is used in the discriminative updates
        if output_h:
            h = sigmoid(z)
            return pr, h
        else:
            return pr

    def discriminative_train(self,v_input,rate=0.001,momentum=0.0,weight_decay=0.001):
        ''' 
        Update weights using discriminative updates.
        These updates use gradient ascent of the log-likelihood of the 
        label probability of the correct label

        input:
        v_input - [v_past, v_visible, v_labels]
            (v_labels contains the binary activation of the correct label)
        '''

        N = v_input.shape[0]


        # things to compute:
        # h_d - hidden unit activations for each label configuration
        # p_d - label unit probabilities
        p_d, h_d = self.label_probabilities(v_input,output_h=True)

        v_vis,v_lab = self.separate_vis_lab(v_input)

        ind, true_labs = np.where(v_lab == 1)


        scale = rate / N


        # prob_scale = (1-p_d) for correct label and -p_d for other labels
        prob_scale = -p_d
        prob_scale[ind,true_labs] += 1
        ps_broad = prob_scale.reshape((N,1,self.Nl)) # make broadcastable across h_d

        p_h_sum = np.sum(ps_broad * h_d, axis=2)

        # compute gradients ----------------------------------------------
        # W = [w,r]
        w_grad = np.dot(v_vis.T, p_h_sum)               # vis<-->hid
        r_grad = np.sum( ps_broad * h_d, axis=0 ).T     # lab<-->hid
        W_grad = self.join_vis_lab(w_grad,r_grad,axis=0)# [vis,lab]<-->hid

        bh_grad = np.sum(p_h_sum,axis=0)                # -->hid

        # bv = [bvv,bvl]                                # -->[vis,lab]
        bvv,bvl = self.separate_vis_lab(self.bv)
        bvv_grad = np.zeros_like(bvv)                   # -->vis
        bvl_grad = np.sum(prob_scale,axis=0)            # -->lab
        # ---------------------------------------------------------------


        if weight_decay > 0.0:
            W_grad += -weight_decay * self.W

        #Wv_grad = self.join_vis_lab(Wvv_grad,Wvl_grad)
        bv_grad = self.join_vis_lab(bvv_grad,bvl_grad)


        if momentum > 0.0:
            self.W_update = momentum * self.W_update + scale*W_grad
            self.bh_update = momentum * self.bh_update + scale*bh_grad
            self.bv_update = momentum * self.bv_update + scale*bv_grad
        else:
            self.W_update = scale*W_grad
            self.bh_update = scale*bh_grad
            self.bv_update = scale*bv_grad

        self.W += self.W_update
        self.bh += self.bh_update
        self.bv += self.bv_update

    def recon_error(self,v0_data,K=1,print_output=False):
        '''compute K-step reconstruction error'''

        vk_mean = self.gibbs_samples(K,v0_data,noisy=0)

        v0_vis,v0_lab = self.separate_vis_lab(v0_data)
        vk_vis,vk_lab = self.separate_vis_lab(vk_mean)
        vis_error = np.mean(np.abs(v0_vis - vk_vis))
        lab_error = np.mean(np.abs(v0_lab - vk_lab))

        lab_probs = self.label_probabilities(v0_data)
        pred_labs = np.argmax(lab_probs,axis=1)
        ind, true_labs = np.where(v0_lab == 1)
        percent_correct = np.mean(pred_labs == true_labs)
        prob_error = np.mean(np.abs(1. - lab_probs[ind,true_labs]))

        if print_output:
            output = '%30s %6.5f' % ('vis error:', vis_error/self.vis_scale) + '\n'
            output += '%30s %6.5f' % ('lab error:', lab_error) + '\n'
            output += '%30s %6.5f' % ('prob error:', prob_error) + '\n'
            output += '%30s %6.5f' % ('class correct:', percent_correct)
            print output
            return output
        else: 
            return percent_correct, prob_error, lab_error, vis_error/self.vis_scale

class CRBM(object):
    """Conditional Restricted Boltzmann Machine (CRBM) using numpy """
    def __init__(self, Nv, Nh, Tv, Th, period=None, vis_unit='binary', vis_scale=0,
            bv = None, Wv_scale = 0.1, dtype='float32'):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        input:
        -----------------

        Nv: number of visible units

        Nh: number of hidden units

        Tv: order of autoregressive weights (RBM has Tv=0)
                (how far into the past do they go?)

        Th: order of past visible to current hidden weights (RBM has Th=0)
                (how far into the past do they go?)

        period: natural repetition period of data [default=Tv]
                (for initializing generative gibbs sampling)

        vis_unit: type of visible unit {'binary','linear'}
                ('linear' = rectified linear unit)

        vis_scale: maximum output value for linear visible units
                (average std_dev is ~= 1 at this scale, 
                so pre-scale training data with this in mind)

        bv: visible bias

        Wv_scale - how much to rescale Wv updates

        other params:
        --------------------

        W: weight between current hidden and visible units (undirected)
            [Nv x Nh]

        Wh: past visible to current hidden weights (directed)
            [Tv*Nv x Nh]

        Wv: past visible to current visible weights (directed)
            [Tv*Nv x Nv]

        bh: hidden bias

        """

        if vis_unit not in ['binary','linear']:
            raise ValueError, 'Unknown visible unit type %s' % vis_unit
        if vis_unit == 'linear':
            if not vis_scale > 0:
                raise ValueError, 'Linear unit scale must be >= 0'
        if vis_unit == 'binary':
            vis_scale = 1.

        T = max(Tv,Th)

        if period is None:
            period = T
        else:
            if period > T:
                raise ValueError, 'period must be <= max(Tv,Th)'


        # W is initialized with `initial_W` which is uniformly sampled
        # from -4.*sqrt(6./(Nv+Nh)) and 4.*sqrt(6./(Nh+Nv))
        # the output of uniform if converted using asarray to dtype
        W = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv+Nh)),
            high  =  4*np.sqrt(6./(Nv+Nh)),
            size  =  (Nv, Nh)),
            dtype =  dtype)

        Wv = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv*Tv+Nv)),
            high  =  4*np.sqrt(6./(Nv*Tv+Nv)),
            size  =  (Nv*Tv, Nv)),
            dtype =  dtype)

        Wh = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv*Th+Nh)),
            high  =  4*np.sqrt(6./(Nv*Th+Nh)),
            size  =  (Nv*Th, Nh)),
            dtype =  dtype)


        bh = np.zeros(Nh,dtype=dtype)

        if bv is None :
            bv = np.zeros(Nv,dtype=dtype)
        else:
            bh = bh.copy()

        # params -------------------------------------------
        self.dtype  = dtype 

        self.Nv     = Nv        # num visible units
        self.Nh     = Nh        # num hidden units
        self.Tv     = Tv        # num vis->vis delay taps
        self.Th     = Th        # num vis->hid delay taps
        self.T      = T         # max(Tv,Th)

        self.period = period        # typical repetition period of sequences
        self.vis_unit = vis_unit    # type of visible output unit
        self.vis_scale = vis_scale  # scale of linear output units

        self.W      = W     # vis<->hid weights
        self.Wv     = Wv    # vis->vis delay weights
        self.Wh     = Wh    # vis->hid delay weights
        self.bv     = bv    # vis bias
        self.bh     = bh    # hid bias

        self.Wv_scale = Wv_scale # rescale Wv updates

        self.W_update = np.zeros((Nv,Nh),dtype=dtype)
        self.Wv_update = np.zeros((Nv*Tv,Nv),dtype=dtype)
        self.Wh_update = np.zeros((Nv*Th,Nh),dtype=dtype)
        self.bh_update = np.zeros((Nh,),dtype=dtype)
        self.bv_update = np.zeros((Nv,),dtype=dtype)

        self.params     = [ 'dtype',
                            'period','vis_unit','vis_scale',
                            'Nv','Nh','Tv','Th','T',
                            'W','Wv','Wh','bh','bv']

    def save_params(self,filename=None):
        '''save parameters to file'''
        if filename is None:
            id = np.random.randint(100000)
            filename = 'CRBM_%u.pkl' % id
            
        params_out = {}  
        for p in self.params:
            params_out[p] = vars(self)[p]
        fp = open(filename,'wb')
        pkl.dump(params_out,fp,protocol=-1)
        fp.close()

        print 'saved %s' % filename

    def load_params(self,filename):
        '''load parameters from file'''
        fp = open(filename,'rb')
        params_in = pkl.load(fp)
        fp.close()
        for key,value in params_in.iteritems():
            vars(self)[key] = value

        Nv,Nh,Tv,Th = self.Nv,self.Nh,self.Tv,self.Th
        dtype = self.dtype

        self.W_update = np.zeros((Nv,Nh),dtype=dtype)
        self.Wv_update = np.zeros((Nv*Tv,Nv),dtype=dtype)
        self.Wh_update = np.zeros((Nv*Th,Nh),dtype=dtype)
        self.bh_update = np.zeros((Nh,),dtype=dtype)
        self.bv_update = np.zeros((Nv,),dtype=dtype)

    def return_params(self):
        '''
        return a formatted string containing scalar parameters
        '''
        output = 'Nv=%u, Nh=%u, vis_unit=%s, vis_scale=%0.2f, Tv=%u, Th=%u, Wv_scale=%g' \
                % (self.Nv,self.Nh,self.vis_unit,self.vis_scale,self.Tv,self.Th,self.Wv_scale)
        return output

    def extract_data(self,v_input):
        Nv = self.Nv
        Tv = self.Tv
        Th = self.Th

        if v_input.ndim == 1:
            v_data = v_input[-Nv:]
            vv_past = v_input[-Nv*(1+Tv):-Nv]
            vh_past = v_input[-Nv*(1+Th):-Nv]
        else:
            v_data = v_input[:,-Nv:]
            vv_past = v_input[:,-Nv*(1+Tv):-Nv]
            vh_past = v_input[:,-Nv*(1+Th):-Nv]

        return v_data, vv_past, vh_past

    def mean_field_h_given_v(self,v,h_bias):
        '''compute mean-field reconstruction of P(ht=1|vt,v<t)'''
        prob =  sigmoid(h_bias + np.dot(v, self.W))
        return prob

    def mean_field_h_given_v_frame(self,v_input):
        '''
        compute mean-field reconstruction of P(ht=1|vt,v<t) 
        and compute h_bias from data

        input:
        v_frames - contains [v_past, v_curr] in a matrix
        '''
        v,vv_past,vh_past = self.extract_data(v_input)
        h_bias = self.bh + np.dot(vh_past,self.Wh)
        return sigmoid(h_bias + np.dot(v, self.W))

    def mean_field_v_given_h(self,h,v_bias):
        '''compute mean-field reconstruction of P(vt|ht,v<t)'''
        x = v_bias + np.dot(h, self.W.T)
        if self.vis_unit == 'binary':
            return sigmoid(x)
        elif self.vis_unit == 'linear':
            return log_1_plus_exp(x) - log_1_plus_exp(x-self.vis_scale)
        return prob

    def sample_h_given_v(self,v,h_bias):
        '''compute samples from P(ht=1|vt,v<t)'''
        prob = self.mean_field_h_given_v(v,h_bias)
        samples = (np.random.uniform(size=prob.shape) < prob).astype(self.dtype)

        return samples, prob

    def sample_v_given_h(self,h,v_bias):
        '''compute samples from P(vt|ht,v<t)'''
        if self.vis_unit == 'binary':
            mean = self.mean_field_v_given_h(h,v_bias)
            samples = (np.random.uniform(size=mean.shape) < mean).astype(self.dtype)
            return samples, mean

        elif self.vis_unit == 'linear':
            x = v_bias + np.dot(h, self.W.T)
            # variance of noise is sigmoid(x) - sigmoid(x - vis_scale)
            stddev = np.sqrt(sigmoid(x) - sigmoid(x - self.vis_scale)) 
            mean =  log_1_plus_exp(x) - log_1_plus_exp(x-self.vis_scale)
            noise = stddev * np.random.standard_normal(size=x.shape)
            samples = np.fmax(0,np.fmin(self.vis_scale, mean + noise))
            return samples, mean

    def cdk(self,K,v_input,rate=0.001,momentum=0.0,weight_decay=0.001,noisy=0):
        '''
        compute K-step contrastive divergence update

        input:

        K - number of gibbs iterations (for cd-K)
        v_input - contains [v_past, v0_data] = [(N x Nv*max(Tv,Th)), (N x Nv)]
        rate - learning rate
        momentum - learning momentum
        weight_decay - L2 regularizer
        noisy - 0 = use h0_mean, use visible means everywhere
                1 = use h0_samp, use visible means everywhere
                2 = use samples everywhere
        '''



        # compute gradient statistics
        v0_data,vv_past,vh_past = self.extract_data(v_input) 

        v_bias,h_bias = self.compute_dynamic_bias(v_input)

        h0_samp,h0_mean = self.sample_h_given_v(v0_data,h_bias)
        hk_samp = h0_samp

        if noisy == 0:
            for k in xrange(K):  # vk_mean <--> hk_samp
                vk_mean = self.mean_field_v_given_h(hk_samp,v_bias)
                hk_samp, hk_mean = self.sample_h_given_v(vk_mean,h_bias)
            h0 = h0_mean 
            vk = vk_mean
            hk = hk_mean
        elif noisy == 1:
            for k in xrange(K):  # vk_mean <--> hk_samp 
                vk_mean = self.mean_field_v_given_h(hk_samp,v_bias)
                hk_samp, hk_mean = self.sample_h_given_v(vk_mean,h_bias)
            h0 = h0_samp # <--
            vk = vk_mean
            hk = hk_mean
        elif noisy == 2:
            for k in xrange(K): # vk_samp <--> hk_samp
                vk_samp, vk_mean = self.sample_v_given_h(hk_samp,v_bias)
                hk_samp, hk_mean = self.sample_h_given_v(vk_samp,h_bias)
            h0 = h0_samp
            vk = vk_samp # <--
            hk = hk_samp # <--


        # compute gradients
        W_grad,Wv_grad,Wh_grad,bv_grad,bh_grad = self.compute_gradients(v_input,h0,vk,hk)

        if weight_decay > 0.0:
            W_grad += weight_decay * self.W
            Wv_grad += weight_decay * self.Wv
            Wh_grad += weight_decay * self.Wh

        if momentum > 0.0:
            self.W_update = momentum * self.W_update - rate*W_grad
            self.Wv_update = momentum * self.Wv_update - self.Wv_scale*rate*Wv_grad
            self.Wh_update = momentum * self.Wh_update - rate*Wh_grad
            self.bh_update = momentum * self.bh_update - rate*bh_grad
            self.bv_update = momentum * self.bv_update - rate*bv_grad
        else:
            self.W_update = -rate*W_grad
            self.Wv_update = -self.Wv_scale*rate*Wv_grad
            self.Wh_update = -rate*Wh_grad
            self.bh_update = -rate*bh_grad
            self.bv_update = -rate*bv_grad

        self.W = self.W + self.W_update
        self.Wv = self.Wv + self.Wv_update
        self.Wh = self.Wh + self.Wh_update
        self.bh = self.bh + self.bh_update
        self.bv = self.bv + self.bv_update

    def compute_gradients(self,v_input,h0,vk,hk):
        v0,vv_past,vh_past = self.extract_data(v_input) 

        N = v0.shape[0]
        N_inv = 1./N

        W_grad = N_inv * (np.dot(vk.T, hk) - np.dot(v0.T, h0))
        Wv_grad = N_inv * (np.dot(vv_past.T, vk) - np.dot(vv_past.T, v0))
        Wh_grad = N_inv * (np.dot(vh_past.T, hk) - np.dot(vh_past.T, h0))

        bv_grad = np.mean(vk - v0,axis=0)
        bh_grad = np.mean(hk - h0,axis=0)

        return W_grad,Wv_grad,Wh_grad,bv_grad,bh_grad

    def compute_dynamic_bias(self,v_input):
        v_data,vv_past,vh_past = self.extract_data(v_input)
        v_bias = self.bv + np.dot(vv_past,self.Wv)
        h_bias = self.bh + np.dot(vh_past,self.Wh)

        return v_bias, h_bias

    def generate(self,seed,num_steps,K,noisy=False):
        '''
        generate a sequence of length num_steps given the seed sequence

        input:
        seed - Nv dimensional sequence of length >= max(Tv,Th)
                flattened using row-major ordering 
                (units in same time step nearest each other)
        num_steps - number of sequence steps to generate
        K - number of gibbs iterations per sample
        noisy - noise level of gibbs samples [0,1,2,3] (see gibbs_samples() method)

        output:
        sequence - Nv dimensional sequence of length num_steps + seed length
        '''

        T = max(self.Tv,self.Th)
        Nv = self.Nv
        frame_size = Nv * T
        hop_size = Nv
        period_size = Nv * self.period

        if len(seed) < frame_size:
            raise ValueError, 'Seed not long enough'

        sequence = np.concatenate( (seed, np.zeros(num_steps * Nv))).astype('float32')

        idx = len(seed) - frame_size

        while idx+frame_size+Nv <= len(sequence):
            v_input = sequence[idx:idx+frame_size+Nv]
            # use samples from one period ago as starting point for Gibbs sampling
            v_input[-Nv:] = v_input[-period_size-Nv:-period_size]


            v_curr = self.gibbs_samples(K,v_input,noisy)
            sequence[idx+frame_size:idx+frame_size+Nv] = v_curr
            
            idx += hop_size

        return sequence

    def gibbs_samples(self,K,v_input,noisy=0):
        '''
        compute a visible unit sample using Gibbs sampling

        input:
        K - number of complete Gibbs iterations
        v_input - [v_past, v_curr_seed] array flattened using row-major ordering
                    * v_past of length Nv*max(Tv,Th) 
                    * v_curr_seed of length Nv
        noisy - 0 = always use visible means and use hidden means to drive final sample
                1 = drive final sample with final hidden sample
                2 = use visible means for updates but use visible and hidden samples for final update
                3 = always use samples for both visible and hidden updates
                note: hidden samples are always used to drive visible reconstructions unless noted otherwise
        '''

        Nv = self.Nv

        v0_data,vv_past,vh_past = self.extract_data(v_input)

        v_bias,h_bias = self.compute_dynamic_bias(v_input)

        h0_samp,h0_mean = self.sample_h_given_v(v0_data,h_bias)

        hk_samp = h0_samp
        hk_mean = h0_mean
        if noisy < 3:
            for k in xrange(K-1): # hk_samp <--> vk_mean
                vk_mean = self.mean_field_v_given_h(hk_samp,v_bias)
                hk_samp, hk_mean = self.sample_h_given_v(vk_mean,h_bias)
        else:
            for k in xrange(K-1): # hk_samp <--> vk_samp
                vk_samp, vk_mean = self.sample_v_given_h(hk_samp,v_bias)
                hk_samp, hk_mean = self.sample_h_given_v(vk_samp,h_bias)

        if noisy == 0:  # hk_mean --> v_mean
            v_mean = self.mean_field_v_given_h(hk_mean,v_bias)
            return v_mean
        elif noisy == 1: # hk_samp --> v_mean
            v_mean = self.mean_field_v_given_h(hk_samp,v_bias)
            return v_mean
        elif noisy > 1:  # hk_samp --> v_samp
            v_samp, v_mean = self.sample_v_given_h(hk_samp,v_bias)
            return v_samp

    def recon_error(self, v_input,K=1,print_output=False):
        '''compute K-step reconstruction error'''

        v0_data,vv_past,vh_past = self.extract_data(v_input)
        vk_mean = self.gibbs_samples(K,v_input,noisy=0)
        recon_error = np.mean(np.abs(v0_data - vk_mean))

        if print_output:
            output = '%30s %6.5f' % ('vis error:', recon_error/self.vis_scale)
            print output
            return output
        else: 
            return recon_error

    def update_stats(self):
        W_stats = [np.min(self.W),np.mean(np.abs(self.W)),np.max(self.W)]
        Wv_stats = [np.min(self.Wv),np.mean(np.abs(self.Wv)),np.max(self.Wv)]
        Wh_stats = [np.min(self.Wh),np.mean(np.abs(self.Wh)),np.max(self.Wh)]
        bh_stats = [np.min(self.bh),np.mean(np.abs(self.bh)),np.max(self.bh)]
        bv_stats = [np.min(self.bv),np.mean(np.abs(self.bv)),np.max(self.bv)]

        W_update_stats = [np.min(self.W_update), np.mean(np.abs(self.W_update)), np.max(self.W_update)]
        Wv_update_stats = [np.min(self.Wv_update), np.mean(np.abs(self.Wv_update)), np.max(self.Wv_update)]
        Wh_update_stats = [np.min(self.Wh_update), np.mean(np.abs(self.Wh_update)), np.max(self.Wh_update)]
        bh_update_stats = [np.min(self.bh_update), np.mean(np.abs(self.bh_update)), np.max(self.bh_update)]
        bv_update_stats = [np.min(self.bv_update), np.mean(np.abs(self.bv_update)), np.max(self.bv_update)]

        param_stats = dict(W=W_stats,Wv=Wv_stats,Wh=Wh_stats,bh=bh_stats,bv=bv_stats)
        update_stats = dict(W=W_update_stats,
                Wv=Wv_update_stats,Wh=Wh_update_stats,bh=bh_update_stats,bv=bv_update_stats)

        return [param_stats, update_stats]

class LCRBM(CRBM):
    """Labeled Conditional Restricted Boltzmann Machine (CRBM) using numpy """
    def __init__(self, Nv, Nh, Nl, Tv, Th, period=None, vis_unit='binary', vis_scale=0,
            bv = None,
            dtype='float32'):
        """
        input:
        -----------------

        (in addition to those defined in CRBM class)

        Nl: number of label units (group of softmax units)
        """

        super(LCRBM,self).__init__(Nv, Nh, Tv, Th, period, vis_unit, vis_scale,
            bv, dtype='float32')


        # add label units to visible units

        # W is initialized with uniformly sampled data
        # from -4.*sqrt(6./(Nv+Nh)) and 4.*sqrt(6./(Nh+Nv))
        W = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv+Nl+Nh)),
            high  =  4*np.sqrt(6./(Nv+Nl+Nh)),
            size  =  (Nv+Nl, Nh)),
            dtype =  dtype)

        Wv = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv*Tv+Nv+Nl)),
            high  =  4*np.sqrt(6./(Nv*Tv+Nv+Nl)),
            size  =  (Nv*Tv, Nv)),
            dtype =  dtype)


        if bv is None :
            bv = np.zeros((Nv+Nl),dtype=dtype)
        else:
            bv = bv.copy()


        # new label-unit params -------------------------------------------
        self.Nl     = Nl        # num label units

        self.W      = W     # (vis+lab)<->hid weights
        self.Wv     = Wv    # vis->vis delay weights
        self.bv     = bv    # vis bias

        self.W_update = np.zeros((Nv+Nl,Nh),dtype=dtype)
        self.Wv_update = np.zeros((Nv*Tv,Nv),dtype=dtype)
        self.bv_update = np.zeros((Nv+Nl,),dtype=dtype)

        self.params     += ['Nl']

    def load_params(self,filename):
        '''load parameters from file'''
        super(LCRBM,self).load_params(filename)

        Nv,Nh,Nl,Tv = self.Nv,self.Nh,self.Nl,self.Tv
        dtype = self.dtype
        self.W_update = np.zeros((Nv+Nl,Nh),dtype=dtype)
        self.Wv_update = np.zeros((Nv*Tv,Nv),dtype=dtype)
        self.bv_update = np.zeros((Nv+Nl,),dtype=dtype)

    def save_params(self,filename=None):
        '''save parameters to file'''
        if filename is None:
            id = np.random.randint(100000)
            filename = 'LCRBM_%u.pkl' % id
        super(LCRBM,self).save_params(filename)

    def return_params(self):
        '''
        return a formatted string containing scalar parameters
        '''

        output = super(LCRBM,self).return_params()
        output = 'Nl=%u, ' % (self.Nl) + output
        return output

    def extract_data(self,v_input):
        Nv = self.Nv
        Nl = self.Nl
        Tv = self.Tv
        Th = self.Th

        Nvl = Nv + Nl

        if v_input.ndim == 1:
            v_data = v_input[-Nvl:]
            vv_past = v_input[-(Nv*Tv+Nvl):-Nvl]
            vh_past = v_input[-(Nv*Th+Nvl):-Nvl]
        else:
            v_data = v_input[:,-Nvl:]
            vv_past = v_input[:,-(Nv*Tv+Nvl):-Nvl]
            vh_past = v_input[:,-(Nv*Th+Nvl):-Nvl]

        return v_data, vv_past, vh_past

    def separate_vis_lab(self,x,axis=1):
        '''
        separate visible unit data from label unit data
        '''

        Nl = self.Nl

        if x.ndim == 1:
            axis = 0

        if axis == 0:
            x_lab = x[-Nl:]
            x_vis = x[:-Nl]
        elif axis == 1:
            x_lab = x[:,-Nl:]
            x_vis = x[:,:-Nl]

        return x_vis, x_lab

    def join_vis_lab(self,x_vis,x_lab,axis=1):
        '''
        join visible unit data to label unit data
        '''
        if x_vis.ndim == 1:
            axis = 0

        x = np.concatenate((x_vis,x_lab),axis=axis)

        return x

    def mean_field_v_given_h(self,h,v_bias):
        '''compute mean-field reconstruction of P(vt|ht,v<t)'''
        x = v_bias + np.dot(h, self.W.T)
        x_vis, x_lab = self.separate_vis_lab(x)
        lab_mean = softmax(x_lab)

        if self.vis_unit == 'binary':
            vis_mean = sigmoid(x_vis)
        elif self.vis_unit == 'linear':
            vis_mean = log_1_plus_exp(x_vis) - log_1_plus_exp(x_vis-self.vis_scale)

        means = self.join_vis_lab(vis_mean,lab_mean)
        return means

    def sample_v_given_h(self,h,v_bias):
        '''compute samples from P(vt|ht,v<t)'''
        if self.vis_unit == 'binary':
            means = self.mean_field_v_given_h(h,v_bias)
            vis_mean,lab_mean = self.separate_vis_lab(means)
            vis_samp = (np.random.uniform(size=vis_mean.shape) < vis_mean).astype(self.dtype)

        elif self.vis_unit == 'linear':
            x = v_bias + np.dot(h, self.W.T)
            x_vis, x_lab = self.separate_vis_lab(x)
            # variance of noise is sigmoid(x_vis) - sigmoid(x_vis - vis_scale)
            vis_stddev = np.sqrt(sigmoid(x_vis) - sigmoid(x_vis - self.vis_scale)) 
            vis_mean =  log_1_plus_exp(x_vis) - log_1_plus_exp(x_vis-self.vis_scale)
            vis_noise = stddev * np.random.standard_normal(size=x.shape)
            vis_samp = np.fmax(0,np.fmin(self.vis_scale, vis_mean + vis_noise))

            lab_mean = softmax(x_lab)
            means = self.join_vis_lab(vis_mean,lab_mean)

        lab_samp = sample_categorical(lab_mean)
        samples = self.join_vis_lab(vis_samp,lab_samp)

        return samples, means

    def compute_gradients(self,v_input,h0,vk,hk):
        v0,vv_past,vh_past = self.extract_data(v_input) 
        v0_vis,v0_lab = self.separate_vis_lab(v0)
        vk_vis,vk_lab = self.separate_vis_lab(vk)

        W_grad = np.dot(vk.T, hk) - np.dot(v0.T, h0)
        Wv_grad = np.dot(vv_past.T, vk_vis) - np.dot(vv_past.T, v0_vis)
        Wh_grad = np.dot(vh_past.T, hk) - np.dot(vh_past.T, h0)

        bv_grad = np.sum(vk - v0,axis=0)
        bh_grad = np.sum(hk - h0,axis=0)

        return W_grad,Wv_grad,Wh_grad,bv_grad,bh_grad

    def compute_dynamic_bias(self,v_input):
        v_data,vv_past,vh_past = self.extract_data(v_input)
        v_bias = np.tile(self.bv,[v_data.shape[0],1])
        v_bias[:,:self.Nv] += np.dot(vv_past,self.Wv)
        h_bias = self.bh + np.dot(vh_past,self.Wh)

        return v_bias, h_bias

    def label_probabilities(self,v_input,output_h=False):
        '''
        compute the activation probability of each label unit given the visible units
        '''

        #compute free energy for each label configuration
        # F(v,c) = -sum(v*bv) - bl[c] - sum(log(1 + exp(z_c)))
        # where z_c = bh + dot(v,W) + r[c]  (r[c] are the weights for label c)
        # also,  v_data = [v,l],  where l are binary "one-hot" labels

        v_data,vv_past,vh_past = self.extract_data(v_input)

        b_hid = self.bh + np.dot(vh_past,self.Wh)
        b_vis, b_lab = self.separate_vis_lab(self.bv)

        v_vis, v_lab = self.separate_vis_lab(v_data)
        W_vis,W_lab = self.separate_vis_lab(self.W,axis=0)


        # the b_vis term cancels out in the softmax
        #F = -np.sum(v_vis*b_vis,axis=1) 
        #F = F.reshape((-1,1)) - b_lab
        F =  - b_lab

        z = b_hid + np.dot(v_vis,W_vis)
        z = z.reshape(z.shape + (1,))
        z = z + W_lab.T.reshape((1,) + W_lab.T.shape)

        hidden_terms = -np.sum(log_1_plus_exp(z), axis=1)
        F = F + hidden_terms

        pr = softmax(-F)

        # compute hidden probs for each label configuration
        # this is used in the discriminative updates
        if output_h:
            h = sigmoid(z)
            return pr, h
        else:
            return pr

    def discriminative_train(self,v_input,rate=0.001,momentum=0.0,weight_decay=0.001):
        ''' 
        Update weights using discriminative updates.
        These updates use gradient ascent of the log-likelihood of the 
        label probability of the correct label

        input:
        v_input - [v_past, v_visible, v_labels]
            (v_labels contains the binary activation of the correct label)
        '''

        N = v_input.shape[0]


        # things to compute:
        # h_d - hidden unit activations for each label configuration
        # p_d - label unit probabilities
        p_d, h_d = self.label_probabilities(v_input,output_h=True)

        v_data,vv_past,vh_past = self.extract_data(v_input)
        v_vis,v_lab = self.separate_vis_lab(v_data)

        ind, true_labs = np.where(v_lab == 1)


        scale = rate / N


        # prob_scale = (1-p_d) for correct label and -p_d for other labels
        prob_scale = -p_d
        prob_scale[ind,true_labs] += 1
        ps_broad = prob_scale.reshape((N,1,self.Nl)) # make broadcastable across h_d

        p_h_sum = np.sum(ps_broad * h_d, axis=2)

        # compute gradients ----------------------------------------------
        # W = [w,r]
        w_grad = np.dot(v_vis.T, p_h_sum)               # vis<-->hid
        r_grad = np.sum( ps_broad * h_d, axis=0 ).T     # lab<-->hid
        W_grad = self.join_vis_lab(w_grad,r_grad,axis=0)# [vis,lab]<-->hid

        Wh_grad = np.dot( vh_past.T , p_h_sum )         # vh_past-->hid 
        bh_grad = np.sum(p_h_sum,axis=0)                # -->hid
        # bv = [bvv,bvl]                                # -->[vis,lab]
        bvv,bvl = self.separate_vis_lab(self.bv)
        bvv_grad = np.zeros_like(bvv)                   # -->vis
        bvl_grad = np.sum(prob_scale,axis=0)            # -->lab
        # ---------------------------------------------------------------


        if weight_decay > 0.0:
            W_grad += -weight_decay * self.W
            Wh_grad += -weight_decay * self.Wh         

        bv_grad = self.join_vis_lab(bvv_grad,bvl_grad)


        if momentum > 0.0:
            self.W_update = momentum * self.W_update + scale*W_grad
            self.Wv_update = 0.0
            self.Wh_update = momentum * self.Wh_update + scale*Wh_grad
            self.bh_update = momentum * self.bh_update + scale*bh_grad
            self.bv_update = momentum * self.bv_update + scale*bv_grad
        else:
            self.W_update = scale*W_grad
            self.Wv_update = 0.0
            self.Wh_update = scale*Wh_grad
            self.bh_update = scale*bh_grad
            self.bv_update = scale*bv_grad

        self.W += self.W_update
        #self.Wv += self.Wv_update
        self.Wh += self.Wh_update
        self.bh += self.bh_update
        self.bv += self.bv_update

    def recon_error(self, v_input,K=1,print_output=False):
        '''compute K-step reconstruction error'''

        v0_data,vv_past,vh_past = self.extract_data(v_input)
        vk_mean = self.gibbs_samples(K,v_input,noisy=0)

        v0_vis,v0_lab = self.separate_vis_lab(v0_data)
        vk_vis,vk_lab = self.separate_vis_lab(vk_mean)
        vis_error = np.mean(np.abs(v0_vis - vk_vis))
        lab_error = np.mean(np.abs(v0_lab - vk_lab))

        lab_probs = self.label_probabilities(v_input)
        pred_labs = np.argmax(lab_probs,axis=1)
        ind, true_labs = np.where(v0_lab == 1)
        percent_correct = np.mean(pred_labs == true_labs)
        prob_error = np.mean(np.abs(1. - lab_probs[ind,true_labs]))

        if print_output:
            output = '%30s %6.5f' % ('vis error:', vis_error/self.vis_scale) + '\n'
            output += '%30s %6.5f' % ('lab error:', lab_error) + '\n'
            output += '%30s %6.5f' % ('prob error:', prob_error) + '\n'
            output += '%30s %6.5f' % ('class correct:', percent_correct)
            print output
            return output
        else: 
            return percent_correct, prob_error, lab_error, vis_error/self.vis_scale

class LabelLayer(object):
    '''
    Softmax Free Energy Classifier layer

    params:
    Ni - num input units
    No - num output units (num classes)
    W - input -> hidden weights
    R - label -> hidden weights
    bl - label bias
    bh - hidden bias
    pretrained - [str] describes pretraining used
    '''
    def __init__(self):
        '''
        init with empty weights
        '''
        self.Ni = 0
        self.No = 0
        self.W = None
        self.R = None
        self.bl = None
        self.bh = None

        self.pretrained = None

        self.params = ['Ni','No','W','R','bl','bh','pretrained']

    def collect_weights_from_rbm(self,rbm):
        '''
        pull necessary parameters from labeled rbm
        '''
        if type(rbm) not in [LRBM,LCRBM]:
            raise TypeError, 'input must be a labeled RBM'

        W,R = rbm.separate_vis_lab(rbm.W,axis=0)

        if type(rbm) is LCRBM:
            W = np.concatenate( (rbm.Wh, W))

        bv,bl = rbm.separate_vis_lab(rbm.bv)

        self.W = W.copy()
        self.R = R.copy()
        self.bl = bl.copy()
        self.bh = rbm.bh.copy()

        self.Ni = self.W.shape[0]
        self.No = rbm.Nl

        self.W_update = gp.zeros(self.W.shape)
        self.R_update = gp.zeros(self.R.shape)
        self.bl_update = gp.zeros(self.bl.shape)
        self.bh_update = gp.zeros(self.bh.shape)

        self.pretrained = str(type(rbm))

    def compute_activations(self,a_in):
        '''
        compute the activation probability of each label unit given the visible units

        input:
        a_in - activations of input layer
        '''

        return self.label_probabilities(a_in)

    def backprop(self,err,a_in,N=None):
        '''
        backpropagate the error signal

        input:
        err - error signal
        a_in - activation of input units
        N - actual number of training examples to average over
        '''

        h_act = sigmoid(self.z)

        # backpropagate the err
        v_back = np.dot(h_act,self.W.T)
        dEdz = np.sum(err.reshape(err.shape + (1,)) * v_back, axis=1) * (a_in*(1-a_in))

        # compute gradients
        gradients = self.compute_gradients(err,a_in,h_act,N)

        return dEdz, gradients

    def label_probabilities(self,v_input,output_h=True):
        '''
        compute the activation probability of each label unit given the visible units
        '''

        #compute free energy for each label configuration
        # F(v,c) = -sum(v*bv) - bl[c] - sum(log(1 + exp(z_c)))
        # where z_c = bh + dot(v,W) + r[c]  (r[c] are the weights for label c)
        # also,  v_data = [v,l],  where l are binary "one-hot" labels

        # the b_vis term cancels out in the softmax
        #F = -np.sum(v_vis*b_vis,axis=1) 
        #F = F.reshape((-1,1)) - b_lab
        F =  - self.bl

        z = self.bh + np.dot(v_input,self.W)
        z = z.reshape((z.shape[0],1,z.shape[1]))
        z = z + self.R.reshape((1,) + self.R.shape)


        hidden_terms = -np.sum(log_1_plus_exp(z), axis=2)
        F = F + hidden_terms

        pr = softmax(-F)

        # save z's for gradient computation
        self.z = z

        return pr

    def update_weights(self,gradients,rate=0.001,momentum=0.0,weight_decay=0.001):
        W_grad, R_grad, bh_grad, bl_grad = gradients

        if weight_decay > 0.0:
            W_grad += -weight_decay * self.W
            R_grad += -weight_decay * self.R

        rate = float(rate)
        if momentum > 0.0:
            momentum = float(momentum)
            self.W_update = momentum * self.W_update + rate*W_grad
            self.R_update = momentum * self.R_update + rate*R_grad
            self.bl_update = momentum * self.bl_update + rate*bl_grad
            self.bh_update = momentum * self.bh_update + rate*bh_grad
        else:
            self.W_update = rate*W_grad
            self.R_update = rate*R_grad
            self.bl_update = rate*bl_grad
            self.bh_update = rate*bh_grad



        self.W += self.W_update
        self.R += self.R_update
        self.bl += self.bl_update
        self.bh += self.bh_update

    def compute_gradients(self,err,a_in,h_act,N=None):
        ''' 
        Compute gradient of correct classification probability
        (gradient of the log-likelihood of the label probability of the correct label)

        input:
        err - error signal
        a_in - input unit activations
        h_act - hidden unit activities for each label configuration [N x Nl x Nh]
        N - number of training examples gradient is averaged over
        '''

        # number of training examples
        if N is None:
            N = a_in.shape[0]

        N_inv = 1./N


        err_broad = err.reshape(err.shape + (1,)) # make broadcastable across h_d

        p_h_sum = np.sum(err_broad * h_act, axis=1)

        # compute gradients ----------------------------------------------
        W_grad = N_inv * np.dot(a_in.T, p_h_sum)               # vis<-->hid
        R_grad = np.mean( err_broad * h_act, axis=0 )     # lab<-->hid
        bh_grad = np.mean(p_h_sum,axis=0)                # -->hid
        bl_grad = np.mean(err,axis=0)            # -->lab





        return [W_grad, R_grad, bh_grad, bl_grad]

    def update_stats(self):
        W_stats = [np.min(self.W),np.mean(np.abs(self.W)),np.max(self.W)]
        R_stats = [np.min(self.R),np.mean(np.abs(self.R)),np.max(self.R)]
        bh_stats = [np.min(self.bh),np.mean(np.abs(self.bh)),np.max(self.bh)]
        bl_stats = [np.min(self.bl),np.mean(np.abs(self.bl)),np.max(self.bl)]

        W_update_stats = [np.min(self.W_update), np.mean(np.abs(self.W_update)), np.max(self.W_update)]
        R_update_stats = [np.min(self.R_update), np.mean(np.abs(self.R_update)), np.max(self.R_update)]
        bh_update_stats = [np.min(self.bh_update), np.mean(np.abs(self.bh_update)), np.max(self.bh_update)]
        bl_update_stats = [np.min(self.bl_update), np.mean(np.abs(self.bl_update)), np.max(self.bl_update)]

        param_stats = dict(W=W_stats,R=R_stats,bh=bh_stats,bl=bl_stats)
        update_stats = dict(W=W_update_stats,R=R_update_stats,bh=bh_update_stats,bl=bl_update_stats)

        return [param_stats, update_stats]

    def classification_error(self,v_input,v_lab,modulo=None):
        '''
        compute label classification error

        input:
        v_input - input unit activations
        v_lab - true labels (binary one-hot encoding)
        modulo - test accuracy is invariant to shift of this amount
        '''

        lab_probs = self.label_probabilities(v_input,output_h=False)
        if modulo is not None:
            N,L = lab_probs.shape
            lab_probs = lab_probs.reshape([N,L/modulo,modulo]).sum(axis=1)
            v_lab_mod = v_lab.reshape([N,L/modulo,modulo]).sum(axis=1)
            ind, true_labs = np.where(v_lab_mod == 1)
        else:
            ind, true_labs = np.where(v_lab == 1)
        #pred_labs = gargmax(lab_probs)
        pred_labs = lab_probs.argmax(axis=1)
        percent_correct = np.mean(pred_labs == true_labs)
        corr_lab_probs = lab_probs[ind,true_labs]
        cross_entropy = -np.mean(np.log(corr_lab_probs))

        #sorted_lab_probs = lab_probs.copy()
        #sorted_lab_probs.sort(axis=1)
        #top_prob_ratios = sorted_lab_probs[:,-1]/corr_lab_probs
        #second_prob_ratios = sorted_lab_probs[:,-2]/corr_lab_probs

        #return percent_correct, cross_entropy, second_prob_ratios 
        #return percent_correct, cross_entropy, top_prob_ratios 
        return percent_correct, cross_entropy, corr_lab_probs

    def get_params(self):
        '''
        return dictionary of parameters
        '''
        params_out = {}
        for p in self.params:
            val = vars(self)[p]
            if type(val) is gp.garray:
                params_out[p] = val.as_numpy_array()
            else:
                params_out[p] = val
        return params_out

    def set_params(self,params_in):
        '''
        load params from dictionary
        '''

        for key,value in params_in.iteritems():
            if type(value) is np.ndarray:
                vars(self)[key] = value
                vars(self)[key + '_update'] = np.zeros(value.shape)
            else:
                vars(self)[key] = value

class NNLayer(object):
    '''
    Basic neural network layer

    params:
    W - weights connecting input units to output units
    b - bias for each output unit
    Ni - num input units
    No - num output units
    pretrained - [str] describes pretraining used
    '''
    def __init__(self):
        '''
        initialize empty NN weights
        '''
        self.Ni = 0
        self.No = 0
        self.W = None
        self.b = None

        self.pretrained = None

        self.params = ['Ni','No','W','b','pretrained']

    def compute_activations(self,a_in):
        '''
        input:
        a_in - [N x Ni] matrix where N is number of observations
                or a list of this type of matrix
        '''
        output = []
        if type(a_in) is list:
            for seq in a_in:
                output += [self.compute_activations(seq)]
            return output
        else:
            return sigmoid(self.b + np.dot(a_in,self.W))

    def backprop(self,err,a_in,N=None):
        '''
        backpropagate error signal and compute gradients

        input:
        err - error signal from previous layer
        a_in - input activations
        N - actual number of training examples at top level
        '''

        
        # backpropagate the error signal
        dEdz = np.dot(err,self.W.T) * (a_in*(1-a_in))

        # compute gradients
        gradients = self.compute_gradients(err,a_in,N)

        return dEdz, gradients

    def update_weights(self,gradients,rate=0.001,momentum=0.0,weight_decay=0.001):
        W_grad, b_grad = gradients

        if weight_decay > 0.0:
            W_grad += -weight_decay * self.W

        rate = float(rate)
        if momentum > 0.0:
            momentum = float(momentum)
            self.W_update = momentum * self.W_update + rate*W_grad
            self.b_update = momentum * self.b_update + rate*b_grad
        else:
            self.W_update = rate*W_grad
            self.b_update = rate*b_grad

        self.W += self.W_update
        self.b += self.b_update

    def compute_gradients(self,err,a_in,N=None):
        '''
        compute gradients from error signal and activations of lower layer
        N - number of training examples to avergae gradient estimate over (for scaling)
        '''

        if N is None:
            N = err.shape[0]

        N_inv = 1./N

        W_grad = N_inv * np.dot(a_in.T,err)
        b_grad = np.mean(err,axis=0)

        return [W_grad, b_grad]

    def update_stats(self):
        W_stats = [np.min(self.W),np.mean(np.abs(self.W)),np.max(self.W)]
        b_stats = [np.min(self.b),np.mean(np.abs(self.b)),np.max(self.b)]

        W_update_stats = [np.min(self.W_update), np.mean(np.abs(self.W_update)), np.max(self.W_update)]
        b_update_stats = [np.min(self.b_update), np.mean(np.abs(self.b_update)), np.max(self.b_update)]

        param_stats = dict(W=W_stats,b=b_stats)
        update_stats = dict(W=W_update_stats,b=b_update_stats)

        return [param_stats, update_stats]

    def get_params(self):
        '''
        return dictionary of parameters
        '''
        params_out = {}
        for p in self.params:
            val = vars(self)[p]
            if type(val) is gp.garray:
                params_out[p] = val.as_numpy_array()
            else:
                params_out[p] = val
        return params_out

    def set_params(self,params_in):
        '''
        load params from dictionary
        '''

        for key,value in params_in.iteritems():
            if type(value) is np.ndarray:
                vars(self)[key] = value
                vars(self)[key + '_update'] = np.zeros(value.shape)
            else:
                vars(self)[key] = value

    def set_weights(self,W,b):
        self.Ni, self.No = W.shape
        self.W = W
        self.b = b

    def collect_weights_from_rbm(self,rbm):
        '''
        Collect neural network recognition weights from rbm layer
        '''

        if type(rbm) not in [RBM,CRBM]:
            raise TypeError, 'invalid layer type'

        W = rbm.W
        if type(rbm) is CRBM:
            W = np.concatenate( (rbm.Wh, rbm.W) ).copy()

        self.W = W.copy()
        self.b = rbm.bh.copy()
        self.Ni, self.No = self.W.shape

        self.W_update = gp.zeros(self.W.shape)
        self.b_update = gp.zeros(self.b.shape)

        self.pretrained = str(type(rbm))

class SoftmaxLayer(NNLayer):
    '''
    Softmax output layer

    params:
    W - weights connecting input units to output units
    b - bias for each output unit
    Ni - num input units
    No - num output units
    '''
    def __init__(self,Ni,No,b=None):
        super(SoftmaxLayer,self).__init__()
        self.init_weights(Ni,No,b)

    def compute_activations(self,a_in):
        '''
        input:
        a_in - [N x Ni] matrix where N is number of observations
                or a list of this type of matrix
        '''
        output = []
        if type(a_in) is list:
            for seq in a_in:
                output += [self.compute_activations(seq)]
            return output
        else:
            return softmax(self.b + np.dot(a_in,self.W))

    def classification_error(self,v_input,v_lab):
        '''
        compute label classification error

        input:
        v_input - input unit activations
        v_lab - true labels (binary one-hot encoding)
        '''

        lab_probs = self.compute_activations(v_input)
        pred_labs = lab_probs.argmax(axis=1)
        ind, true_labs = np.where(v_lab == 1)
        percent_correct = np.mean(pred_labs == true_labs)
        #cross_entropy = -np.mean(np.log(lab_probs[ind,true_labs]))
        corr_lab_probs = lab_probs[ind,true_labs]
        cross_entropy = -np.mean(np.log(corr_lab_probs))

        return percent_correct, cross_entropy, corr_lab_probs

    def init_weights(self,Ni,No,b=None):
        '''
        set number of input/ouput units, bias, and random initial weights
        '''

        self.Ni, self.No = Ni, No

        W = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Ni+No)),
            high  =  4*np.sqrt(6./(Ni+No)),
            size  =  (Ni, No)),
            dtype =  'float32')

        if b is None :
            b = np.zeros(No)

        self.W = W
        self.b = b

        self.W_update = np.zeros(self.W.shape)
        self.b_update = np.zeros(self.b.shape)

class NeuralNet(object):
    ''' 
    Labeled Conditional Deep Belief Network
    - Discriminative Neural Network
    - CRBM at first level.  LCRBM or LRBM at top level.
    This is basically a container for the constituent layers of the network.

    params:
    rbms - list containing the rbm at each layer 
    layers - list containing the recognition weights corresponding to each rbm
    '''

    def __init__(self):
        self.layers = list()
        self.rbms = list()
        self.num_layers = 0
        self.frame_size = []

    def return_params(self):
        '''
        return a formatted string containing scalar parameters
        '''
        output = ''
        for l in xrange(self.num_layers):
            if type(self.layers[l]) is LabelLayer:
                Nh = self.layers[l].bh.size
                Ni = self.layers[l].Ni
                No = self.layers[l].No
                try:
                    model = str(type(self.rbms[l]))
                except:
                    model = 'mystery layer'
                output += 'level%u: Ni=%u, No=%u, Nh=%u, %s\n' % (l,Ni,No,Nh,model)
            else:
                Ni = self.layers[l].Ni
                No = self.layers[l].No
                try:
                    model = str(type(self.rbms[l]))
                except:
                    model = 'mystery layer'
                output += 'level%u: Ni=%u, No=%u, %s\n' % (l,Ni,No,model)
        output = output[:-1] # remove trailing newline

        return output

    def add_rbm(self,rbm):
        '''
        Append a layer to the end of the list
        and collect the current rbm recognition weights into the neural network
        '''
        if type(rbm) not in [RBM,CRBM,LCRBM,LRBM]:
            raise TypeError, 'invalid layer type'

        if type(rbm) in [LRBM,LCRBM]:
            layer = LabelLayer()
        else:
            layer = NNLayer()

        layer.collect_weights_from_rbm(rbm)

        # compute frame size for CRBMs.
        # input layer CRBMs are provided with pre-framed data,
        # so only higher-level CRBMs need input activations to be reframed
        if type(rbm) in [CRBM,LCRBM]:
            self.frame_size += [rbm.Th]
        else:
            self.frame_size += [0]


        self.layers += [layer]
        self.rbms += [rbm]
        self.num_layers += 1

        if self.num_layers > 1:
            if self.layers[-1].Ni != self.layers[-2].No * (self.frame_size[-1]+1):
                raise ValueError, 'num input units != num output units of previous layer'

    def add_layer(self,layer):
        '''
        append a layer to the neural net
        '''
        if type(layer) in [RBM,CRBM,LCRBM,LRBM]:
            self.add_rbm(layer)

        elif type(layer) in [LabelLayer,NNLayer,SoftmaxLayer]:
            self.layers += [layer]
            self.num_layers += 1

            if self.num_layers > 1:
                if self.layers[-1].Ni != self.layers[-2].No:
                    raise ValueError, 'num input units != num output units of previous layer'
        else:
            raise TypeError, 'invalid layer type'
    def train_backprop(self,v_input, v_lab,rate,momentum,weight_decay):
        '''
        train network using backpropagation of log-likelihood error signal

        input:
        v_input - values of first layer visible input units
        v_lab - true labels for training
        rate - learning_rate [array]
                array of learning rate for each level
        momentum - learning momentum [scalar]
        weight_decay - weight decay [scalar]
                not, array of weight decay for each level
        '''

        activations = {}
        activations[0] = v_input

        # up-pass
        L = self.num_layers
        for l in xrange(0,L):
            activations[l+1] = self.layers[l].compute_activations(activations[l])

        # compute error
        ind, true_labs = np.where(v_lab == 1)
        err = v_lab - activations[L]

        err = {L:err}
        gradients = {}

        # backpropagate the error, and compute gradients
        for l in xrange(L,0,-1):
            err[l-1], gradients[l-1] = self.layers[l-1].backprop(err[l],activations[l-1])

        # update weights
        for l in xrange(0,L):
            self.layers[l].update_weights(gradients[l],rate[l],momentum,weight_decay)

    def classification_error(self,data,labels,print_output=False,modulo=None):
        '''
        compute classification error
        '''

        #activations = data
        #for l in xrange(self.num_layers-1):
        #    activations = self.layers[l].compute_activations(activations)
        activations = self.compute_activations(data)

        percent_correct, cross_entropy, correct_lab_probs = self.layers[-1].classification_error(activations,labels,modulo)

        if print_output:
            output = '%30s %6.5f' % ('cross entropy:', cross_entropy) + '\n'
            output +=  '%30s %6.5f' % ('class correct:', percent_correct)
            print output
            return percent_correct, cross_entropy, correct_lab_probs, output
        else:
            return percent_correct, cross_entropy, correct_lab_probs

    def compute_activations(self,data):
        '''
        compute activations up to input to output layer
        '''

        activations = data
        for l in xrange(self.num_layers-1):
            activations = self.layers[l].compute_activations(activations)

        return activations

    def compute_output(self,data):
        '''
        compute activations of output units
        '''
        activations = self.compute_activations(data)
        output = self.layers[-1].label_probabilities(activations,output_h=False)

        return output

    def save_params(self,filename=None):
        '''
        save neural network parameters
        '''
        if filename is None:
            id = np.random.randint(100000)
            filename = 'NN_%u.pkl' % id
            
        params_out = {}  
        params_out['num_layers'] = self.num_layers
        params_out['frame_size'] = self.frame_size
        params_out['layers'] = []
        for l in self.layers:
            params_out['layers'] += [l.get_params()]

        fp = open(filename,'wb')
        pkl.dump(params_out,fp,protocol=-1)
        fp.close()

        print 'saved %s' % filename

    def load_params(self,filename):
        '''
        load parameters from file
        '''
        fp = open(filename,'rb')
        params_in = pkl.load(fp)
        fp.close()

        self.num_layers = params_in['num_layers']
        self.frame_size = params_in['frame_size']
        self.layers = []
        for l in xrange(self.num_layers-1):
            layer = NNLayer()
            layer.set_params(params_in['layers'][l])
            self.layers += [layer]

        layer = LabelLayer()
        layer.set_params(params_in['layers'][self.num_layers-1])
        self.layers += [layer]

        #print 'loaded %s' % filename

class NeuralNetFramed(NeuralNet):
    ''' 
    Labeled Conditional Deep Belief Network
    - Discriminative Neural Network
    - CRBM at first level.  LCRBM or LRBM at top level.
    This is basically a container for the constituent layers of the network.

    params:
    rbms - list containing the rbm at each layer 
    layers - list containing the recognition weights corresponding to each rbm
    '''

    def compute_activations(self,data):
        '''
        compute output unit activations 
        '''

        activations = subseq_to_frames(data)

        N = data.shape[0]


        # up-pass
        L = self.num_layers
        for l in xrange(0,L-1):
            act = self.layers[l].compute_activations(activations)
            activations = act.reshape((N,-1))

        return activations

    def train_backprop(self,v_input, v_lab,rate,momentum,weight_decay):
        '''
        train network using backpropagation of log-likelihood error signal

        input:
        v_input - values of first layer visible input units
        v_lab - true labels for training
        rate - learning_rate [scalar]
                not, array of learning rate for each level
        momentum - learning momentum [scalar]
                not, array of momentum for each level
        weight_decay - weight decay [scalar]
                not, array of weight decay for each level
        '''



        activations = {}
        activations[0] = subseq_to_frames(v_input)

        N = v_input.shape[0]


        # up-pass
        L = self.num_layers
        for l in xrange(0,L):
            act = self.layers[l].compute_activations(activations[l])
            activations[l+1] = act.reshape((N,-1))

        # compute error
        ind, true_labs = np.where(v_lab == 1)
        err = v_lab - activations[L]

        err = {L:err}
        gradients = {}


        # backpropagate the error, and compute gradients
        for l in xrange(L,0,-1):
            S = self.layers[l-1].W.shape
            if l < L and err[l].shape[1] != S[1]:
                err[l]= err[l].reshape((-1,S[1]))
            err[l-1], gradients[l-1] = self.layers[l-1].backprop(err[l],activations[l-1],N)


        # update weights
        for l in xrange(0,L):
            self.layers[l].update_weights(gradients[l],rate[l],momentum,weight_decay)

def compute_binary_labels(labels,num_labels):
    '''
    return a binary one-hot encoding of the labels
    '''

    num_frames = labels.shape[0]

    binary_labels = np.zeros((num_frames,num_labels))
    if labels.ndim == 2:
        binary_labels[np.arange(num_frames),labels[:,-1]-1] = 1
    elif labels.ndim == 1:
        binary_labels[np.arange(num_frames),labels-1] = 1

    return binary_labels

def sigmoid(x):
    '''
    compute logistic sigmoid function avoiding overflow
    '''
    overflow = x < -88
    if np.any(overflow):
        in_range = np.logical_not(overflow)
        y = np.zeros_like(x)
        y[in_range] = 1./(1. + np.exp(-x[in_range]))
        xo = x[overflow]
        y[overflow] = np.exp(xo)/(1. + np.exp(xo))
    else:
        y = 1./(1. + np.exp(-x))

    return y

def softmax(x):
    '''
    compute softmax function for each row
    while avoiding over/underflow
    '''
    m = np.max(x,axis=1).reshape((-1,1)) # max for each row
    y = np.exp(x - m)
    y /= np.sum(y,axis=1).reshape((-1,1))

    return y

def sample_categorical(probs):
    '''
    sample from categorical distribution (1-sample multinomial distribution)

    input:
    probs - probabilities in each row add to one [N x K]

    output:
    samples - [N x K] binary array with a single 1 per row
    '''
    if probs.ndim == 1:
        probs = probs.reshape((1,-1))

    N = probs.shape[0]
    cdf = np.cumsum(probs, axis=1)[:,:-1]
    uni = np.random.uniform(size=(N,1))
    category = np.sum(uni >= cdf,axis=1)
    samples = np.zeros_like(probs)
    samples[np.arange(N),category] = 1

    return samples

def log_1_plus_exp(x):
    '''
    compute y = np.log(1+np.exp(x)) avoiding overflow
    '''
    y = np.zeros_like(x)
    pos = x > 0
    neg = np.logical_not(pos)
    x_gtz = x[pos]

    y[pos] = np.log(1 + np.exp(-x_gtz)) + x_gtz
    y[neg] = np.log(1 + np.exp(x[neg]))

    return y

def frame_sequence_data(frame_size,sequential_data,sequential_labels=None):
    '''
    construct overlapping frames of training data of the correct size

    input:
    frame_size - number of sequential observations per frame
    sequential_data - list of numpy arrays [N_o,Nv] 
                        where N_o is the number of observations 
                        and can be different for each list element
    sequential_labels - list of numpy arrays [N_o,Nl]  (optional)
                        where N_o is the number of observations 
                        and can be different for each list element

    output:
    sequence_data - numpy array with framed observations in rows
    sequence_ind - list of slices containing bounds of each sequence in the above array
    sequence_labels - [optional] 1D numpy array of labels for each frame
    '''


    
    #curr_ind = 0
    #inds = []

    if sequential_labels is not None:
        all_frames = []
        all_labels = []
        for seq,lab in zip(sequential_data,sequential_labels):
            N_os, Nv = seq.shape
            N_ol =  lab.size
            
            if N_os != N_ol:
                raise ValueError, 'data/label dimension mismatch'
            num_frames = N_os - frame_size + 1

            frames = np.zeros((num_frames,frame_size * Nv),dtype='float32')
            labels = np.zeros((num_frames,1), dtype='uint16')
            start = 0
            end = frame_size 

            for i in xrange(num_frames):
                frames[i] = seq[start:end].flatten()
                labels[i] = lab[end-1]
                start += 1
                end += 1


            all_frames += [frames]
            all_labels += [labels]
            #inds += [slice(curr_ind,curr_ind+num_frames)]
            #curr_ind += num_frames


        #return np.concatenate(all_frames), inds, np.concatenate(all_labels)
        return all_frames, all_labels

    else:
        all_frames = []
        for seq in sequential_data:
            N_o, Nv = seq.shape
            num_frames = N_o - frame_size + 1

            frames = np.zeros((num_frames,frame_size * Nv),dtype='float32')
            start = 0
            end = frame_size # exclusive

            for i in xrange(num_frames):
                frames[i] = seq[start:end].flatten()
                start += 1
                end += 1

            all_frames += [frames]
            #inds += [slice(curr_ind,curr_ind+num_frames)]
            #curr_ind += num_frame

        #return np.concatenate(all_frames), inds
        return all_frames 

def frame_subseqs(frame_size,subseq_data,subseq_labels=None):
    '''
    construct overlapping frames of training data of the correct size

    input:
    frame_size - number of sequential observations per frame
    subseq_data - list of numpy arrays [N_o,Nv] 
                        where N_o is the number of observations 
                        and can be different for each list element
    subseq_labels - list of numpy arrays [N_o,Nl]  (optional)
                        where N_o is the number of observations 
                        and can be different for each list element

    output:
    sequence_data - numpy array with framed observations in rows
    sequence_ind - list of slices containing bounds of each sequence in the above array
    sequence_labels - [optional] 1D numpy array of labels for each frame
    '''


    
    num_subseqs,subseq_length,data_dim = subseq_data.shape
    frames_per_subseq = subseq_length - frame_size + 1
    frame_data = np.zeros((num_subseqs,frames_per_subseq,data_dim*frame_size),dtype='float32')
    frame_labels = np.zeros((num_subseqs,frames_per_subseq,1),dtype='uint16')

    if subseq_labels is not None:
        #for seq,lab in zip(subseq_data,subseq_labels):
        for i in xrange(num_subseqs):
            seq = subseq_data[i]
            lab = subseq_labels[i]
            
            start = 0
            end = frame_size 

            for j in xrange(frames_per_subseq):
                frame_data[i][j] = seq[start:end].flatten()
                frame_labels[i][j] = lab[end-1]
                start += 1
                end += 1

        return frame_data, frame_labels

    else:
        all_frames = []
        for seq in subseq_data:
            N_o, Nv = seq.shape
            num_frames = N_o - frame_size + 1

            frames = np.zeros((num_frames,frame_size * Nv),dtype='float32')
            start = 0
            end = frame_size # exclusive

            for i in xrange(num_frames):
                frames[i] = seq[start:end].flatten()
                start += 1
                end += 1

            all_frames += [frames]
            #inds += [slice(curr_ind,curr_ind+num_frames)]
            #curr_ind += num_frame

        #return np.concatenate(all_frames), inds
        return all_frames 

def reframe_subseqs(frame_size,subseq_data,subseq_labels=None):
    '''
    construct overlapping frames of training data of the correct size

    input:
    frame_size - number of sequential observations per frame
    subseq_data - list of numpy arrays [N_o,Nv] 
                        where N_o is the number of observations 
                        and can be different for each list element
    subseq_labels - list of numpy arrays [N_o,Nl]  (optional)
                        where N_o is the number of observations 
                        and can be different for each list element

    output:
    sequence_data - numpy array with framed observations in rows
    sequence_ind - list of slices containing bounds of each sequence in the above array
    sequence_labels - [optional] 1D numpy array of labels for each frame
    '''


    
    num_subseqs,num_old_frames,old_frame_size = subseq_data.shape
    frames_per_old_frame = old_frame_size - frame_size + 1
    frame_data = np.zeros((num_subseqs,frames_per_old_frame,),dtype='float32')
    frame_labels = np.zeros((num_subseqs,frames_per_subseq,1),dtype='uint16')

    if subseq_labels is not None:
        #for seq,lab in zip(subseq_data,subseq_labels):
        for i in xrange(num_subseqs):
            seq = subseq_data[i]
            lab = subseq_labels[i]
            
            start = 0
            end = frame_size 

            for j in xrange(frames_per_subseq):
                frame_data[i][j] = seq[start:end].flatten()
                frame_labels[i][j] = lab[end-1]
                start += 1
                end += 1

        return frame_data, frame_labels

    else:
        all_frames = []
        for seq in subseq_data:
            N_o, Nv = seq.shape
            num_frames = N_o - frame_size + 1

            frames = np.zeros((num_frames,frame_size * Nv),dtype='float32')
            start = 0
            end = frame_size # exclusive

            for i in xrange(num_frames):
                frames[i] = seq[start:end].flatten()
                start += 1
                end += 1

            all_frames += [frames]
            #inds += [slice(curr_ind,curr_ind+num_frames)]
            #curr_ind += num_frame

        #return np.concatenate(all_frames), inds
        return all_frames 

def frames_to_subseq(frames,frames_per_subseq):
    '''
    convert [NxD] frame matrix to [SxFxD] subsequence matrix
    '''
    F = frames_per_subseq
    D = frames.shape[1]
    subseq = frames.reshape((-1,F,D))

    return subseq

def subseq_to_frames(subseq):
    '''
    convert [SxFxD] subsequence matrix to [NxD] frame matrix
    '''
    D = subseq.shape[2]
    frames = subseq.reshape((-1,D))

    return frames


class StatContainer(object):
    '''
    holds update stats for learning algorithms
    '''
    def __init__(self,num_epochs):
        '''
        input:
        names - list of strings naming each variable to hold stats for
        '''

        self.num_epochs = num_epochs

    def init_stats(self,stats_in):
        '''
        initialize the stats dictionaries with first sample
        '''

        names = stats_in[0].keys()

        zero_stats = np.zeros((self.num_epochs,3))

        param_stats = {}
        update_stats = {}
        stat_names = []

        for n in names:
            stat_names += [n]
            param_stats[n] = zero_stats.copy()
            update_stats[n] = zero_stats.copy()

        self.param_stats = param_stats
        self.update_stats = update_stats
        self.stat_names = stat_names
        self.epoch = -1

        self.add_stats(stats_in)

    def add_stats(self,stats_in):
        '''
        add a single epoch worth of stats to the array

        input:
        stats_in - 2-element list of dicts [param_stats,update_stats]
        '''
        self.epoch += 1

        param_stats_in = stats_in[0]
        update_stats_in = stats_in[1]

        for n in self.stat_names:
            self.param_stats[n][self.epoch] = param_stats_in[n]
            self.update_stats[n][self.epoch] = update_stats_in[n]

    def print_stats(self):
        '''
        print the stats from most recent epoch and output maximum ratio
        '''
        print 'update ratios'
        max_ratio = 0.0
        for n in self.stat_names:
            ratio = self.update_stats[n][self.epoch]/self.param_stats[n][self.epoch][1]
            max_ratio = max(ratio[1],max_ratio)
            print '\t%s:\t' % n,
            for v in ratio:
                print '% .8f ' % v,
            print

        print 'average magnitudes'
        for n in self.stat_names:
            val = self.param_stats[n][self.epoch]
            print '\t%s:\t' % n,
            for v in val:
                print '% .8f ' % v,
            print

        return max_ratio



def train_rbm(rbm,training_data,validation_data,params={}):
    '''
    train an rbm using contrastive divergence

    input:
    rbm - an initialized rbm class
    training_data - [N x Nv] matrix of N observations of Nv dimensions
    validation_data - [M x Nv] matrix of M observations of Nv dimensions
    params - dictionary of parameters

    params:
    num_epochs
    batch_size
    learning_rate - list of two same size lists: [targets,pcts]
    learning_momentum - list of two same size lists: [targets,pcts]
    gibbs_iters - list of two same size lists: [targets,pcts]
    weight_decay - L2 regularizer weight
    [disabled]adjust_rate - whether to dynamically adjust the learning rate
                    to retain an average max update ratio around 0.001
    update_target - target percentage of magnitude for updates
                    (used to adjust learning rate)
    decay_target - (learning_rate) percentage of pre-decay value to decay to
    decay_period - (learning_rate) percentage of num_epochs to decay over
    noisy - the noise level to use when drawing gibbs samples
                (see cdk methods of rbm classes)
    reshuffle - how often to reshuffle the data
                (set to > num_epochs to avoid reshuffling)
    '''


    # gather learning parameters --------------------------------------------
    num_epochs = params.get('num_epochs', 300) #one extra for initial stats
    batch_size = params.get('batch_size', 100)

    if num_epochs <= 0:
        print 'num_epochs <= 0, skipping\n'
        return None

    epoch_pct = np.linspace(0,1,num_epochs)

    # learning rate
    targets, pcts = params.get('learning_rate', [[0.0001, 0.0001],[0,1]])
    learning_rate = np.r_[np.interp(epoch_pct,pcts,targets),0]

    # momentum
    targets, pcts = params.get('learning_momentum', [[0.001,0.05,0.05],[0,0.05,1]])
    learning_momentum = np.interp(epoch_pct,pcts,targets)

    # gibbs iterations
    targets, pcts = params.get('gibbs_iters', [[1,1],[0,1]])
    K = np.round(np.interp(epoch_pct,pcts,targets)).astype('uint16')

    weight_decay = params.get('weight_decay', 0.001)
    #adjust_rate = params.get('adjust_rate', True)
    update_target = params.get('update_target', None)
    noisy = params.get('noisy', 0)
    reshuffle = params.get('reshuffle', num_epochs+1)

    train_type = params.get('train_type', 'cd')

    # learning rate decay parameter (used when update_target != None)
    decay_target = params.get('decay_target',0.1) # percentage of pre-decay value
    decay_period = params.get('decay_period',0.05) * num_epochs # time to decay over

    decay_start = int(num_epochs - decay_period)
    alpha = decay_target ** (1./decay_period) #autoregressive decay parameter

    # monitoring params
    save_hidden = False
    save_weights = False

    rng = np.random.RandomState(123) # init random number generator

    # ----------------------------------------------------------------------


    print '\n\nTraining RBM'
    print '-------------------------------------'
    print datetime.datetime.now()
    print type(rbm)
    print rbm.return_params() # model params
    print params # learning params
    print '-------------------------------------'
    print '\n'


    validation_size = validation_data.shape[0]
    training_size = training_data.shape[0]
    num_batches = training_size/batch_size
    num_leftover = training_size - num_batches * batch_size

    # collect batches
    batches = []
    for batch in xrange(num_batches):
        batches += [slice(batch*batch_size,(batch+1)*batch_size)]
    if num_leftover > 0:
        batches += [slice(num_batches*batch_size,num_batches*batch_size+num_leftover)]
        num_batches += 1

    stats = StatContainer(num_epochs+1)
    stats.init_stats(rbm.update_stats())


    param_id = int(time.time() - 1334729157)

    if save_hidden: 
        bh_at_epoch = np.zeros((num_epochs+1,rbm.Nh),dtype='float32')
        bv_at_epoch = np.zeros((num_epochs+1,rbm.Nv),dtype='float32')

        bh_at_epoch[0] = rbm.bh
        bv_at_epoch[0] = rbm.bv

        hidden_act = rbm.mean_field_h_given_v_frame(training_data)

        fig = pl.figure(1); pl.clf(); ax = fig.add_subplot(111)
        pl.imshow(hidden_act,cmap = 'gray', aspect='auto', interpolation='nearest')
        fig.savefig('results/activations_at_epoch_%.4u.png' % (0,))

    if save_weights:
        weights = rbm.W
        fig = pl.figure(1); pl.clf(); ax = fig.add_subplot(111)
        pl.imshow(weights,cmap = 'gray', aspect='auto', interpolation='nearest')
        fig.savefig('results/W_at_epoch_%.4u.png' % (0,))


    t0 = time.time()

    for epoch in xrange(0, num_epochs):
        t = time.time()

        if (epoch > 0) and (epoch % reshuffle == 0):
            print '\n-----RESHUFFLE TRAINING DATA-----\n'
            #perm = np.random.permutation(training_data.shape[0])
            perm = rng.permutation(training_data.shape[0])
            training_data = training_data[perm]

        
        print '\nepoch:', epoch+1
        print 'learning rate:', learning_rate[epoch]
        print 'learning momentum:', learning_momentum[epoch]
        if train_type == 'cd':
            print 'contrastive divergence:', K[epoch]
        elif train_type == 'discriminative':
            print 'discriminative updates'
        #if train_type == 'cd':
        #    print 'contrastive divergence:', K[epoch]
        #    rbm.cdk(K[epoch],training_data[batches[0]],
        #            learning_rate[epoch],learning_momentum[epoch],
        #            weight_decay,noisy)
        #elif train_type == 'discriminative':
        #    print 'discriminative updates'
        #    rbm.discriminative_train(training_data[batches[0]],
        #            learning_rate[epoch],learning_momentum[epoch],
        #            weight_decay)




        for batch in xrange(0,num_batches):
            if train_type == 'cd':
                rbm.cdk(K[epoch],training_data[batches[batch]],
                        learning_rate[epoch],learning_momentum[epoch],
                        weight_decay,noisy)
            elif train_type == 'discriminative':
                rbm.discriminative_train(training_data[batches[batch]],
                        learning_rate[epoch],learning_momentum[epoch],
                        weight_decay)
            if batch == 0:
                stats.add_stats(rbm.update_stats())


        max_update = stats.print_stats()

        print '\ntraining data'
        rbm.recon_error(training_data[:validation_size],K[epoch],print_output=True)
        print '\nvalidation data'
        rbm.recon_error(validation_data,K[epoch],print_output=True)


        if update_target is not None:
            if epoch < decay_start:
                # adjust learning rate to the sweet spot
                if max_update < 0.1 * update_target:
                    learning_rate[epoch+1] = learning_rate[epoch] * 2
                elif max_update > 10 * update_target:
                    learning_rate[epoch+1] = learning_rate[epoch] * 0.5
                elif max_update < 0.9 * update_target:
                    learning_rate[epoch+1] = learning_rate[epoch] * 1.1
                elif max_update > 1.2 * update_target:
                    learning_rate[epoch+1] = learning_rate[epoch] * 0.9
                else:
                    learning_rate[epoch+1] = learning_rate[epoch]
            else:
                # learning rate decays to a fraction of value before decay start
                learning_rate[epoch+1] = alpha * learning_rate[epoch]

        print 'time: ', time.time() - t, 'sec'

        if save_hidden: 
            bh_at_epoch[epoch+1] = rbm.bh
            bv_at_epoch[epoch+1] = rbm.bv

            hidden_act = rbm.mean_field_h_given_v_frame(training_data)

            fig = pl.figure(1); pl.clf(); ax = fig.add_subplot(111)
            pl.imshow(hidden_act,cmap = 'gray', aspect='auto', interpolation='nearest')
            fig.savefig('results/activations_at_epoch_%.4u.png' % (epoch+1,))

        if save_weights:
            weights = rbm.W
            fig = pl.figure(1); pl.clf(); ax = fig.add_subplot(111)
            pl.imshow(weights,cmap = 'gray', aspect='auto', interpolation='nearest')
            fig.savefig('results/W_at_epoch_%.4u.png' % (epoch+1,))


    total_time = time.time() - t0
    print '\ntotal time: ', total_time, 'sec'

    print '\ntraining data'
    train_error = rbm.recon_error(training_data[:validation_size],K[epoch],print_output=True)
    print '\nvalidation data'
    validation_error = rbm.recon_error(validation_data,K[epoch],print_output=True)

    print_training_info(params,rbm,train_error,validation_error,param_id,total_time)
    rbm.save_params('RBM_%u.pkl' % param_id)



    return stats

def train_neural_net(neural_net,training_data,training_labels, validation_data,validation_labels,params={}):
    '''
    train a neural net using backpropagation

    input:
    neural_net - a NeuralNet class with pre-trained weights
    training_data - [N x Nv] matrix of N observations of Nv dimensions
    validation_data - [M x Nv] matrix of M observations of Nv dimensions
    training_labels - [N x Nl] matrix of N observations of Nl dimensions
    validation_labels - [M x Nl] matrix of M observations of Nl dimensions
    params - dictionary of parameters

    params:
    num_epochs
    batch_size
    learning_rate - list of two same size lists: [targets,pcts]
    learning_momentum - list of two same size lists: [targets,pcts]
    weight_decay - L2 regularizer weight
    [disabled]adjust_rate - whether to dynamically adjust the learning rate
                    to retain an average max update ratio around 0.001
    update_target - target percentage of magnitude for updates
                    (used to adjust learning rate)
    decay_target - (learning_rate) percentage of pre-decay value to decay to
    decay_period - (learning_rate) percentage of num_epochs to decay over
    reshuffle - how often to reshuffle the data
    independent_rates - [bool] does each layer have its own dynamic learning rate?
    '''


    # gather learning parameters --------------------------------------------
    num_epochs = params.get('num_epochs', 300) 
    batch_size = params.get('batch_size', 100)

    if num_epochs <= 0:
        raise ValueError, 'num_epochs must be > 0'

    epoch_pct = np.linspace(0,1,num_epochs)

    # learning rate
    targets, pcts = params.get('learning_rate', [[0.0001, 0.0001],[0,1]])
     

    learning_rate = np.r_[np.interp(epoch_pct,pcts,targets),0]
    learning_rate = np.tile(learning_rate,[neural_net.num_layers,1]).T
    independent_rates = params.get('independent_rates', False)

    # momentum
    targets, pcts = params.get('learning_momentum', [[0.001,0.05,0.05],[0,0.05,1]])
    learning_momentum = np.interp(epoch_pct,pcts,targets)

    weight_decay = params.get('weight_decay', 0.001)
    update_target = params.get('update_target', None)
    reshuffle = params.get('reshuffle', num_epochs+1)

    # learning rate decay parameter (used when update_target != None)
    decay_target = params.get('decay_target',0.1) # percentage of pre-decay value
    decay_period = params.get('decay_period',0.1) * num_epochs # time to decay over

    decay_start = int(num_epochs - decay_period)
    alpha = decay_target ** (1./decay_period) #autoregressive decay parameter


    rng = np.random.RandomState(123) # init random number generator


    # ----------------------------------------------------------------------

    print '\n\nTraining Neural Net'
    print '-------------------------------------'
    print datetime.datetime.now()
    print type(neural_net)
    print neural_net.return_params() # model params
    print params # learning params
    print '-------------------------------------'
    print '\n'


    validation_size = validation_data.shape[0]
    training_size = training_data.shape[0]
    num_batches = training_size/batch_size
    num_leftover = training_size - num_batches * batch_size

    # collect batches
    batches = []
    for batch in xrange(num_batches):
        batches += [slice(batch*batch_size,(batch+1)*batch_size)]
    if num_leftover > 0:
        batches += [slice(num_batches*batch_size,num_batches*batch_size+num_leftover)]
        num_batches += 1

    stats = []
    for layer in neural_net.layers:
        stats += [StatContainer(num_epochs+1)]
        stats[-1].init_stats(layer.update_stats())

    #max_val_correct = 0.0
    corresponding_val_correct = 0.0
    min_val_entropy = np.inf

    param_id = int(time.time() - 1334729157)

    t0 = time.time()

    for epoch in xrange(0,num_epochs):
        t = time.time()

        if (epoch > 0) and (epoch % reshuffle == 0):
            print '\n-----RESHUFFLE TRAINING DATA-----\n'
            #perm = np.random.permutation(training_data.shape[0])
            perm = rng.permutation(training_data.shape[0])
            training_data = training_data[perm]
            training_labels = training_labels[perm]

        #neural_net.train_backprop(training_data[batches[0]],training_labels[batches[0]],
        #        learning_rate[epoch], learning_momentum[epoch], weight_decay)

        for batch in xrange(0,num_batches):
            neural_net.train_backprop(training_data[batches[batch]],training_labels[batches[batch]],
                    learning_rate[epoch], learning_momentum[epoch], weight_decay)
            if batch == 0:
                for i in xrange(neural_net.num_layers):
                    stats[i].add_stats(neural_net.layers[i].update_stats())

        print '\nepoch:', epoch+1
        print 'learning rate:', learning_rate[epoch]
        print 'learning momentum:', learning_momentum[epoch]
        print 'discriminative backpropagation'

        max_update = np.zeros(neural_net.num_layers)
        #max_update = 0
        for l in xrange(len(stats)-1,-1,-1):
            print 'Layer', l
            max_update[l] = stats[l].print_stats()

        if not independent_rates:
            max_update[:] = np.max(max_update)

        print '\ntraining data'
        train_correct, train_entropy, train_lab_probs, train_output = neural_net.classification_error(training_data[:validation_size],
                training_labels[:validation_size],print_output=True)
        print '\nvalidation data'
        val_correct, val_entropy, val_lab_probs, val_output = neural_net.classification_error(validation_data,
                validation_labels,print_output=True)
        #if val_correct > max_val_correct:
        if val_entropy < min_val_entropy:
            neural_net.save_params('NN_%u_val.pkl'%param_id)
            #max_val_correct = val_correct
            min_val_entropy = val_entropy
            corresponding_val_correct = val_correct
            best_val_output = val_output
            best_train_output = train_output
            best_val_time = time.time() - t0
        #print 'max validation accuracy', max_val_correct
        print 'min validation entropy', min_val_entropy
        print 'corresponding accuracy', corresponding_val_correct


        if update_target is not None:
            if epoch < decay_start:
                for l in xrange(neural_net.num_layers):
                    # adjust learning rate to the sweet spot
                    if max_update[l] < 0.1 * update_target:
                        learning_rate[epoch+1,l] = learning_rate[epoch,l] * 2
                    elif max_update[l] > 10 * update_target:
                        learning_rate[epoch+1,l] = learning_rate[epoch,l] * 0.5
                    elif max_update[l] < 0.9 * update_target:
                        learning_rate[epoch+1,l] = learning_rate[epoch,l] * 1.1
                    elif max_update[l] > 1.2 * update_target:
                        learning_rate[epoch+1,l] = learning_rate[epoch,l] * 0.9
                    else:
                        learning_rate[epoch+1,l] = learning_rate[epoch,l]
            else:
                # learning rate decays to a fraction of value before decay start
                learning_rate[epoch+1] = alpha * learning_rate[epoch]

        print 'time: ', time.time() - t, 'sec'


    neural_net.save_params('NN_%u.pkl'%param_id)
    total_time = time.time() - t0
    print '\ntotal time: ', total_time, 'sec'


    print_training_info(params,neural_net,best_train_output,best_val_output,param_id,total_time,best_val_time=best_val_time)

    return stats, param_id

def print_training_info(learning_params,model,final_train_error,final_val_error,param_id,total_time=None,best_val_time=None,filename=None):
    '''
    print trianing summary

    input:
    learning_params - struct of params used in learning
    model - instance of learning model
    final_train_error - string containing training error
    final_val_error - string containing validation error
    filename - to save to
    '''
    if filename is None:
        filename = 'training_info.txt'
    fp = open(filename,'a')
    fp.write(str(datetime.datetime.now()) + '\n')
    fp.write(str(type(model)) + '\n')
    model_params = model.return_params()
    fp.write(model_params + '\n')
    fp.write(str(learning_params) + '\n')
    fp.write('best training error:\n')
    fp.write(final_train_error + '\n')
    fp.write('best validation error:\n')
    fp.write(final_val_error + '\n')
    if total_time is not None:
        fp.write('total time: %g\n' % total_time)
    if best_val_time is not None:
        fp.write('best validation time: %g\n' % best_val_time)
    fp.write('parameter id: %u\n'%param_id)
    fp.write('\n\n')

    fp.close()

def print_test_info(neural_net,test_data,test_labels,filename=None,modulo=None): 
    correct, entropy, corr_lab_probs, output = neural_net.classification_error(test_data,
            test_labels,print_output=True,modulo=modulo)

    if filename is None:
        filename = 'training_info.txt'
    fp = open(filename,'a')
    fp.write(str(datetime.datetime.now()) + '\n')
    fp.write('test error:\n')
    fp.write(output + '\n')
    fp.write('\n\n')

    fp.close()

    return correct, entropy


class ViterbiDecoder:
    ''' 
    Viterbi decoding using transition probabilities and posterior probability vectors

    params:
    -----------------
    trans_matrix - matrix of transition probabilities
    '''
    def __init__(self,num_states,slot=0.9,prior=None):
        '''
        inputs:
        num_states - number of states
        slot - probability of a transition to the next sequential state
        prior - array containing initial state probabilities
        '''

        dtype = 'float32'

        epsilon = float(1 - slot)/(num_states-1)

        trans = epsilon * np.ones((num_states,num_states),dtype=dtype)
        for i in xrange(num_states-1):
            trans[i,i+1] = slot
        trans[-1,0] = slot

        if prior is None:
            prior = 1./num_states * np.ones(num_states,dtype=dtype)
        else:
            prior = prior.astype(dtype)


        # params -----------------------
        self.trans = trans.copy()
        self.prior = prior.copy()
        self.log_trans = np.log(trans) # transition prob matrix
        self.log_prior = np.log(prior) # initial probability of each state
        self.alpha = None # log-prob of backwards-decoded state
        self.n = -1  #frame index
        self.dtype = dtype

    def process(self,posterior):
        '''
        compute most likely state given current state posterior
        '''

        # increment frame number
        self.n += 1

        posterior = posterior.copy()

        # prevent -inf log
        posterior[posterior <= 0] = np.finfo(self.dtype).tiny
        log_posterior = np.log(posterior)

        if self.alpha == None: 
            # assign alpha
            self.alpha = log_posterior + self.log_prior
            state = np.argmax(self.alpha)

        else:

            self.alpha = log_posterior + np.max( self.alpha.reshape((-1,1)) + self.log_trans, axis=0 )
            state = np.argmax(self.alpha)

        return state, self.alpha
    def reset(self):
        self.n = -1
        self.alpha = None

class PatternMatcher(object):
    '''
    Simple drum pattern matcher
    '''
    def __init__(self):
        self.dtype = 'float32'

        time_sig = 4
        tatums_per_beat = 4
        num_tatums = time_sig * tatums_per_beat
        num_drums = 3
        num_measures = 2
        

        pattern = []
        pattern.append([ [12,0,0], [1,5,4], [0,0,0], [5.7,5,4] ])
        pattern.append([ [10,0,0], [0,0,0], [1.4,7.1,5.8], [1.3,2,1.5] ])
        for i in xrange(len(pattern)):

            # fill off-beat tatums with zeros
            a = np.zeros((num_tatums,num_drums))
            for j in xrange(len(pattern[i])):
                a[j*4] = pattern[i][j]


            a = a.flatten()

            # circular shift pattern by every tatum amount
            A = np.zeros((num_tatums,a.size))
            for j in xrange(num_tatums):
                A[j] = np.roll(a,-j*num_drums)

            # extend pattern to two measures
            A = np.tile(A,num_measures)
            # add current tatum to pattern
            A = np.append(A,A[:,0:num_drums],axis=1)

            pattern[i] = A.astype(self.dtype)

        self.patterns = pattern
        self.pattern_matrix = np.concatenate(self.patterns,axis=0)
        self.num_tatums = num_tatums


    def get_pattern_matrix(self):

        return self.pattern_matrix

    def return_labels(self,test_data,modulo=None):
        '''
        return predicted tatum labels
        '''
        results = []
        for pattern in self.patterns:
            pred = np.dot(test_data, pattern.T)
            if modulo is not None:
                N = test_data.shape[0]
                L = self.num_tatums
                pred = pred.reshape([N,L/modulo,modulo]).sum(axis=1)
            results.append(pred[None,:,:])

        results = np.concatenate(results,axis=0)
        rel_strength = np.max(results,axis=0)
        rel_strength = rel_strength/np.sum(rel_strength,axis=1)[:,None]

        predicted_labels = np.argmax(rel_strength, axis=1) + 1

        return predicted_labels, rel_strength


    def classification_error(self,test_data,test_labels,modulo=None,ratio_type=None):
        '''
        return predicted label accuracy
        '''
        predicted_labels,rel_strength = self.return_labels(test_data,modulo=modulo)

        if modulo is not None:
            N = test_data.shape[0]
            L = self.num_tatums
            test_labels_mod = ((test_labels - 1) % modulo) + 1
            true_labels = test_labels_mod.flatten()
        else:
            true_labels = test_labels.flatten()

        percent_correct = np.mean(predicted_labels == true_labels)


        ind = np.arange(test_labels.shape[0])
        correct_rel_strength = rel_strength[ind,true_labels-1]
        eps = np.finfo('float32').tiny
        correct_rel_strength[correct_rel_strength <= 0] = eps
        rel_entropy = np.mean(-np.log(correct_rel_strength))

        if ratio_type is 'second':
            sorted_rel_strength = np.sort(rel_strength,axis=1)
            #top_strength_ratio = sorted_rel_strength[:,-1]/correct_rel_strength
            second_strength_ratio = correct_rel_strength/(sorted_rel_strength[:,-2]+eps)

            return percent_correct, rel_entropy, second_strength_ratio
        elif ratio_type is 'margin':
            # if pred_lab is correct, compute ratio of top:second probs
            # if incorrect, compute ratio of corr:top probs
            sorted_rel_strength = np.sort(rel_strength,axis=1)
            strength_margin = sorted_rel_strength[:,-1]/sorted_rel_strength[:,-2]
            incorr_lab_mask = predicted_labels != true_labels
            strength_margin[incorr_lab_mask] = correct_rel_strength[incorr_lab_mask]/sorted_rel_strength[incorr_lab_mask,-1]

            return percent_correct, rel_entropy, strength_margin 
        else:
            return percent_correct, rel_entropy, correct_rel_strength
