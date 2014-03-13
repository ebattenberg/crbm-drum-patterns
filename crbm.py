import cPickle as pkl
import pdb
import datetime
import time

import numpy as np
import pylab as pl
import scipy.stats
import scipy.special
from scipy.special import gamma
from scipy.misc import factorial

import gnumpy as gp

import data_helper



class RBM(object):
    '''
    Restricted Boltzmann Machine (RBM) using numpy
    '''
    def __init__(self, params={}):
        '''
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

        bv: visible bias

        other params:
        -----------------

        W: weight between current hidden and visible units (undirected)
            [Nv x Nh]

        bh: hidden bias
        '''

        dtype = 'float32'


        Nv = params['Nv']
        Nh = params['Nh']
        vis_unit = params.get('vis_unit','binary')
        vis_scale = params.get('vis_scale')
        bv = params.get('bv')

        Th = params.get('Th',0)


        if vis_unit not in ['binary','linear']:
            raise ValueError, 'Unknown visible unit type %s' % vis_unit
        if vis_unit == 'linear':
            if vis_scale is None:
                raise ValueError, 'Must set vis_scale for linear visible units'
        elif vis_unit == 'binary':
            vis_scale = 1.


        # W is initialized with `initial_W` which is uniformly sampled
        # from -4.*sqrt(6./(Nv+Nh)) and 4.*sqrt(6./(Nh+Nv))
        # the output of uniform if converted using asarray to dtype
        W = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv+Nh)),
            high  =  4*np.sqrt(6./(Nv+Nh)),
            size  =  (Nv, Nh)),
            dtype =  dtype)
        W = gp.garray(W)

        bh = gp.zeros(Nh)


        if bv is None :
            bv = gp.zeros(Nv)
        else:
            bv = gp.garray(bv)

        # params -------------------------------------------
        self.dtype  = 'float32' 

        self.Nv     = Nv        # num visible units
        self.Nh     = Nh        # num hidden units

        self.Th     = Th        # used for framing input

        self.vis_unit = vis_unit    # type of visible output unit
        self.vis_scale = vis_scale  # scale of linear output units

        self.W      = W     # vis<->hid weights
        self.bv     = bv    # vis bias
        self.bh     = bh    # hid bias

        self.W_update = gp.zeros((Nv,Nh))
        self.bh_update = gp.zeros((Nh,))
        self.bv_update = gp.zeros((Nv,))

        self.params     = [ 'dtype',
                            'vis_unit','vis_scale',
                            'Nv','Nh',
                            'W','bh','bv']

    def save_params(self,filename=None):
        '''
        save parameters to file
        '''
        if filename is None:
            fileid = np.random.randint(100000)
            filename = 'RBM_%u.pkl' % fileid
            
        params_out = {}  
        for p in self.params:
            val = vars(self)[p]
            if type(val) is gp.garray:
                params_out[p] = val.as_numpy_array()
            else:
                params_out[p] = val
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
        for key,value in params_in.iteritems():
            vars(self)[key] = value

        Nv,Nh = self.Nv,self.Nh
        dtype = self.dtype

        self.W_update = gp.zeros((Nv,Nh))
        self.bh_update = gp.zeros((Nh,))
        self.bv_update = gp.zeros((Nv,))

        self.W = gp.garray(self.W)
        self.bh = gp.garray(self.bh)
        self.bv = gp.garray(self.bv)

    def return_params(self):
        '''
        return a formatted string containing scalar parameters
        '''
        output = 'Nv=%u, Nh=%u, vis_unit=%s, vis_scale=%0.2f' \
                % (self.Nv,self.Nh,self.vis_unit,self.vis_scale)
        return output

    def mean_field_h_given_v(self,v):
        '''
        compute mean-field reconstruction of P(h=1|v)
        '''
        prob =  sigmoid(self.bh + gp.dot(v, self.W))
        return prob

    def mean_field_v_given_h(self,h):
        '''
        compute mean-field reconstruction of P(v|h)
        '''
        x = self.bv + gp.dot(h, self.W.T)
        if self.vis_unit == 'binary':
            return sigmoid(x)
        elif self.vis_unit == 'linear':
            return log_1_plus_exp(x) - log_1_plus_exp(x-self.vis_scale)
        return prob

    def sample_h_given_v(self,v):
        '''
        compute samples from P(h|v)
        '''
        prob = self.mean_field_h_given_v(v)
        samples = prob.rand() < prob

        return samples, prob

    def sample_v_given_h(self,h):
        '''
        compute samples from P(v|h)
        '''
        if self.vis_unit == 'binary':
            mean = self.mean_field_v_given_h(h)
            samples = mean.rand() < mean
            return samples, mean

        elif self.vis_unit == 'linear':
            x = self.bv + gp.dot(h, self.W.T)
            # variance of noise is sigmoid(x) - sigmoid(x - vis_scale)
            stddev = gp.sqrt(sigmoid(x) - sigmoid(x - self.vis_scale)) 
            mean =  log_1_plus_exp(x) - log_1_plus_exp(x-self.vis_scale)
            noise = stddev * gp.randn(x.shape)
            samples = mean + noise
            samples[samples < 0] = 0
            samples[samples > self.vis_scale] = self.vis_scale
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


        rate = float(rate)
        if momentum > 0.0:
            momentum = float(momentum)
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
        W_grad = N_inv * (gp.dot(vk.T, hk) - gp.dot(v0.T, h0))
        bv_grad = gp.mean(vk - v0,axis=0)
        bh_grad = gp.mean(hk - h0,axis=0)

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
        '''
        compute K-step reconstruction error
        '''

        vk_mean = self.gibbs_samples(K,v0_data,noisy=0)
        recon_error = gp.mean(gp.abs(v0_data - vk_mean))

        if print_output:
            output = '%30s %6.5f' % ('vis error:', recon_error/self.vis_scale)
            print output
            return output
        else: 
            return recon_error

    def update_stats(self):
        W_stats = [gp.min(self.W),gp.mean(gp.abs(self.W)),gp.max(self.W)]
        bh_stats = [gp.min(self.bh),gp.mean(gp.abs(self.bh)),gp.max(self.bh)]
        bv_stats = [gp.min(self.bv),gp.mean(gp.abs(self.bv)),gp.max(self.bv)]

        W_update_stats = [gp.min(self.W_update), gp.mean(gp.abs(self.W_update)), gp.max(self.W_update)]
        bh_update_stats = [gp.min(self.bh_update), gp.mean(gp.abs(self.bh_update)), gp.max(self.bh_update)]
        bv_update_stats = [gp.min(self.bv_update), gp.mean(gp.abs(self.bv_update)), gp.max(self.bv_update)]

        param_stats = dict(W=W_stats,bh=bh_stats,bv=bv_stats)
        update_stats = dict(W=W_update_stats,
                bh=bh_update_stats,bv=bv_update_stats)

        return [param_stats, update_stats]

class LRBM(RBM):
    '''
    Labeled Restricted Boltzmann Machine
    '''
    def __init__(self, params={}):
        '''
        input:
        -----------------

        (in addition to those defined in RBM class)

        Nl: number of label units (group of softmax units)
        '''

        dtype = 'float32'

        super(LRBM,self).__init__(params)

        bv = params.get('bv')
        Nl = params['Nl']
        Nv = self.Nv
        Nh = self.Nh


        # add label units to visible units

        # W is initialized with uniformly sampled data
        # from -4.*sqrt(6./(Nv+Nh)) and 4.*sqrt(6./(Nh+Nv))
        W = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv+Nl+Nh)),
            high  =  4*np.sqrt(6./(Nv+Nl+Nh)),
            size  =  (Nv+Nl, Nh)),
            dtype =  dtype)
       
        W = gp.garray(W)


        if bv is None :
            bv = gp.zeros((Nv+Nl))
        else:
            bv = gp.garray(bv)


        # new label-unit params -------------------------------------------
        self.Nl     = Nl        # num label units

        self.W      = W     # (vis+lab)<->hid weights
        self.bv     = bv    # vis bias

        self.W_update = gp.zeros((Nv+Nl,Nh))
        self.bv_update = gp.zeros((Nv+Nl,))

        self.params     += ['Nl']

    def load_params(self,filename):
        '''load parameters from file'''
        super(LRBM,self).load_params(filename)

        Nv,Nh,Nl,= self.Nv,self.Nh,self.Nl
        dtype = self.dtype
        self.W_update = gp.zeros((Nv+Nl,Nh))
        self.bv_update = gp.zeros((Nv+Nl,))

    def save_params(self,filename=None):
        '''save parameters to file'''
        if filename is None:
            fileid = np.random.randint(100000)
            filename = 'LRBM_%u.pkl' % fileid

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

        x = gp.concatenate((x_vis,x_lab),axis=axis)

        return x

    def mean_field_v_given_h(self,h):
        '''compute mean-field reconstruction of P(v|h)'''
        x = self.bv + gp.dot(h, self.W.T)
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
            vis_samp = vis_mean.rand() < vis_mean

        elif self.vis_unit == 'linear':
            x = self.bv + gp.dot(h, self.W.T)
            x_vis,x_lab = self.separate_vis_lab(x)
            # variance of noise is sigmoid(x_vis) - sigmoid(x_vis - vis_scale)
            vis_stddev = gp.sqrt(sigmoid(x_vis) - sigmoid(x_vis - self.vis_scale)) 
            vis_mean =  log_1_plus_exp(x_vis) - log_1_plus_exp(x_vis-self.vis_scale)
            vis_noise = stddev * gp.random.standard_normal(size=x.shape)
            vis_samp = vis_mean + vis_noise
            vis_samp[vis_samp < 0] = 0
            vis_samp[vis_samp > self.vis_scale] = self.vis_scale

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

        z = b_hid + gp.dot(v_vis,W_vis)
        z = z.reshape(z.shape + (1,))
        z = z + W_lab.T.reshape((1,) + W_lab.T.shape)

        hidden_terms = -gp.sum(log_1_plus_exp(z), axis=1)
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

        ind, true_labs = gp.where(v_lab == 1)


        #scale = float(rate / N)
        N_inv = 1./N



        # prob_scale = (1-p_d) for correct label and -p_d for other labels
        prob_scale = -p_d
        prob_scale[ind,true_labs] += 1
        ps_broad = prob_scale.reshape((N,1,self.Nl)) # make broadcastable across h_d

        p_h_sum = gp.sum(ps_broad * h_d, axis=2)

        # compute gradients ----------------------------------------------
        # W = [w,r]
        w_grad = gp.dot(v_vis.T, p_h_sum)               # vis<-->hid
        r_grad = gp.sum( ps_broad * h_d, axis=0 ).T     # lab<-->hid
        W_grad = N_inv * self.join_vis_lab(w_grad,r_grad,axis=0)# [vis,lab]<-->hid

        bh_grad = gp.mean(p_h_sum,axis=0)                # -->hid

        # bv = [bvv,bvl]                                # -->[vis,lab]
        bvv,bvl = self.separate_vis_lab(self.bv)
        bvv_grad = gp.zeros(bvv.shape)                   # -->vis
        bvl_grad = gp.mean(prob_scale,axis=0)            # -->lab
        # ---------------------------------------------------------------




        if weight_decay > 0.0:
            W_grad += -weight_decay * self.W

        #Wv_grad = self.join_vis_lab(Wvv_grad,Wvl_grad)
        bv_grad = self.join_vis_lab(bvv_grad,bvl_grad)

        rate = float(rate)


        if momentum > 0.0:
            momentum = float(momentum)
            self.W_update = momentum * self.W_update + rate*W_grad
            self.bh_update = momentum * self.bh_update + rate*bh_grad
            self.bv_update = momentum * self.bv_update + rate*bv_grad
        else:
            self.W_update = rate*W_grad
            self.bh_update = rate*bh_grad
            self.bv_update = rate*bv_grad



        self.W += self.W_update
        self.bh += self.bh_update
        self.bv += self.bv_update

    def recon_error(self,v0_data,K=1,print_output=False):
        '''compute K-step reconstruction error'''

        vk_mean = self.gibbs_samples(K,v0_data,noisy=0)

        v0_vis,v0_lab = self.separate_vis_lab(v0_data)
        vk_vis,vk_lab = self.separate_vis_lab(vk_mean)
        vis_error = gp.mean(gp.abs(v0_vis - vk_vis))
        lab_error = gp.mean(gp.abs(v0_lab - vk_lab))

        lab_probs = self.label_probabilities(v0_data)
        #pred_labs = gargmax(lab_probs)
        pred_labs = lab_probs.argmax(axis=1)
        ind, true_labs = gp.where(v0_lab == 1)
        percent_correct = gp.mean(pred_labs == true_labs)
        cross_entropy = -gp.mean(gp.log(lab_probs[ind,true_labs]))
        #prob_error = gp.mean(gp.abs(1. - lab_probs[ind,true_labs]))

        if print_output:
            output = '%30s %6.5f' % ('vis error:', vis_error/self.vis_scale) + '\n'
            output += '%30s %6.5f' % ('lab error:', lab_error) + '\n'
            #output += '%30s %6.5f' % ('prob error:', prob_error) + '\n'
            output += '%30s %6.5f' % ('cross entropy:', cross_entropy) + '\n'
            output += '%30s %6.5f' % ('class correct:', percent_correct)
            print output
            return output
        else: 
            return percent_correct, cross_entropy, lab_error, vis_error/self.vis_scale

class CRBM(object):
    '''Conditional Restricted Boltzmann Machine (CRBM) using gnumpy '''
    def __init__(self, params={}):
        '''
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

        '''

        dtype = 'float32'

        Nv = params['Nv']
        Nh = params['Nh']

        Tv = params['Tv']
        Th = params['Th']
        T = max(Tv,Th)
        period = params.get('period',T)

        vis_unit = params.get('vis_unit','binary')
        vis_scale = params.get('vis_scale')
        bv = params.get('bv')
        Wv_scale = params.get('Wv_scale',0.01)

        if vis_unit not in ['binary','linear']:
            raise ValueError, 'Unknown visible unit type %s' % vis_unit
        if vis_unit == 'linear':
            if vis_scale is None:
                raise ValueError, 'Must set vis_scale for linear visible units'
        elif vis_unit == 'binary':
            vis_scale = 1.


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
        W = gp.garray(W)

        Wv = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv*Tv+Nv)),
            high  =  4*np.sqrt(6./(Nv*Tv+Nv)),
            size  =  (Nv*Tv, Nv)),
            dtype =  dtype)
        Wv = gp.garray(Wv)

        Wh = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv*Th+Nh)),
            high  =  4*np.sqrt(6./(Nv*Th+Nh)),
            size  =  (Nv*Th, Nh)),
            dtype =  dtype)
        Wh = gp.garray(Wh)


        bh = gp.zeros(Nh)

        if bv is None :
            bv = gp.zeros(Nv)
        else:
            bv = gp.garray(bv)

        # params -------------------------------------------
        self.dtype  = 'float32' 

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

        self.W_update = gp.zeros((Nv,Nh))
        self.Wv_update = gp.zeros((Nv*Tv,Nv))
        self.Wh_update = gp.zeros((Nv*Th,Nh))
        self.bh_update = gp.zeros((Nh,))
        self.bv_update = gp.zeros((Nv,))

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
            val = vars(self)[p]
            if type(val) is gp.garray:
                params_out[p] = val.as_numpy_array()
            else:
                params_out[p] = val
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

        self.W_update = gp.zeros((Nv,Nh))
        self.Wv_update = gp.zeros((Nv*Tv,Nv))
        self.Wh_update = gp.zeros((Nv*Th,Nh))
        self.bh_update = gp.zeros((Nh,))
        self.bv_update = gp.zeros((Nv,))

        self.W = gp.garray(self.W)
        self.Wv = gp.garray(self.Wv)
        self.Wh = gp.garray(self.Wh)
        self.bh = gp.garray(self.bh)
        self.bv = gp.garray(self.bv)

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
        prob =  sigmoid(h_bias + gp.dot(v, self.W))
        return prob

    def mean_field_h_given_v_frame(self,v_input):
        '''
        compute mean-field reconstruction of P(ht=1|vt,v<t) 
        and compute h_bias from data

        input:
        v_frames - contains [v_past, v_curr] in a matrix
        '''
        v,vv_past,vh_past = self.extract_data(v_input)
        h_bias = self.bh + gp.dot(vh_past,self.Wh)
        return sigmoid(h_bias + gp.dot(v, self.W))

    def mean_field_v_given_h(self,h,v_bias):
        '''compute mean-field reconstruction of P(vt|ht,v<t)'''
        x = v_bias + gp.dot(h, self.W.T)
        if self.vis_unit == 'binary':
            return sigmoid(x)
        elif self.vis_unit == 'linear':
            return log_1_plus_exp(x) - log_1_plus_exp(x-self.vis_scale)
        return prob

    def sample_h_given_v(self,v,h_bias):
        '''compute samples from P(ht=1|vt,v<t)'''
        prob = self.mean_field_h_given_v(v,h_bias)
        samples = prob.rand() < prob

        return samples, prob

    def sample_v_given_h(self,h,v_bias):
        '''compute samples from P(vt|ht,v<t)'''
        if self.vis_unit == 'binary':
            mean = self.mean_field_v_given_h(h,v_bias)
            samples = mean.rand() < mean
            return samples, mean

        elif self.vis_unit == 'linear':
            x = v_bias + gp.dot(h, self.W.T)
            # variance of noise is sigmoid(x) - sigmoid(x - vis_scale)
            stddev = gp.sqrt(sigmoid(x) - sigmoid(x - self.vis_scale)) 
            mean =  log_1_plus_exp(x) - log_1_plus_exp(x-self.vis_scale)
            noise = stddev * gp.randn(x.shape)
            samples = mean + noise
            samples *= samples > 0
            samples_over = samples - self.vis_scale
            samples_over *= samples_over > 0
            samples_over -= samples_over
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
        noisy - 0 = use hidden samples, but means as final values
                1 = use visible and hidden samples, but means as final values
                2 = use visible and hidden samples, samples for final hidden values, means for final visibles
                3 = use samples everywhere.
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
                vk_samp, vk_mean = self.sample_v_given_h(hk_samp,v_bias)
                hk_samp, hk_mean = self.sample_h_given_v(vk_samp,h_bias)
            h0 = h0_mean # <--
            vk = vk_mean
            hk = hk_mean
        elif noisy == 2:
            for k in xrange(K): # vk_samp <--> hk_samp
                vk_samp, vk_mean = self.sample_v_given_h(hk_samp,v_bias)
                hk_samp, hk_mean = self.sample_h_given_v(vk_samp,h_bias)
            h0 = h0_samp
            vk = vk_mean # <--
            hk = hk_samp # <--
        elif noisy == 3:
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

        rate = float(rate)
        if momentum > 0.0:
            momentum = float(momentum)
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

        W_grad = N_inv * (gp.dot(vk.T, hk) - gp.dot(v0.T, h0))
        Wv_grad = N_inv * (gp.dot(vv_past.T, vk) - gp.dot(vv_past.T, v0))
        Wh_grad = N_inv * (gp.dot(vh_past.T, hk) - gp.dot(vh_past.T, h0))

        bv_grad = gp.mean(vk - v0,axis=0)
        bh_grad = gp.mean(hk - h0,axis=0)

        return W_grad,Wv_grad,Wh_grad,bv_grad,bh_grad

    def compute_dynamic_bias(self,v_input):
        v_data,vv_past,vh_past = self.extract_data(v_input)
        v_bias = self.bv + gp.dot(vv_past,self.Wv)
        h_bias = self.bh + gp.dot(vh_past,self.Wh)

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
        sequence = gp.garray(sequence)

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
        recon_error = gp.mean(gp.abs(v0_data - vk_mean))

        if print_output:
            output = '%30s %6.5f' % ('vis error:', recon_error/self.vis_scale)
            print output
            return output
        else: 
            return recon_error

    def update_stats(self):
        W_stats = [gp.min(self.W),gp.mean(gp.abs(self.W)),gp.max(self.W)]
        Wv_stats = [gp.min(self.Wv),gp.mean(gp.abs(self.Wv)),gp.max(self.Wv)]
        Wh_stats = [gp.min(self.Wh),gp.mean(gp.abs(self.Wh)),gp.max(self.Wh)]
        bh_stats = [gp.min(self.bh),gp.mean(gp.abs(self.bh)),gp.max(self.bh)]
        bv_stats = [gp.min(self.bv),gp.mean(gp.abs(self.bv)),gp.max(self.bv)]

        W_update_stats = [gp.min(self.W_update), gp.mean(gp.abs(self.W_update)), gp.max(self.W_update)]
        Wv_update_stats = [gp.min(self.Wv_update), gp.mean(gp.abs(self.Wv_update)), gp.max(self.Wv_update)]
        Wh_update_stats = [gp.min(self.Wh_update), gp.mean(gp.abs(self.Wh_update)), gp.max(self.Wh_update)]
        bh_update_stats = [gp.min(self.bh_update), gp.mean(gp.abs(self.bh_update)), gp.max(self.bh_update)]
        bv_update_stats = [gp.min(self.bv_update), gp.mean(gp.abs(self.bv_update)), gp.max(self.bv_update)]

        param_stats = dict(W=W_stats,Wv=Wv_stats,Wh=Wh_stats,bh=bh_stats,bv=bv_stats)
        update_stats = dict(W=W_update_stats,
                Wv=Wv_update_stats,Wh=Wh_update_stats,bh=bh_update_stats,bv=bv_update_stats)

        return [param_stats, update_stats]

class LCRBM(CRBM):
    '''Labeled Conditional Restricted Boltzmann Machine (CRBM) using numpy '''
    def __init__(self, params={}):
        '''
        input:
        -----------------

        (in addition to those defined in CRBM class)

        Nl: number of label units (group of softmax units)
        '''

        super(LCRBM,self).__init__(params)

        dtype = 'float32'

        bv = params.get('bv')
        Nl = params['Nl']
        Nv = self.Nv
        Nh = self.Nh
        Tv = self.Tv


        # add label units to visible units

        # W is initialized with uniformly sampled data
        # from -4.*sqrt(6./(Nv+Nh)) and 4.*sqrt(6./(Nh+Nv))
        W = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv+Nl+Nh)),
            high  =  4*np.sqrt(6./(Nv+Nl+Nh)),
            size  =  (Nv+Nl, Nh)),
            dtype =  dtype)
        W = gp.garray(W)

        Wv = np.asarray( np.random.uniform(
            low   = -4*np.sqrt(6./(Nv*Tv+Nv)),
            high  =  4*np.sqrt(6./(Nv*Tv+Nv)),
            size  =  (Nv*Tv, Nv)),
            dtype =  dtype)
        Wv = gp.garray(Wv)


        if bv is None :
            bv = gp.zeros(Nv+Nl)
        else:
            bv = gp.garray(bv)

        cumsum = np.zeros((Nl,Nl),dtype=self.dtype)
        cumsum[np.triu_indices(Nl)] = 1
        cumsum = gp.garray(cumsum)

        # new label-unit params -------------------------------------------
        self.Nl     = Nl        # num label units
        self.cumsum =  cumsum   # upper triangular matrix for gpu cumsum

        self.W      = W     # (vis+lab)<->hid weights
        self.Wv     = Wv    # vis->vis delay weights
        self.bv     = bv    # vis bias

        self.W_update = gp.zeros((Nv+Nl,Nh))
        self.Wv_update = gp.zeros((Nv*Tv,Nv))
        self.bv_update = gp.zeros((Nv+Nl,))

        self.params     += ['Nl','cumsum']

    def load_params(self,filename):
        '''load parameters from file'''
        super(LCRBM,self).load_params(filename)

        Nv,Nh,Nl,Tv = self.Nv,self.Nh,self.Nl,self.Tv
        dtype = self.dtype
        self.W_update = gp.zeros(self.W.shape)
        self.Wv_update = gp.zeros(self.Wv.shape)
        self.bv_update = gp.zeros(self.bv.shape)

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

        x = gp.concatenate((x_vis,x_lab),axis=axis)

        return x

    def mean_field_v_given_h(self,h,v_bias):
        '''compute mean-field reconstruction of P(vt|ht,v<t)'''
        x = v_bias + gp.dot(h, self.W.T)
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
            vis_samp = vis_mean.rand() < vis_mean

        elif self.vis_unit == 'linear':
            x = v_bias + gp.dot(h, self.W.T)
            x_vis,x_lab = self.separate_vis_lab(x)
            # variance of noise is sigmoid(x_vis) - sigmoid(x_vis - vis_scale)
            vis_stddev = gp.sqrt(sigmoid(x_vis) - sigmoid(x_vis - self.vis_scale)) 
            vis_mean =  log_1_plus_exp(x_vis) - log_1_plus_exp(x_vis-self.vis_scale)
            vis_noise = vis_stddev * gp.randn(x_vis.shape)
            vis_samp = vis_mean + vis_noise
            vis_samp *= vis_samp > 0
            vis_over = vis_samp - self.vis_scale
            vis_over *= vis_over > 0
            vis_samp -= vis_over

            lab_mean = softmax(x_lab)
            means = self.join_vis_lab(vis_mean,lab_mean)

        #lab_samp = sample_categorical(lab_mean,self.cumsum)
        lab_samp = lab_mean
        samples = self.join_vis_lab(vis_samp,lab_samp)

        return samples, means

    def compute_gradients(self,v_input,h0,vk,hk):
        v0,vv_past,vh_past = self.extract_data(v_input) 
        v0_vis,v0_lab = self.separate_vis_lab(v0)
        vk_vis,vk_lab = self.separate_vis_lab(vk)

        N = v0.shape[0]
        N_inv = 1./N

        W_grad = N_inv * (gp.dot(vk.T, hk) - gp.dot(v0.T, h0))
        Wv_grad = N_inv * (gp.dot(vv_past.T, vk_vis) - gp.dot(vv_past.T, v0_vis))
        Wh_grad = N_inv * (gp.dot(vh_past.T, hk) - gp.dot(vh_past.T, h0))

        bv_grad = gp.mean(vk - v0,axis=0)
        bh_grad = gp.mean(hk - h0,axis=0)

        return W_grad,Wv_grad,Wh_grad,bv_grad,bh_grad

    def compute_dynamic_bias(self,v_input):
        v_data,vv_past,vh_past = self.extract_data(v_input)
        v_bias = gp.tile(self.bv,[v_data.shape[0],1]).copy()
        v_bias[:,:self.Nv] += gp.dot(vv_past,self.Wv)
        h_bias = self.bh + gp.dot(vh_past,self.Wh)

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

        b_hid = self.bh + gp.dot(vh_past,self.Wh)
        b_vis, b_lab = self.separate_vis_lab(self.bv)

        v_vis, v_lab = self.separate_vis_lab(v_data)
        W_vis,W_lab = self.separate_vis_lab(self.W,axis=0)


        # the b_vis term cancels out in the softmax
        #F = -np.sum(v_vis*b_vis,axis=1) 
        #F = F.reshape((-1,1)) - b_lab
        F =  - b_lab

        z = b_hid + gp.dot(v_vis,W_vis)
        z = z.reshape(z.shape + (1,))
        z = z + W_lab.T.reshape((1,) + W_lab.T.shape)

        hidden_terms = -gp.sum(log_1_plus_exp(z), axis=1)
        F = F + hidden_terms

        pr = softmax(-F)

        # compute hidden probs for each label configuration
        # this is used in the discriminative updates
        if output_h:
            h = sigmoid(z)
            return pr, h
        else:
            return pr

    def recon_error(self, v_input,K=1,print_output=False):
        '''compute K-step reconstruction error'''

        v0_data,vv_past,vh_past = self.extract_data(v_input)
        vk_mean = self.gibbs_samples(K,v_input,noisy=0)

        v0_vis,v0_lab = self.separate_vis_lab(v0_data)
        vk_vis,vk_lab = self.separate_vis_lab(vk_mean)
        vis_error = gp.mean(gp.abs(v0_vis - vk_vis))
        lab_error = gp.mean(gp.abs(v0_lab - vk_lab))

        lab_probs = self.label_probabilities(v_input)
        #pred_labs = gargmax(lab_probs)
        pred_labs = lab_probs.argmax(axis=1)
        ind, true_labs = gp.where(v0_lab == 1)
        percent_correct = gp.mean(pred_labs == true_labs)
        cross_entropy = -gp.mean(gp.log(lab_probs[ind,true_labs]))
        #prob_error = gp.mean(gp.abs(1. - lab_probs[ind,true_labs]))

        if print_output:
            output = '%30s %6.5f' % ('vis error:', vis_error/self.vis_scale) + '\n'
            output += '%30s %6.5f' % ('lab error:', lab_error) + '\n'
            #output += '%30s %6.5f' % ('prob error:', prob_error) + '\n'
            output += '%30s %6.5f' % ('cross entropy:', cross_entropy) + '\n'
            output += '%30s %6.5f' % ('class correct:', percent_correct)
            print output
            return output
        else: 
            return percent_correct, cross_entropy, lab_error, vis_error/self.vis_scale

    def generate(self,seed,num_steps,K,start_beat=1,noisy=False):
        '''
        generate a sequence of length num_steps given the seed sequence

        input:
        seed - Nv dimensional sequence of length >= max(Tv,Th)
                flattened using row-major ordering 
                (units in same time step nearest each other)
        num_steps - number of sequence steps to generate
        K - number of gibbs iterations per sample
        start_beat - beat number to start on
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


        Nl = self.Nl
        beat_labels = (np.arange(num_steps) + start_beat - 1) % Nl

        sequence = np.concatenate( (seed, np.zeros(num_steps * Nv))).astype('float32')
        sequence = gp.garray(sequence)

        idx = len(seed) - frame_size

        beat_idx = 0
        while idx+frame_size+Nv <= len(sequence):

            print idx+frame_size+Nv, 'of', len(sequence)

            v_input = sequence[idx:idx+frame_size+Nv]
            l_curr = gp.zeros(Nl)
            l_curr[beat_labels[beat_idx]] = 1


            #v_input[-Nv:] = gp.rand(Nv) 
            v_input[-Nv:] = v_input[-period_size-Nv:-period_size]
            v_input = gp.concatenate([v_input,l_curr])

            v_curr = self.gibbs_samples_labels_clamped(K,v_input[None,:],noisy)
            sequence[idx+frame_size:idx+frame_size+Nv] = v_curr[0,:-Nl]
            
            idx += hop_size
            beat_idx += 1

        return sequence

    def gibbs_samples_labels_clamped(self,K,v_input,noisy=0):
        '''
        compute a visible unit sample using Gibbs sampling with label units clamped

        input:
        K - number of complete Gibbs iterations
        v_input - [v_past, v_curr_seed, l_curr] array flattened using row-major ordering
                    * v_past of length Nv*max(Tv,Th) 
                    * v_curr_seed of length Nv
                    * l_curr of length Nl
        noisy - 0 = always use visible means and use hidden means to drive final sample
                1 = drive final sample with final hidden sample
                2 = use visible means for updates but use visible and hidden samples for final update
                3 = always use samples for both visible and hidden updates
                note: hidden samples are always used to drive visible reconstructions unless noted otherwise
        '''

        Nv = self.Nv
        Nl = self.Nl

        v0_data,vv_past,vh_past = self.extract_data(v_input)
        l0_data = v0_data[:,-Nl:] # original labels

        v_bias,h_bias = self.compute_dynamic_bias(v_input)

        h0_samp,h0_mean = self.sample_h_given_v(v0_data,h_bias)



        hk_samp = h0_samp
        hk_mean = h0_mean
        if noisy < 3:
            for k in xrange(K-1): # hk_samp <--> vk_mean
                vk_mean = self.mean_field_v_given_h(hk_samp,v_bias)
                vk_mean[:,-Nl:] = l0_data
                hk_samp, hk_mean = self.sample_h_given_v(vk_mean,h_bias)
        else:
            for k in xrange(K-1): # hk_samp <--> vk_samp
                vk_samp, vk_mean = self.sample_v_given_h(hk_samp,v_bias)
                vk_samp[:,-Nl:] = l0_data
                hk_samp, hk_mean = self.sample_h_given_v(vk_samp,h_bias)

        if noisy == 0:  # hk_mean --> v_mean
            v_mean = self.mean_field_v_given_h(hk_mean,v_bias)
            #pdb.set_trace()
            return v_mean
        elif noisy == 1: # hk_samp --> v_mean
            v_mean = self.mean_field_v_given_h(hk_samp,v_bias)
            return v_mean
        elif noisy > 1:  # hk_samp --> v_samp
            v_samp, v_mean = self.sample_v_given_h(hk_samp,v_bias)
            return v_samp

def gargmax(x):
    ''' 
    compute argmax on gpu (across rows)
    '''
    maxes = gp.max(x,axis=1)
    locs = x >= maxes.reshape((-1,1))
    num_maxes = gp.sum(locs,axis=1)
    if gp.any(num_maxes > 1):
        N = x.shape[0]
        args = np.zeros(N,dtype='int64')
        inds = gp.where(locs)
        args[inds[0]] = inds[1]
    else:
        args = gp.where(locs)[1]

    return args

def sigmoid(x):
    '''
    compute logistic sigmoid function avoiding overflow
    '''
    return gp.logistic(x)

def softmax(x):
    '''
    compute softmax function for each row
    while avoiding over/underflow
    '''
    m = gp.max(x,axis=1).reshape((-1,1)) # max for each row
    y = gp.exp(x - m)
    y /= gp.sum(y,axis=1).reshape((-1,1))

    return y

def sample_categorical(probs,cumsum):
    '''
    sample from categorical distribution (1-sample multinomial distribution)

    input:
    probs - probabilities in each row add to one [N x K]
    cumsum - square upper triangular matrix of ones of size K

    output:
    samples - [N x K] binary array with a single 1 per row
    '''
    if probs.ndim == 1:
        probs = probs.reshape((1,-1))

    N = probs.shape[0]
    #cdf = np.cumsum(probs, axis=1)[:,:-1]
    cdf = gp.dot(probs,cumsum)[:,:-1]
    #uni = np.random.uniform(size=(N,1))
    uni = gp.rand((N,1))
    category = gp.sum(uni >= cdf,axis=1)
    samples = gp.zeros(probs.shape)
    samples[np.arange(N),category] = 1

    return samples

def log_1_plus_exp(x):
    '''
    compute y = np.log(1+np.exp(x)) avoiding overflow
    '''
    return gp.log_1_plus_exp(x)

def train_rbm(rbm,training_data,params={}):
    '''
    train an rbm using contrastive divergence

    input:
    rbm - an initialized rbm class
    training_data - [N x Nv] matrix of N observations of Nv dimensions
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

        if epoch % reshuffle == 0:
            print '\n-----SHUFFLE TRAINING DATA-----\n'
            perm = rng.permutation(training_data.shape[0])
            training_data = training_data[perm]

        
        print '\nepoch:', epoch+1
        print 'learning rate:', learning_rate[epoch]
        print 'learning momentum:', learning_momentum[epoch]
        print 'contrastive divergence:', K[epoch]



        for batch in xrange(0,num_batches):
            rbm.cdk(K[epoch],training_data[batches[batch]],
                    learning_rate[epoch],learning_momentum[epoch],
                    weight_decay,noisy)
            if batch == 0:
                stats.add_stats(rbm.update_stats())


        max_update = stats.print_stats()

        print '\ntraining data'
        rbm.recon_error(training_data,K[epoch],print_output=True)


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
    train_error = rbm.recon_error(training_data,K[epoch],print_output=True)

    return stats

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

def set_initial_biases(params,training_data):
    '''
    set visible unit biases of CRBM to the appropriate value given training data statistics
    '''

    # initial vis unit bias
    Nv = params['Nv']
    Nl = params['Nl']
    pv = training_data[:,-(Nv+Nl):-Nl].mean(axis=0)
    pl = training_data[:,-Nl:].mean(axis=0)
    if params.get('vis_unit') == 'linear':
        bv = pv
    else:
        bv = gp.log(pv/(1-pv) + eps)
    eps = float(np.finfo('float32').eps)
    bl = gp.log(pl+eps) - gp.log(1-pl+eps)
    params['bv'] = gp.concatenate( (bv, bl) )



