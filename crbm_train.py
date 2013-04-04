
import numpy as np
import gnumpy as gp
import time

import data_helper
reload(data_helper)
import crbm
reload(crbm)




# training parameters
num_labels = 16 # number of beat labels, 16 per measure here (16th notes)
num_drums = 3
vis_scale = 8

batch_size = 100
cd_epochs = 300
gibbs_iters = [ [1,1,15], [0,0.2,1] ]
#gibbs_iters = [ [1,1,1], [0,0.5,1] ]

reshuffle = 100
momentum = [ [0.01,0.9,0.9], [0,0.05,1] ]
cd_weight_decay = 0.01

update_target = 0.001

decay_target = 0.1
decay_period = 0.1

# learning parameters
# see crbm.train_rbm() for more on these parameters
learning_params = {
        'num_epochs':cd_epochs,
        'batch_size':batch_size,
        'learning_momentum': momentum,
        'gibbs_iters': gibbs_iters,
        'weight_decay': cd_weight_decay,
        'update_target': update_target,
        'reshuffle': reshuffle,
        'train_type': 'cd',
        'noisy': 1  # see crbm.CRBM.cdk() for details on 'noisy'
        }


# model parameters
# see crbm.CRBM.__init__() for more on these parameters
params = {
        'model':crbm.LCRBM,
        'Nh':32,
        'Nv':num_drums,
        'Nl':num_labels,
        'Tv':4,
        'Th':32,
        'vis_unit':'linear',
        'vis_scale':vis_scale
        }
        

# load and setup training data
input_training_data = data_helper.setup_training_data(params,midi_dir='./data/')

crbm.set_initial_biases(params,input_training_data)

# instantiate CRBM model
model = params['model'](params)


# loading existing model
#param_id = 29081
param_id = None
if param_id is not None:
    model.load_params('saved_params/CRBM_%u.pkl' % param_id)
    print 'loaded %s: %u\n' % (str(type(model)),param_id)


gp.free_reuse_cache()

stats = crbm.train_rbm(model,
        input_training_data,learning_params)



# save params
param_id = None
if param_id is None:
    param_id = int(time.time() - 1364943068)

model.save_params('saved_params/CRBM_%u.pkl' % param_id)


gp.free_reuse_cache()
