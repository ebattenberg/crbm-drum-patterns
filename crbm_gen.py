
import numpy as np
import gnumpy as gp
import pylab as pl

import data_helper
reload(data_helper)
import crbm
reload(crbm)
import midi_tools


# load and setup training data
#input_training_data = data_helper.setup_training_data(params,midi_dir='./data/')

num_drums = 3
num_labels = 16
params = {
        'model':crbm.LCRBM,
        'Nh':8,
        'Nv':num_drums,
        'Nl':num_labels,
        'Tv':4,
        'Th':16,
        'vis_unit':'linear',
        'vis_scale':10
        }
# instantiate CRBM model
model = crbm.LCRBM(params)


# loading existing model
#param_id = 174209
param_id = None
if param_id is not None:
    model.load_params('saved_params/CRBM_%u.pkl' % param_id)
    print 'loaded %s: %u\n' % (str(type(model)),param_id)

# generation params
#seed = 
num_steps = 12*16
K = 3
noisy = 2

#filename = './data/sixteenth_note_bass_drum_patterns.mid'
filename = './data/blast_beat.mid'
start_beat = 1
offset = 8*16
T = max(model.Tv,model.Th)
seed_data = data_helper.get_seed_pattern(filename,model,offset)


sequence = model.generate(seed_data,num_steps,K,start_beat,noisy).reshape((-1,model.Nv)).T/model.vis_scale
##sequence = drum_matrix
beats = midi_tools.label_drum_matrix(sequence.shape[1],period=16,offset=0)
#
#quarters = plot_vstack_beat(sequence,beats)
sequence = sequence.as_numpy_array()

pl.clf()
pl.imshow(sequence,origin='lower')
pl.show()

ref_midi_file = filename
output_dir = './output/'
tatums_per_beat = 4
output_filename = 'generated_%u.midi' % np.random.randint(100000)
midioutput = midi_tools.drum_matrix_to_midi(sequence,tatums_per_beat,
        output_dir+output_filename,ref_midi_file)
print output_filename

gp.free_reuse_cache()


