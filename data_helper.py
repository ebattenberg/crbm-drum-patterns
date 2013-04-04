import re
import os

import numpy as np
import gnumpy as gp

import midi_tools

def load_data(midi_dir,verbose=False):
    '''
    load drum pattern midi data 
    '''
    tatums_per_beat = 4 # 16th notes
    time_sig = 4
    num_labels = tatums_per_beat * time_sig
    num_drums = 3
    subseq_length = 12 * tatums_per_beat * time_sig
    period  = time_sig * tatums_per_beat

    pattern = re.compile('(.+).mid\Z')
    midi_files = [pattern.match(f).group(1) for f in os.listdir(midi_dir) if pattern.match(f)]


    if verbose:
        print '\nloading midi training data'
    drum_patterns = {}
    sequential_data = []
    sequential_labels = []
    total_subseqs = 0
    for midi_file in midi_files:

        filename = midi_dir + midi_file + '.mid'
        if verbose:
            print filename 

        drum_matrix, tempo = midi_tools.midi_to_drum_matrix(filename,tatums_per_beat=tatums_per_beat) 
        beat_num = midi_tools.label_drum_matrix(drum_matrix.shape[1],period=period)
        drum_matrix = drum_matrix[:num_drums].T
        drum_patterns[filename] = {'pattern':drum_matrix, 'beat':beat_num, 'time_sig':time_sig}
        num_subseqs = drum_matrix.shape[0]/subseq_length
        new_total_subseqs = total_subseqs + num_subseqs
        if verbose:
            print total_subseqs, '--', new_total_subseqs
        total_subseqs = new_total_subseqs
        sequential_data += [drum_matrix[:num_subseqs*subseq_length].reshape((num_subseqs,subseq_length,num_drums))]
        sequential_labels += [beat_num[:num_subseqs*subseq_length].reshape((num_subseqs,subseq_length))]

    sequential_data = np.concatenate(sequential_data)
    sequential_labels = np.concatenate(sequential_labels)

    return sequential_data, sequential_labels, num_labels

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

def setup_training_data(params,midi_dir,verbose=False):
    '''
    load and setup training data

    input:
    T - max-lag for computing frame size
    '''

    # load training data
    sequential_data, sequential_labels, num_labels = load_data(midi_dir)

    T = max(params['Tv'],params['Th']) # max look-behind
    # convert sequences into subsequences of length T+1
    subseq_data, subseq_labels = frame_subseqs(T+1,sequential_data,sequential_labels)
    subseq_data *= params['vis_scale'] # put training data at correct scale
    training_data = subseq_to_frames(subseq_data)

    Nl = params['Nl']
    training_labels = compute_binary_labels(subseq_to_frames(subseq_labels),Nl)
    input_training_data = gp.concatenate((gp.garray(training_data),
                                            gp.garray(training_labels)),axis=1)

    return input_training_data

def get_seed_pattern(filename,model,offset=0):
    num_drums = model.Nv
    period = 16
    tatums_per_beat = 4

    T = max(model.Tv,model.Th)


    # get seed data
    seed_matrix, tempo = midi_tools.midi_to_drum_matrix(filename,tatums_per_beat=tatums_per_beat) 
    #seed_beats = midi_tools.label_drum_matrix(seed_matrix.shape[1],period=period)
    seed_matrix = seed_matrix[:num_drums].T * model.vis_scale

    seed_range = slice(offset,offset+T)
    seed = seed_matrix[seed_range].flatten()
    #beats = seed_beats[seed_range]

    return seed
