
import midi
import numpy as np
from scipy.interpolate import interp1d
import pylab as p
import pdb

def midi_to_drum_matrix(midifile,tatums_per_beat=4):
    '''
    convert drum midi events into a tatum synchronous matrix

    input:
    midifile - filename of input midi file
    tatums_per_beat - number of tatums per beat 
                    (e.g. for 16th note subdivisions set to 4)

    output:
    drum even matrix
    '''
    parsed_midi = midi.read_midifile(midifile) #gets a list of events
    parsed_midi.make_ticks_abs() #turns the ticks into absolute ms times
    events = parsed_midi[0]
    resolution = parsed_midi.resolution
    #print 'resolution:', resolution
    tempo = 0
    for e in events:
        if e.name == 'Set Tempo':
            tempo = e.get_bpm()
            #print 'detected tempo:',tempo
            break
    if tempo is 0:
        print 'did not detect tempo'

    tatum = resolution/tatums_per_beat

    # include only onsets and foot control messages
    onsets = [e for e in events if e.name == 'Note On' and e.get_velocity() > 0]
    control = [e for e in events if e.name == 'Control Change']

    oTick = np.array([o.tick for o in onsets])
    oTatum = np.round(oTick/float(tatum)).astype('int')
    oTickJitter = np.abs(oTick - oTatum*tatum)
    if np.any(oTickJitter > 1):
        print 'WARNING: tick jitter of more than one in file %s' % midifile
        raise ValueError, 'tick jitter of more than one tick'
    oNoteNum = np.array([o.get_pitch() for o in onsets])
    oVel = np.array([o.get_velocity() for o in onsets])/127.

    footcontrol = 4
    cTick = np.array([c.tick for c in control if c.get_control() == footcontrol])
    cTatum = cTick/float(tatum)
    cVal = np.array([c.get_value() for c in control if c.get_control() == footcontrol])

    onsetArray = np.vstack((oTatum,oNoteNum,oVel))
    controlArray = np.vstack((cTatum,cVal))

    # convert arrays to matrices
    notes = np.unique(oNoteNum)
    notesList = notes.tolist()

    numFrames = max(oTatum.max(),cTatum.max())+1
    onsetMatrix = np.zeros((notes.size,numFrames))
    for n in range(len(notes)):
        matchedFrames = oNoteNum==notes[n]
        onsetMatrix[n,oTatum[matchedFrames]] = oVel[matchedFrames]

    drums = ['bass','snare','cHihat','oHihat','ride','fHihat']
    drumNotes = [ [36], [38,40], [42,22], [46, 26], [51,53], [44]]
    drumNotesList = [note for sublist in drumNotes for note in sublist]
    numDrums = len(drums)
    drumMatrix = np.zeros((numDrums,numFrames))

    # check for midi notes that cannot be mapped to drums
    unrecognized_notes = [n for n in notesList if n not in drumNotesList]
    if len(unrecognized_notes) > 0:
        raise ValueError, 'unrecognized drum notes: %s' % str(unrecognized_notes)


    # map midi note onsets to rows of drum matrix
    for d in range(len(drums)):
        for n in drumNotes[d]:
            if n in notesList:
                drumMatrix[d] += onsetMatrix[notesList.index(n)]
    # adjust open/closed hihat with footcontrol
    fc = interp1d(cTatum,cVal,bounds_error=False,fill_value=0)
    fc_values = fc(np.arange(numFrames))
    hihatFilter = fc_values > 90 # closed hihat threshold
    drumMatrix[drums.index('cHihat'),hihatFilter] += drumMatrix[drums.index('oHihat'),hihatFilter]
    drumMatrix[drums.index('oHihat'),hihatFilter] = 0

    # make sure amplitude never exceeds 1.0 
    # (if two notes on same drum occur simultaneously)
    drumMatrix[drumMatrix > 1.0] = 1.0

    numOnsets = (drumMatrix > 0).sum()
    #print 'extracted %u notes over %u tatums' % (numOnsets,numFrames)


    return drumMatrix, tempo

def label_drum_matrix(numTatums,period,offset=0):
    '''
    compute a vector of ground-truth beat number data to for 
    drumMatrix

    input:
    drumMatrix - matrix of with drum-wise activations in rows
    period - measure length in tatums
    offset - number of tatums before the first beat of the first measure

    output:
    beat_num - vector containing beat numbers
    '''


    beat_num = np.zeros(numTatums,dtype='uint16')
    measure = np.arange(period) + 1
    complete_measures = (numTatums-offset)/period
    partial_measure_beats = numTatums - offset - complete_measures*period
    beat_num = np.hstack( (np.zeros(offset), np.tile(measure,complete_measures), measure[:partial_measure_beats] ) ).astype('uint16')

    return beat_num

def drum_matrix_to_midi(drum_matrix,tatums_per_beat,output_filename,midi_ref_file,notes=None):
    '''
    write a drum pattern matrix to a midi file

    input:
    drum_matrix - drum onsets to be converted to midi 
                [one drum per row, one tatum per column]
    tatums_per_beat - number of subdivisions per beat
    output_filename - file to write midi output to
    midi_ref_file - midi file to steal header info (tempo, etc.) from
    notes - (optional) the midi notes to write each drum column to
    '''

    if notes is None:
        notes = [36, 38, 42, 46, 51]

    parsed_midi = midi.read_midifile(midi_ref_file) #gets a list of events
    parsed_midi.make_ticks_abs() #turns the ticks into absolute ms times
    events = parsed_midi[0]
    resolution = parsed_midi.resolution
    print 'resolution:', resolution
    tempo = 0
    for e in events:
        if e.name == 'Set Tempo':
            tempo = e.get_bpm()
            print 'detected tempo:',tempo
            break
    if tempo is 0:
        print 'did not detect tempo'

    tatum = resolution/tatums_per_beat

    # ignore small amplitude onsets
    thresh = 0.05 * drum_matrix.max()
    onset_locs = np.where(drum_matrix > thresh)

    # convert tatums to ticks
    num_tatums = drum_matrix.shape[1]
    ticks = np.arange(num_tatums) * tatum

    
    # create note on/off events
    events = []
    for i in xrange(len(onset_locs[0])):

        note = notes[onset_locs[0][i]]
        tick = ticks[onset_locs[1][i]]
        vel = int(max(0,min(127, 127 * drum_matrix[onset_locs[0][i],onset_locs[1][i]])))

        on = midi.NoteOnEvent()
        on.channel = 9
        on.set_pitch(note)
        on.set_velocity(vel)
        on.tick = tick
        events += [on]

        off = midi.NoteOffEvent()
        off.channel = 9
        off.set_pitch(note)
        off.set_velocity(64)
        off.tick = tick + 60
        events += [off]
    # create hihat footcontrol events
    # (for now just set all to max [closed hihat])

    for t in ticks:

        val = 127 # all the way down

        control = midi.ControlChangeEvent()
        control.channel = 9
        control.set_control(4)
        control.set_value(val)
        control.tick = t
        events += [control]


    # sort events in tick order
    inds = np.argsort([e.tick for e in events])
    # reuse midi headers
    new_track = midi.Track(parsed_midi[0])

    for e in parsed_midi[0]:
        if e.name == 'Note Off' or e.name == 'Note On' or e.name == 'Control Change':
            new_track.remove(e)
        if e.name == 'End of Track':
            end_of_track = e
            new_track.remove(e)

    # set end of track to time of last event + one beat
    end_of_track.tick = ticks[-1] + resolution

    for i in inds:
        new_track += [events[i]]
    new_track += [end_of_track]
    parsed_midi[0] = new_track
    parsed_midi[0].make_ticks_rel()

    midi.write_midifile(output_filename,parsed_midi)

    return parsed_midi

def align_matrices(groundTruth, testData, max_shift=30):
    '''
    this assumes that testData is a delayed version of groundTruth,
    returns optimal alignment (shift value)
    '''

    rows = min(groundTruth.shape[0], testData.shape[0])


    corr = np.zeros((max_shift+1,))

    for shift in range(max_shift+1):
        td = testData[:,shift:]
        columns = min(td.shape[1],groundTruth.shape[1])
        corr[shift] = np.sum(td[:rows,:columns] * groundTruth[:rows,:columns])

    return np.argmax(corr)

def plot_vstack(rows, clear=None):
    if clear == None:
        p.clf()
    index = 0
    for row in rows:
        p.plot(row + index,'o-')
        index += 1
    p.show()

def plot_subs(rows):
    colors = ['b','g','r','c','m','y','k']
    x = np.arange(rows.shape[1])
    fig = p.figure()
    num_rows = rows.shape[0]
    index = 0
    for row in rows:
        if index == 0:
            ax0 = fig.add_subplot(num_rows,1,num_rows)
            ax = ax0
        else:
            ax = fig.add_subplot(num_rows,1,num_rows-index,sharex=ax0)
        ax.stem(x[row>0],row[row>0],'%s-'%colors[index],'%so'%colors[index])
        index += 1
    p.show()

def plot_vstack_beat(onsets,beats,tatums_per_quarter=4,clear=None):
    '''
    stack drum activation amplitudes per drum and plot quarter note numbers
    '''

    # determine quarter note labels
    period = beats.max()

    quarters = np.zeros_like(beats).astype('float32')
    quarter_locs = (beats-1) % tatums_per_quarter == 0
    quarters[quarter_locs] = (beats-1)[quarter_locs]/tatums_per_quarter + 1
    quarters /= quarters.max()

    
    if clear == None:
        p.clf()
    index = 0
    for row in onsets:
        p.plot(row + index,'o-')
        index += 1
    p.plot(quarters + index,'o-')
    ticklocs = np.where(quarters > 0)[0]
    p.xticks(ticklocs)
    p.yticks(np.arange(index))
    p.grid(b=True)
    p.show()


    return quarters


