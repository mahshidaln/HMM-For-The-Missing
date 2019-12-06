import os
import json
import numpy as np
from numpy.random import randn
import pandas as pd
import csv
from BaumWelch import baum_welch
 
def hmm (piece_length, initial, emission, transition, transition2, notes):
    hidden_no = transition.shape[0]
    emission_no = emission.shape[1]
    hidden_states = np.arange(0, hidden_no, dtype = int)
    output_states = np.arange(0, emission_no, dtype = int)
    hidden_array = np.zeros(piece_length, dtype = int)
    output_array = np.zeros(piece_length, dtype = int)
    hidden_array[0] = np.random.choice(hidden_states, size = 1, p = initial)
    
    for i in range(1, piece_length):
        hidden_array[i] = np.random.choice(hidden_states, size = 1, p = transition[hidden_array[i-1], :])
    for i in range(0, piece_length):
        output_array[i] = np.random.choice(output_states, size = 1, p = emission[hidden_array[i], :])
     
    result = np.zeros(len(output_array))
    for i in range(0, len(output_array)):
        result[i] = notes[output_array[i]]
 
    return (result, hidden_array)


def pre_process(data_path, input_filename):

    input_file = data_path + input_filename
    with open(input_file, encoding = "ISO-8859-1") as fd:
        reader = csv.reader(fd)
        rows = [row for idx, row in enumerate(reader)]
    song = pd.DataFrame(rows)
    row, column = np.where(song == ' Header')
    quarter_note = song.iloc[row,5].values.astype(int)[0]
    row, column = np.where(song == ' Time_signature')
    numerator = song.iloc[row, 3].values.astype(int)[0]
    denominator = song.iloc[row, 4].values.astype(int)[0]**2
    try:
        row, column = np.where(song == ' Key_signature')
        key = song.iloc[row,3].values.astype(int)[0]
    except:
        key = None
    
    song_order = song.loc[song.iloc[:,0] == np.max(song.iloc[:,0])]
    song_order = song_order[song_order.iloc[:, 2].isin([' Note_on_c', ' Note_off_c'])]
    time = np.array(song_order.iloc[:,1]).astype(int)
    notes = np.array(song_order.iloc[:,4]).astype(int)
    velocity = np.array(song_order.iloc[:,5]).astype(int)
    measures = np.round(np.max(time)/quarter_note)/numerator
    quarter_note = quarter_note
    actual = np.arange(0, quarter_note*measures*numerator, quarter_note).astype(int) 
    
    nearest = [actual[(np.abs(actual-time[i])).argmin()] for i in range(len(time))]
    time = np.array(nearest).astype(int)
    return(quarter_note, numerator, denominator, key, measures, time, notes, velocity, song, song_order.index)


def generate(data_path, converter_path, input_filename, output_filename, quarter_note, hidden_no, threshold):
    quarter_note, numerator, denominator, key, measures, time, notes, velocity, song, song_index = pre_process(data_path, input_filename)

    used_notes = np.unique(notes)
    un = len(used_notes)
    notes_index = np.array([int(np.where(used_notes == notes[i])[0]) for i in range(0, len(notes))])
    notes_no = len(notes)
      
    iteratio1, p1, initial1, emission1, transition1 = baum_welch(notes_no, hidden_no, un, notes_index, threshold)
    new_notes, hidden_array  = hmm(notes_no, initial1, emission1, transition1, None, used_notes) 

    song.iloc[song_index, 1] = time
    song.iloc[song_index, 4] = new_notes
    song.iloc[song_index, 5] = velocity
    song.iloc[song_index[np.where(velocity !=0)], 2] = ' Note_on_c'
    song.iloc[song_index[np.where(velocity ==0)], 2] = ' Note_off_c'

    output_split = output_filename.split('.')
    output_csv_filename = output_split[0] + '-hn' + str(hidden_no)+ '-qn' + str(quarter_note) + '-th' + str(threshold) + '.' + output_split[1]
    output_midi_filename = output_split[0] + '-hn' + str(hidden_no)+ '-qn' + str(quarter_note) + '-th' + str(threshold) + '.' + 'mid'

    input_split = input_filename.split('.')
    output_dir = data_path + input_split[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_csv_file = output_dir + '/' + output_csv_filename
    output_midi_file = output_dir + '/' + output_midi_filename
    song.to_csv(output_csv_file, header = None, index = False)
    
    cmd = converter_path + "Csvmidi.exe" + " " + output_csv_file + " " + output_midi_file
    #print (cmd)
    os.system(cmd)
    
    return(time, notes, new_notes, hidden_array, initial1, emission1, transition1) 

with open("config.json") as json_file:
    config = json.load(json_file)
data_path = config["data_path"]
converter_path = config["converter_path"]

new_time, original_notes , new_notes, hidden, initial, emission, transition = generate(data_path, converter_path,'river.csv', 'river3.csv', 256,  7, 1E-6)
