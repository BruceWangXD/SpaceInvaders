import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wave, struct
from scipy.io import wavfile
import os


def read_arduino(ser,inputBufferSize):
#     data = ser.readline((inputBufferSize+1)*2)
    data = ser.read((inputBufferSize+1)*2)
    out =[(int(data[i])) for i in range(0,len(data))]
    return out

def process_data(data, cal_mean):
    data_in = np.array(data)
    result = []
    i = 1
    while i < len(data_in)-1:
        if data_in[i] > 127:
            # Found beginning of frame
            # Extract one sample from 2 bytes
            intout = (np.bitwise_and(data_in[i],127))*128
            i = i + 1
            intout = intout + data_in[i]
            result = np.append(result,intout)
        i=i+1
    return np.flip(np.array(result)-cal_mean)


# ser.read works by waiting for <inputBufferSize> bytes from the port

def read_arduinbro(wav_array, inputBufferSize, k):
#    data = ser.readline(inputBufferSize)
    if inputBufferSize*(k+1) < len(wav_array):
        data = wav_array[(inputBufferSize*(k)):(inputBufferSize*(k+1))]
    else:
        data = wav_array[(inputBufferSize*(k))::]
    return np.flip(data)




def load_training_data(path = "/Users/billydodds/Documents/Uni/DATA3888/Aqua10/Datasets/Good Data - Sandeep no errors/",
                       scale_factor= 512/(2**13 - 1),
                       blacklist = ["blink", "different", "fast", "slow", "eyebrow"],
                       whitelist = ["right", "left"]):


    files = os.listdir(path)

    waves = {}
    labels = {}

    for file in files:
        filters = [x not in file.lower() for x in blacklist]
        filters.extend([x in file.lower() for x in whitelist])
        if np.all(np.array(filters)):
            if file[-4::] == ".wav":
                samprate, wav_array = wavfile.read(path+file)

                wav_array = wav_array*scale_factor
                
                # Centre at y=0:
                wav_array -= 512


                waves[file[:-4]] = wav_array
            elif file[-4::] == ".txt":
                labels_dat = pd.read_csv(path+file, sep=",\t", skiprows=1)
                labels_dat.columns = ["label", "time"]
                # Change depending on whether L is coded as 1 or as 2
                labels_dat.label = ["L" if label == 1 else "R" for label in labels_dat.label]

                labels[file[:-4].replace(".", "")] = labels_dat


    print(waves.keys(), labels.keys())

    assert set(waves.keys()).difference(set(labels.keys())) == set()
    
    return waves, labels, samprate
    