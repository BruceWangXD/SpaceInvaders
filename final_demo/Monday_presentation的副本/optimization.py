import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wave, struct
from copy import deepcopy
import serial
#!pip3 install catch22
# !pip3 install weighted_levenshtein
#from weighted_levenshtein import lev

from scipy.io import wavfile


from load_data import load_training_data

from plot_data import plot_labelled_wave, plot_predictions, plot_detection_errors

from classifier import (streaming_classifier,
                                   three_pronged_smoothing_classifier,
                                   two_pronged_smoothing_classifier,
                                   one_pronged_smoothing_classifier,
                                   zeroes_classifier)#,
                                   #catch22_knn_classifier)


# Read example datasss
#%matplotlib notebook
baudrate = 230400
# cport = '/dev/cu.usbmodem142301'  # set the correct port before you run it
cport = "/dev/cu.usbserial-DJ00E33Q"
#cport = '/dev/tty.usbmodem141101'  # set the correct port before run it
ser = serial.Serial(port=cport, baudrate=baudrate)    
# take example data
inputBufferSize = 1000   # 20000 = 1 second
ser.timeout = inputBufferSize/20000.0  # set read timeout 20000
#ser.set_buffer_size(rx_size = inputBufferSize)

samprate=10000
hyp_event_smart_threshold_window=5
hyp_event_smart_threshold_factor=0.31

#ts_zero_crossings
def ts_zero_crossings(x):
    return np.sum(x[0:-1]*x[1::] <= 0)

det_window=0.35
w = 0.6485 - det_window

hyp_detection_buffer_end = max(w/2, 1/samprate)
hyp_detection_buffer_start = w/2
window_size = det_window + hyp_detection_buffer_end + hyp_detection_buffer_start

buffer_size = inputBufferSize/10000
N_loops_over_window = int(np.ceil(window_size/buffer_size))



streaming_classifier(
                    ser,
                    samprate,
                    classifier = one_pronged_smoothing_classifier,
                    window_size = window_size, # time plotted in window [s]
                    N_loops_over_window = N_loops_over_window, # implicitly defines buffer to be 1/x of the window
                    total_time = None,  # max time
                    hyp_detection_buffer_end = hyp_detection_buffer_end, # seconds - how much time to shave off either end of the window in order to define the middle portion
                    hyp_detection_buffer_start = hyp_detection_buffer_start,
                    hyp_event_smart_threshold_window = hyp_event_smart_threshold_window, 
                    hyp_event_smart_threshold_factor = hyp_event_smart_threshold_factor, 
                    hyp_calibration_statistic_function = lambda x: ts_zero_crossings(x)/(len(x)/samprate)*det_window, # Function that calculates the calibration statistic
                    hyp_test_statistic_function = ts_zero_crossings, # Function that calculates the test statistic
                    hyp_event_history = 11,
        #             hyp_timeout = 20,
                    hyp_consecutive_triggers = 2,
                    hyp_consecutive_reset = 10,
    #                 zeroes_consec_threshold = 0.1, 
    #                 using_zeroes_classifier = key == "Zeros classifier",
                    plot = False,
                    store_events = False, 
                    verbose=False,
                    live = True,
        #             dumb_threshold = True,
                    flip_threshold = True
        #             timeout = True
            )