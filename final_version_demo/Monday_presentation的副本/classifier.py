
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


import os
from message import create_msg
from load_data import read_arduino, process_data, read_arduinbro



def three_pronged_smoothing_classifier(arr, samprate, downsample_rate=10, window_size_seconds=0.3, max_loops=10):
    arr_ds = arr[0::downsample_rate]
    
    fs = samprate/downsample_rate
    dt = 1/fs
    t = np.arange(0, (len(arr_ds)*dt), dt)

    # Smooth wave
    window_length = int(window_size_seconds*samprate/downsample_rate + 1)
    filtered_arr = signal.savgol_filter(arr_ds, window_length, 1)

    # Indices of positive maxima
    max_locs = np.array(signal.argrelextrema(filtered_arr, np.greater)[0])
    max_vals = filtered_arr[max_locs]
    max_locs = max_locs[max_vals > 0]
    
    # Indices of negative minima
    min_locs = np.array(signal.argrelextrema(filtered_arr, np.less)[0])
    min_vals = filtered_arr[min_locs]
    min_locs = min_locs[min_vals < 0]
    
    # Appended indices
    max_min_locs = np.append(max_locs, min_locs)
    
    # Values of above indices
    max_min_values = filtered_arr[max_min_locs]
    
    # Absolute value of those values
    abs_max_min_values = np.abs(max_min_values)
    
    # A vector with a length equal to the number of minimums: all '-1' to say minimum
    numMin = [-1]*len(min_locs)
    # A vector with a length equal to the number of maximums: all '1' to say maximum
    numMax = [1]*len(max_locs)
    
    # Vector same size as max_min_values with first half being maximums and second half being minimums
    isMin = np.append(numMax, numMin)
    
    # Stack the three vectors
    val_and_idx = np.vstack([abs_max_min_values, max_min_locs, isMin])
    
    # Sort the magnitudes of the extrema in descending order (-1 indicates descending)
    val_and_idx_sorted = val_and_idx[ :, (-1*val_and_idx[0]).argsort()]

    classificationFound = False
    
    # We will continue looping until we have an appropriate classification. This relies on having the extrema INTERCHANGE between max and min (no two min right next to eachother)
    loops = 0
    while not classificationFound and loops < max_loops:
        
        # Take the top three magnitudes
        top_3 = val_and_idx_sorted[:, 0:3]
        
        # Sort according to the indices of those values
        top_3_sorted = top_3[ :, top_3[1].argsort()]
        
        # Break if we run out of turning points
        if top_3_sorted.shape != (3, 3):
            return "_"
        
        # If two min or two max occur one after the other, we know we have an inappropriate result so we delete one of those doubled min/max
        if top_3_sorted[2, 0]*top_3_sorted[2, 1] > 0:
            val_and_idx_sorted = np.delete(val_and_idx_sorted, 1, 1)
        elif top_3_sorted[2, 1]*top_3_sorted[2, 2] > 0:
            val_and_idx_sorted = np.delete(val_and_idx_sorted, 2, 1)
        else:
            classificationFound = True
        
        loops += 1
    
    if np.sum(top_3_sorted[2, :]) == -1:
        return 'L'
    elif np.sum(top_3_sorted[2, :]) == 1:
        return 'R'
    else:
        return "_"
    
    
    
    
def two_pronged_smoothing_classifier(arr, samprate, downsample_rate=10, window_size_seconds=0.3, max_loops=10):
    arr_ds = arr[0::downsample_rate]
    
    fs = samprate/downsample_rate
    dt = 1/fs
    t = np.arange(0, (len(arr_ds)*dt), dt)

    # Smooth wave
    window_length = int(window_size_seconds*samprate/downsample_rate + 1)
    filtered_arr = signal.savgol_filter(arr_ds, window_length, 1)

    # Indices of positive maxima
    max_locs = np.array(signal.argrelextrema(filtered_arr, np.greater)[0])
    max_vals = filtered_arr[max_locs]
    max_locs = max_locs[max_vals > 0]
    
    # Indices of negative minima
    min_locs = np.array(signal.argrelextrema(filtered_arr, np.less)[0])
    min_vals = filtered_arr[min_locs]
    min_locs = min_locs[min_vals < 0]
    
    # Appended indices
    max_min_locs = np.append(max_locs, min_locs)
    
    # Values of above indices
    max_min_values = filtered_arr[max_min_locs]
    
    # Absolute value of those values
    abs_max_min_values = np.abs(max_min_values)
    
    # A vector with a length equal to the number of minimums: all '-1' to say minimum
    numMin = [-1]*len(min_locs)
    # A vector with a length equal to the number of maximums: all '1' to say maximum
    numMax = [1]*len(max_locs)
    
    # Vector same size as max_min_values with first half being maximums and second half being minimums
    isMin = np.append(numMax, numMin)
    
    # Stack the three vectors
    val_and_idx = np.vstack([abs_max_min_values, max_min_locs, isMin])
    
    # Sort the magnitudes of the extrema in descending order (-1 indicates descending)
    val_and_idx_sorted = val_and_idx[ :, (-1*val_and_idx[0]).argsort()]

    classificationFound = False
    
    # We will continue looping until we have an appropriate classification. This relies on having the extrema INTERCHANGE between max and min (no two min right next to eachother)
    loops = 0
    while not classificationFound and loops < max_loops:
        
        # Take the top three magnitudes
        top_2 = val_and_idx_sorted[:, 0:2]
        
        # Sort according to the indices of those values
        top_2_sorted = top_2[ :, top_2[1].argsort()]
        
        # Break if we run out of turning points
        if top_2_sorted.shape != (3, 2):
            return "_"
        
        # If two min or two max occur one after the other, we know we have an inappropriate result so we delete one of those doubled min/max
        if top_2_sorted[2, 0]*top_2_sorted[2, 1] > 0:
            val_and_idx_sorted = np.delete(val_and_idx_sorted, 1, 1)
        else:
            classificationFound = True
        
        loops += 1
    
    if top_2_sorted[2, 0] == -1:
        return 'L'
    elif top_2_sorted[2, 0] == 1:
        return 'R'
    else:
        return "_"
    
    
    
    
    

#ZEROES CLASSIFIER:
#Looks for x samples (after downsampling) that are consecutively positive / negative.
#Classifies only using the first hump of the wave.
from numba import njit

@njit
def consecutive(data, stepsize=0):        #returns a list of sub-arrays, grouped by the same consecutive value (in this case they are groups of consecutive 1s or -1s)                       
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

@njit #numba 'decorator' that performs just-in-time (jit) compilation
def zeroes_classifier(arr, samprate, downsample_rate=10, ave_height = 10, consec_seconds = 0.2):
    arr_ds = arr[0::downsample_rate]
    arr_sign = np.sign(arr_ds)            #returns array of -1s and 1s, depending on if the number was negative or positive respectively
    i = 0
    split_arrays = consecutive(arr_sign)  #returns a list of sub-arrays, grouped by the same consecutive value (in this case they are groups of consecutive 1s or -1s)
    for sub_arr in split_arrays:
        if len(sub_arr) > consec_seconds * samprate / downsample_rate:          #RHS converts seconds to number of samples
           
            # #if there were 'consec_seconds' seconds of no zero-crossings, then check if the average height is bigger than 'ave_height'
            if np.mean(arr_ds[i:(i + len(sub_arr) - 1)]) > ave_height:          
                return 'R'
            elif np.mean(arr_ds[i:(i + len(sub_arr) - 1)]) < -1 * ave_height:
                return 'L'
        i += len(sub_arr)
    return '_'                            #unable to classify because there were not 'consec_seconds' seconds of no zero-crossings

def one_pronged_smoothing_classifier(arr, samprate, downsample_rate=10, window_size_seconds=0.3, max_loops=10):
    arr_ds = arr[0::downsample_rate]
    
    fs = samprate/downsample_rate
    dt = 1/fs
    t = np.arange(0, (len(arr_ds)*dt), dt)

    # Smooth wave
    window_length = int(window_size_seconds*samprate/downsample_rate + 1)
    filtered_arr = signal.savgol_filter(arr_ds, window_length, 1)

    # Indices of positive maxima
    max_locs = np.array(signal.argrelextrema(filtered_arr, np.greater)[0])
    max_vals = filtered_arr[max_locs]
    max_locs = max_locs[max_vals > 0]
    
    # Indices of negative minima
    min_locs = np.array(signal.argrelextrema(filtered_arr, np.less)[0])
    min_vals = filtered_arr[min_locs]
    min_locs = min_locs[min_vals < 0]
    
    # Appended indices
    max_min_locs = np.append(max_locs, min_locs)

    
    # Values of above indices
    max_min_values = filtered_arr[max_min_locs]
    
    # Absolute value of those values
    abs_max_min_values = np.abs(max_min_values)
    
    # A vector with a length equal to the number of minimums: all '-1' to say minimum
    numMin = [-1]*len(min_locs)
    # A vector with a length equal to the number of maximums: all '1' to say maximum
    numMax = [1]*len(max_locs)
    
    # Vector same size as max_min_values with first half being maximums and second half being minimums
    isMin = np.append(numMax, numMin)
    
    # Stack the three vectors
    val_and_idx = np.vstack([abs_max_min_values, max_min_locs, isMin])
    
    # Sort the magnitudes of the extrema in descending order (-1 indicates descending)
    val_and_idx_sorted = val_and_idx[ :, (-1*val_and_idx[0]).argsort()]
    
    """                 # idk why but this chunk broke something after adding code that clears the classifier_optimisation.csv file
    with open("print.txt", "w") as file:
        file.write(str(val_and_idx_sorted.shape))
    """

    if val_and_idx_sorted.shape == (3, 0):
        if val_and_idx_sorted[2] == -1:
            return 'L'
        elif val_and_idx_sorted[2] == 1:
            return 'R'
        else:
            return "_"
    else:
        if val_and_idx_sorted[2, 0] == -1:
            return 'L'
        elif val_and_idx_sorted[2, 0] == 1:
            return 'R'
        else:
            return "_"
        


# catch22 kNN classifier (using stepwise selected features)
#arr: the array (the event) to be classified (a numpy array)

#Prep:

# from catch22 import catch22_all
# import catch22
# from sklearn.neighbors import KNeighborsClassifier

# path = ""
# step_csv = "catch22_step_selected_features.csv"

# catch22_step_training_data = pd.read_csv(path+step_csv)

# X_train = catch22_step_training_data.iloc[:,0:-1]
# y_labels = catch22_step_training_data.iloc[:,-1]
    
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(X_train, y_labels)

# def catch22_knn_classifier(arr, samprate, downsample_rate=10):
#     arr_ds = arr[0::downsample_rate]
#     arr_list = arr_ds.tolist() # single catch22 feature won't take numpy arrays, only lists or tuples
    
#     feature_one = catch22.DN_HistogramMode_5(arr_list)
#     feature_two = catch22.SB_BinaryStats_mean_longstretch1(arr_list)
#     feature_three = catch22.FC_LocalSimple_mean1_tauresrat(arr_list)
#     feature_four = catch22.DN_OutlierInclude_p_001_mdrmd(arr_list)
#     feature_five = catch22.SP_Summaries_welch_rect_area_5_1(arr_list)
    
#     test_features = [[feature_one, feature_two, feature_three, feature_four, feature_five]]
    
#     return neigh.predict(test_features)[0] # returns a single item list, so use index 0 to return the prediction itself





def streaming_classifier(
    wav_array, # Either the array from file (or ser if live = True)
    samprate,
    classifier = three_pronged_smoothing_classifier, 
    window_size = 1.5, # Total detection window [s]
    N_loops_over_window = 15, # implicitly defines buffer to be 1/x of the window
    hyp_detection_buffer_end = 0.3, # seconds - how much time to shave off end of the window in order to define the middle portion
    hyp_detection_buffer_start = 0.7, # seconds - how much time to shave off start of the window in order to define the middle portion
    hyp_event_threshold = 20,
    hyp_event_smart_threshold_window = 5, # The length of the calibration period to define the threshold
    hyp_calibration_statistic_function = lambda x: np.max(x) - np.min(x), # Function that calculates the calibration statistic
    hyp_test_statistic_function = lambda x: np.max(x) - np.min(x), # Function that calculates the test statistic
    hyp_event_smart_threshold_factor = 0.5, # The scale factor of the calibration range that will become the threshold
    hyp_event_history = 5, # How many historical event detection results are kept in memory (whether the test criteria failed or passed)
    hyp_consecutive_triggers = 3, # How many threshold triggers need to occur in a row for an event to be called
    hyp_consecutive_reset = 1, # How many threshold failures need to occur in a row for the classifier to be primed for a new event
    hyp_timeout = 10,
    
    # Zeros Classifier Parameters
    using_zeroes_classifier = False,            
    plot_zeroes_classifier = False,                # Plot the waves that took an unusually long time for the zeroes classifier
    use_smart_hyp_zeroes_height_threshold = False,      # Whether or not you use the smart threshold as the height threshold for the zeroes classifier
    zeroes_height_threshold = 10,                  # Only used if using_zeroes_classifier is true
    zeroes_consec_threshold = 0.2,                 # Only used if using_zeroes_classifier is true
    
    dumb_threshold = False,
    total_time = None,  # max time. If none, it goes forever!
    plot = False, # Whether to plot the livestream data
    store_events = False, # Whether to return the classification window array for debugging purposes
    store_times = False,        # Store time taken for each classification
    nil_classifier = False,     # Does not classify, just gives L as its prediction always. Used for time complexity purposes.
    verbose=False, # lol
    live = False, # Whether we're live
    timeout = False,
    flip_threshold = False):
    
    
    ### Initialisation ###
    IPC_FIFO_NAME = "hello_ipc"
#     try:
#         os.mkfifo('/Users/brucewong/desktop/demo/Monday_presentation?????????/'+IPC_FIFO_NAME)
#     except:
#         pass
    fifo = os.open(IPC_FIFO_NAME, os.O_WRONLY)


    if total_time is None:
        try:
            total_time = len(wav_array)/samprate
        except:
            total_time = 1000000 # Just a large number
    if store_events:
        predictions_storage = []
    
    predictions = ""
    predictions_timestamps = []

    
    # Initialise variables
    inputBufferSize = int(window_size/N_loops_over_window * samprate)
    N_loops =(total_time*samprate)//inputBufferSize  # len(wav_array)//inputBufferSize 
    T_acquire = inputBufferSize/samprate    # length of time that data is acquired for 
    N_loops_over_window = window_size/T_acquire    # total number of loops to cover desire time window
    
    if timeout:
        timer = hyp_timeout
        


    # Initialise plot
    if plot:
        min_y = -200 #np.min(wav_array)
        max_y = 200 #np.max(wav_array)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        plt.ion()
        fig.show()
        fig.canvas.draw()


    # Hyperparameter conversions
    hyp_detection_buffer_start_ind = int(round(hyp_detection_buffer_start * samprate))
    hyp_detection_buffer_end_ind = int(round(hyp_detection_buffer_end * samprate))
    
    
    # Initialise Calibration
    calibrate = True
    cal_mean = 0
    N_loops_calibration = hyp_event_smart_threshold_window//(window_size/N_loops_over_window)
    

    # Initialise Event History
    event_history = np.array([False]*hyp_event_history)
    primed = True

    
    ### Start stream ###
    classification_times = []
    for k in range(0,int(N_loops)):
        
        if live:
            data = read_arduino(wav_array,inputBufferSize)
            data_temp = process_data(data, cal_mean)
        else:
            data_temp = read_arduinbro(wav_array, inputBufferSize, k)


        if k < N_loops_over_window:
            if k==0:
                data_cal = data_temp
                data_plot = data_temp
            else:
                data_plot = np.append(data_temp, data_plot)
                if calibrate:
                    data_cal = np.append(data_temp, data_cal)
            continue
        else:
            data_plot = np.roll(data_plot,len(data_temp))
            data_plot[0:len(data_temp)] = data_temp
            
            if calibrate:
                data_cal = np.append(data_temp,data_cal)

                if (k > N_loops_calibration):
                    cal_mean = np.mean(data_cal)
                    st_range = hyp_calibration_statistic_function(data_cal-cal_mean)
                    
                    
                    if not dumb_threshold:
                        hyp_event_threshold = st_range*hyp_event_smart_threshold_factor
                    # with open("./print.txt", "a") as file:
                    #     file.write(str(hyp_event_threshold)+',')
                    calibrate = False
                continue


        ### CLASSIFIER ###
        
        # Event Detection
        interval = data_plot[hyp_detection_buffer_start_ind:-hyp_detection_buffer_end_ind] # Take middle part of window
        test_stat = hyp_test_statistic_function(interval) # Calculate test stat (defaults to range) 
        
        if flip_threshold:
            is_event = (test_stat < hyp_event_threshold) # Test threshold
        else:
            is_event = (test_stat > hyp_event_threshold) # Test threshold

        # Record History
        event_history[1::] = event_history[0:-1]
        event_history[0] = is_event


        # Pass window to classifier
        if np.all(event_history[0:hyp_consecutive_triggers]) and primed:
            if store_times:
                if nil_classifier:
                    start = time.time_ns()
                    prediction = 'L'
                    end = time.time_ns()
                    time_taken = end - start
                    classification_times.append(time_taken)
                else:
                    if using_zeroes_classifier:
                        if use_smart_hyp_zeroes_height_threshold:
                            start = time.time_ns()
                            prediction = classifier(data_plot, samprate, consec_seconds = zeroes_consec_threshold, ave_height = hyp_event_threshold)
                            end = time.time_ns()
                            time_taken = end - start
                            classification_times.append(time_taken)
                            if plot_zeroes_classifier and time_taken > 998800: 
                                plt.figure()
                                plt.plot(data_plot)

                        else:
                            start = time.time_ns()
                            prediction = classifier(data_plot, samprate, consec_seconds = zeroes_consec_threshold, ave_height = zeroes_height_threshold)
                            end = time.time_ns()
                            time_taken = end - start
                            classification_times.append(time_taken)
                            if plot_zeroes_classifier and time_taken > 50000: 
                                plt.figure()
                                plt.plot(data_plot)
                    else:
                        start = time.time_ns()
                        prediction = classifier(data_plot, samprate)
                        end = time.time_ns()
                        time_taken = end - start
                        classification_times.append(time_taken)
            else:
                if nil_classifier:
                    prediction = 'L'
                else:
                    if using_zeroes_classifier:
                        if use_smart_hyp_zeroes_height_threshold:
                            prediction = classifier(data_plot, samprate, consec_seconds = zeroes_consec_threshold, ave_height = hyp_event_threshold)
                        else:
                            prediction = classifier(data_plot, samprate, consec_seconds = zeroes_consec_threshold, ave_height = zeroes_height_threshold)
                    else:
                        prediction = classifier(data_plot, samprate)
            
#             msg = create_msg(prediction)
#             os.write(fifo, msg)
#fifo
            
            predictions += prediction
            

            if prediction=='R':
                prediction='R'
            else:
                prediction='L'
            

            content = f"{prediction}".encode("utf8")
            msg = create_msg(content)
            os.write(fifo, msg)

            #print(f"CONGRATULATIONS, ITS AN {prediction}!") if verbose else None

            if store_events:
                predictions_storage.append(data_plot)
            
            # Record time interval of event
            end_time = round(k*inputBufferSize/samprate, 2)
            start_time = round(end_time - window_size, 2)
            predictions_timestamps.append((start_time, end_time))
            
            timer = hyp_timeout

            primed = False
        
        if not timeout:
            if np.all(~event_history[0:hyp_consecutive_reset]):
                primed = True
        else:
            timer -= 1
            if timer < 0:
                primed = True


        ## PLOT ##

        if plot:
            t = (min(k+1,N_loops_over_window))*inputBufferSize/samprate*np.linspace(0,1,(data_plot).size)
            ax1.clear()
            # Debugging Annotations
            if np.all(event_history[0:hyp_consecutive_triggers]) and timer >0:
                ax1.annotate(f"ITS AN {prediction}!!!", (window_size/2, max_y-50))
            
            ax1.annotate(f"{event_history}", (window_size/2, max_y-70))
            ax1.set_xlim(0, window_size)
            ax1.set_ylim(min_y, max_y)
            plt.xlabel('time [s]')
            ax1.plot(t,data_plot)
            fig.canvas.draw()    
            plt.show()
    
    os.close(fifo)

    if store_events and store_times:
        return predictions, predictions_timestamps, predictions_storage, classification_times
    elif store_events:
        return predictions, predictions_timestamps, predictions_storage
    elif store_times:
        return predictions, predictions_timestamps, classification_times
    else:
        return predictions, predictions_timestamps