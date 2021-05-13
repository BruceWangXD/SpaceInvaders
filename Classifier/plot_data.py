import numpy as np
import matplotlib.pyplot as plt



def plot_labelled_wave(wav_array, samprate, labels_dat, ax, i, title="", before_buffer = 0.7,
                       after_buffer = 1, shade_alpha=0.2, wave_alpha=1, ymin = -100, ymax = 100):
    
    time_seq = np.linspace(1, len(wav_array), len(wav_array))/samprate

    

    left_events_bool = np.array([False]*len(time_seq))
    for time in labels_dat.time[labels_dat.label == "L"]:
        left_events_bool = ( (time_seq > time - before_buffer) & (time_seq < time+after_buffer) ) | left_events_bool

    right_events_bool = np.array([False]*len(time_seq))
    for time in labels_dat.time[labels_dat.label == "R"]:
        right_events_bool = ( (time_seq > time - before_buffer) & (time_seq < time+after_buffer) ) | right_events_bool



    ax[i].plot(time_seq, wav_array, alpha=wave_alpha)
    
    

    ax[i].fill_between(time_seq, ymax, ymin,
                     where = left_events_bool,
                     color = 'g',
                     alpha=shade_alpha)

    ax[i].fill_between(time_seq, ymax, ymin,
                     where = right_events_bool,
                     color = 'r',
                     alpha=shade_alpha)
    
    ax[i].set_title(title)
    
    
    
    
def plot_predictions(wav_array, samprate, labels_dat, predictions, predictions_timestamps, ax, i,
                     title="", before_buffer = 0.7, after_buffer = 1, actual_alpha=0.2,
                     wave_alpha=1, pred_alpha = 0.5, miny = -100, maxy = 100):
    
    time_seq = np.linspace(1, len(wav_array), len(wav_array))/samprate

    

    left_events_bool = np.array([False]*len(time_seq))
    for time in labels_dat.time[labels_dat.label == "L"]:
        left_events_bool = ( (time_seq > time - before_buffer) & (time_seq < time+after_buffer) ) | left_events_bool

    right_events_bool = np.array([False]*len(time_seq))
    for time in labels_dat.time[labels_dat.label == "R"]:
        right_events_bool = ( (time_seq > time - before_buffer) & (time_seq < time+after_buffer) ) | right_events_bool
        
    left_preds_bool = np.array([False]*len(time_seq))
    right_preds_bool = np.array([False]*len(time_seq))
    idk_preds_bool = np.array([False]*len(time_seq))
    for pred, times in zip(predictions, predictions_timestamps):
        if pred == "L":
            left_preds_bool = ( (time_seq > times[0]) & (time_seq < times[1]) ) | left_preds_bool
        elif pred == "R":
            right_preds_bool = ( (time_seq > times[0]) & (time_seq < times[1]) ) | right_preds_bool
        else:
            idk_preds_bool = ( (time_seq > times[0]) & (time_seq < times[1]) ) | idk_preds_bool
            

    ax[i].plot(time_seq, wav_array, alpha=wave_alpha)

    
    # Plot actuals
    ax[i].fill_between(time_seq, maxy, miny,
                     where = left_events_bool,
                     color = 'g',
                     label = "L",
                     alpha=actual_alpha)

    ax[i].fill_between(time_seq, maxy, miny,
                     where = right_events_bool,
                     color = 'r',
                     label = "R",
                     alpha=actual_alpha)
    
    # Plot predictions
    ax[i].fill_between(time_seq, maxy, miny,
                     where = left_preds_bool,
                     color = 'g',
                     label = "Pred L",
                     alpha=pred_alpha)

    ax[i].fill_between(time_seq, maxy, miny,
                     where = right_preds_bool,
                     color = 'r',
                     label = "Pred R",
                     alpha=pred_alpha)
    
    ax[i].fill_between(time_seq, maxy, miny,
                     where = idk_preds_bool,
                     color = 'y',
                     label = "Pred idk",
                     alpha=pred_alpha)
    
    ax[i].set_title(title)
    ax[i].legend()
    