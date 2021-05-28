import serial
import numpy as np
import scipy.signal


def read_arduino(ser, inputBufferSize):
    data = ser.read(inputBufferSize)
    out = [(int(data[i])) for i in range(0, len(data))]
    return out


def process_data(data):
    data_in = np.array(data)
    result = []
    i = 1
    while i < len(data_in) - 1:
        if data_in[i] > 127:
            # Found beginning of frame
            # Extract one sample from 2 bytes
            intout = (np.bitwise_and(data_in[i], 127)) * 128
            i = i + 1
            intout = intout + data_in[i]
            result = np.append(result, intout)
        i = i + 1
    return result

##gai
def eye_detection(interval):
    max_index = interval.index(max(interval))
    min_index = interval.index(min(interval))
    if max_index < min_index:
        return "L"
    else:
        return "R"


def LR_detection(wave_list, count=0, event=0, amplitude=800, limit=2000, length=5000):
    seq_length = len(wave_list)
    
    ####gaigaigai

    # std=np.std(wave_list)
    # diff=np.max(wave_list)-np.min(wave_list)
    # peaks=len(scipy.signal.find_peaks(wave_list, height=30, prominence=20)[0])

    # maxval=np.argmax(wave_list)
    # minval=np.argmin(wave_list)


    ####gaigaigai



    
    event_table = [-1] * seq_length
    event_string = ""
    for i in range(0, seq_length - 1):
        #gaiiiii
        if wave_list[i] > 1000 and event == 0:
            event = 3
            event_string += "B"
        elif wave_list[i] > amplitude  and event == 0:
            event = 1
            event_string += "L"

        # if wave_list[i] > amplitude and event==0:
        #     if wave_list[i]>1000:
        #         event=3
        #         event_string+='B'
        #     else:
        #         event=1
        #         event_string+='L'

            #change_change
        elif event == 0 and wave_list[i]<-800:# < (0 - amplitude) :
            event = 2
            event_string += "R"
            ######3##
        # else:
        #     event_string+='R'

        if event != 0:
            event_table.append(event)
            if wave_list[i] < limit and wave_list[i] > (0 - limit):
                count += 1
            elif count > length:
                count = 0
                event = 0

        # event_string += " "
        # event_string += str(wave_list[i])

    return event_table, event_string



# def LR_detection(wave_list, count=0, event=0, amplitude=800, limit=2000, length=5000):
#     seq_length = len(wave_list)
    
#     ####gaigaigai

#     std=np.std(wave_list)
#     diff=np.max(wave_list)-np.min(wave_list)
#     peaks=len(scipy.signal.find_peaks(wave_list, height=30, prominence=20)[0])

#     maxval=np.argmax(wave_list)
#     minval=np.argmin(wave_list)



#     event_table = [-1] * seq_length
#     event_string=''
    
    
#     if peaks>9:# and event == 0:
        
#         event_string += "F"
#     elif std>14 and diff>70:   ###change 70
#         if maxval>minval:
#             event_string+='L'
#         else:
#             event_string+='R'
#     else:
#         event_string+='N'

    
    


    



    ####gaigaigai
#####commenthere
# def old_main():
#     # Read example data
#     baudrate = 230400
#     cport = "/dev/cu.usbserial-DJ00DVON"  # set the correct port before run it
#     ser = serial.Serial(port=cport, baudrate=baudrate)

#     inputBufferSize = 10000  # keep betweein 2000-20000
#     ser.timeout = inputBufferSize / 20000.0  # set read timeout, 20000 is one second
#     # ser.set_buffer_size(rx_size = inputBufferSize)

#     total_time = 30.0;  # time in seconds [[1 s = 20000 buffer size]]
#     max_time = 10.0;  # time plotted in window [s]
#     N_loops = 20000.0 / inputBufferSize * total_time

#     T_acquire = inputBufferSize / 20000.0  # length of time that data is acquired for
#     N_max_loops = max_time / T_acquire  # total number of loops to cover desire time window

#     k = 0
#     while True:
#     # while k < N_loops:  # Will end early so can't run forever.
#         data = read_arduino(ser, inputBufferSize)
#         data_temp = process_data(data)
#         # if k <= N_max_loops:
#         #     if k == 0:
#         #         data_plot = data_temp
#         #     else:
#         #         data_plot = np.append(data_temp, data_plot)
#         #     t = (min(k + 1, N_max_loops)) * inputBufferSize / 20000.0 * np.linspace(0, 1, (data_plot).size)
#         # else:
#         #     data_plot = np.roll(data_plot, len(data_temp))
#         #     data_plot[0:len(data_temp)] = data_temp
#         # t = (min(k + 1, N_max_loops)) * inputBufferSize / 20000.0 * np.linspace(0, 1, (data_plot).size)

#         # (plt.plot(data_temp))
#         detect = LR_detection(data_temp, length=len(data_temp))[1]
#         if len(detect) > 0:
#             print(detect)

#         k += 1




####add
    # def Detection(seq, std_thresh, diff_thresh, prom_threshold):
    #     std = np.std(seq)
    #     diff = np.max(seq) - np.min(seq)
    #     peaks = len(scipy.signal.find_peaks(seq, prominence=prom_threshold)[0])
    #     maxval = np.argmax(seq)
    #     minval = np.argmin(seq)
    #     if peaks > 2:
    #         return 'fl'
    #     elif std > std_thresh and diff > diff_thresh:
    #         if maxval > minval:
    #             return 'L'
    #         else:
    #             return 'R'
    #     else:
    #         return 'NA'




###add






def main():
    baudrate = 230400
    cport = "/dev/cu.usbserial-DJ00E328"  # set the correct port before run it
    ser = serial.Serial(port=cport, baudrate=baudrate)

    inputBufferSize = 9000  # keep betweein 2000-20000
    ser.timeout = inputBufferSize / 20000.0  # set read timeout, 20000 is one second

    sd_threshold = 1
    increment = inputBufferSize / 10

    # lower_interval = 1
    # max_time = max(xtime) * window_size
    #
    # while (max_time > lower_interval + window_size) {
    # upper_interval = lower_interval + window_size
    # interval = Y[lower_interval:upper_interval]
    # testStat < - sd(interval)
    # if (testStat > thresholdEvents)
    # {
    #     predicted = LR_detection(interval)
    # predicted_labels = c(predicted_labels, predicted)
    # predicted_time = c(predicted_time, lower_interval + window_size / 2)
    # lower_interval < - lower_interval + window_size
    # } else {
    #     lower_interval < - lower_interval + increment
    # }
    # }  ## end while
    # return (paste(predicted_labels, collapse=""))

    while True:
        data = read_arduino(ser, inputBufferSize)
        data_temp = process_data(data)

        # mean = sum(data_temp) / len(data_temp)
        # variance = sum([((x - mean) ** 2) for x in data_temp]) / len(data_temp)
        # sd = variance ** 0.5
        
        
        # print(sd)
        # print(data_temp)


#gai
        # if sd > sd_threshold:
        #     predicted = eye_detection(data_temp)
        #     print(predicted)

        # detect=LR_detection(data_temp,length=len(data_temp))

        
        
        
        detect = LR_detection(data_temp, length=len(data_temp))[1]
        if len(detect) > 0:
            print(detect)


if __name__ == '__main__':
    main()

    
