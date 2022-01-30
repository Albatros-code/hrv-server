from sklearn.linear_model import LinearRegression
import csv
import fitparse
import json
from plotnine import *

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import math

fit_file = fitparse.FitFile('./uploads/2022-01-27-12-16-30.fit')

# load RR intervals from the fit file
# RRs = []
# for record in fit_file.get_messages('record'):
#     for record_data in record:
#         print('-----------------------')
#         print(record_data)
#         for RR_interval in record_data.value:
#             if RR_interval is not None:
#                 RRs.append(RR_interval)

# Iterate over all messages of type "record"
# (other types include "device_info", "file_creator", "event", etc)


def get_data(fit_file, data_name):
    array = []
    for record in fit_file.get_messages("record"):
        # Records can contain multiple pieces of data (ex: timestamp, latitude, longitude, etc)
        for data in record:
            # Print the name and value of the data (and the units if it has any)
            # if data.units:
            #     # print(" * {}: {} ({})".format(data.name, data.value, data.units))
            # else:
            #     # print(" * {}: {}".format(data.name, data.value))
            if data.name == data_name:
                array.append(data.value)
    return array


def DFA(pp_values, lower_scale_limit, upper_scale_limit):
    scaleDensity = 30  # scales DFA is conducted between lower_scale_limit and upper_scale_limit
    # order of polynomial fit (linear = 1, quadratic m = 2, cubic m = 3, etc...)
    m = 1

    # initialize, we use logarithmic scales
    start = np.log(lower_scale_limit) / np.log(10)
    stop = np.log(upper_scale_limit) / np.log(10)
    scales = np.floor(np.logspace(np.log10(math.pow(10, start)),
                      np.log10(math.pow(10, stop)), scaleDensity))
    F = np.zeros(len(scales))
    count = 0

    for s in scales:
        rms = []
        # Step 1: Determine the "profile" (integrated signal with subtracted offset)
        x = pp_values
        y_n = np.cumsum(x - np.mean(x))
        # Step 2: Divide the profile into N non-overlapping segments of equal length s
        L = len(x)
        shape = [int(s), int(np.floor(L/s))]
        nwSize = int(shape[0]) * int(shape[1])
        # beginning to end, here we reshape so that we have a number of segments based on the scale used at this cycle
        Y_n1 = np.reshape(y_n[0:nwSize], shape, order="F")
        Y_n1 = Y_n1.T
        # end to beginning
        Y_n2 = np.reshape(y_n[len(y_n) - (nwSize):len(y_n)], shape, order="F")
        Y_n2 = Y_n2.T
        # concatenate
        Y_n = np.vstack((Y_n1, Y_n2))

        # Step 3: Calculate the local trend for each 2Ns segments by a least squares fit of the series
        for cut in np.arange(0, 2 * shape[1]):
            xcut = np.arange(0, shape[0])
            pl = np.polyfit(xcut, Y_n[cut, :], m)
            Yfit = np.polyval(pl, xcut)
            arr = Yfit - Y_n[cut, :]
            rms.append(np.sqrt(np.mean(arr * arr)))

        if (len(rms) > 0):
            F[count] = np.power((1 / (shape[1] * 2)) *
                                np.sum(np.power(rms, 2)), 1/2)
        count = count + 1

    pl2 = np.polyfit(np.log2(scales), np.log2(F), 1)
    alpha = pl2[0]
    return alpha


def computeFeatures(df, x):
    features = []
    step = 120
    # for index in range(3, 13):
    for index in range(0, int(round(np.max(x)/step))):

        array_rr = df.loc[(df['timestamp'] >= (index*step))
                          & (df['timestamp'] <= (index+1)*step), 'RR']*1000
        # compute heart rate
        heartrate = round(60000/np.mean(array_rr), 2)
        # compute rmssd
        NNdiff = np.abs(np.diff(array_rr))
        rmssd = round(np.sqrt(np.sum((NNdiff * NNdiff) / len(NNdiff))), 2)
        # compute sdnn
        sdnn = round(np.std(array_rr), 2)
        # dfa, alpha 1
        alpha1 = DFA(array_rr.to_list(), 4, 16)

        curr_features = {
            'timestamp': index*step,
            'heartrate': heartrate,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'alpha1': alpha1,
        }

        features.append(curr_features)

    features_df = pd.DataFrame(features)
    return features_df


def calculate(fit_file):
    RRs = []
    for record in fit_file.get_messages('hrv'):
        for record_data in record:
            for RR_interval in record_data.value:
                if RR_interval is not None:
                    RRs.append(RR_interval)
    # Removing artifacts and plotting the RR intervals
    artifact_correction_threshold = 0.05
    filtered_RRs = []
    for i in range(len(RRs)):
        if RRs[(i-1)]*(1-artifact_correction_threshold) < RRs[i] < RRs[(i-1)]*(1+artifact_correction_threshold):
            filtered_RRs.append(RRs[i])

    x = np.cumsum(filtered_RRs)

    df = pd.DataFrame()
    df['timestamp'] = x
    df['RR'] = filtered_RRs

    features_df = computeFeatures(df, x)
    features_df.head()
    # Update #1: clean up and color coding
    threshold_sdnn = 10  # rather arbitrary, based on visual inspection of the data
    features_df_filtered = features_df.loc[features_df['sdnn']
                                           < threshold_sdnn, ]

    average_alfa1 = round(np.mean(features_df['alpha1']), 2)

    return {"df": features_df_filtered, "hrv_avg": average_alfa1}


# heart_rate = get_data(fit_file, 'heart_rate')
# timestamp = get_data(fit_file, 'timestamp')
result = calculate(fit_file)
# data = {
#     "timestamp": [x.isoformat() for x in timestamp if True],
#     "heart_rate": heart_rate,
# "hrv": hrv["hrv"],
# "hrv_avg": hrv["hrv_avg"],
# }
parsed_data = [{"time": x[0], "hr": round(np.mean(x[1]), 0), "alfa1": round(np.mean(x[4]), 2)}
               for x in result['df'].values.tolist() if True]

res = {
    "results": parsed_data,
    "hrv_avg": result['hrv_avg']
}

print(json.dumps(res))
# print(data)
