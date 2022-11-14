#import requisite modules/libraries (needs cleaned up)
from ast import increment_lineno
from audioop import avg
from concurrent.futures.thread import _threads_queues
from optparse import Values
import os
from statistics import mean, variance
import sys
import glob
from sklearn import preprocessing
import json
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax
import keras_tuner
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
import seaborn
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#settings for where main test/train data is (json_filepath) and where one-off song prediction file(s) are (predict_this_filepath)
json_filepath = "D:\Coding\AMGR\ML\MachineLearningV0.1\data"
predict_this_filepath = "D:\\Coding\\AMGR\ML\\MachineLearningV0.1 - Copy\\predict_these\\reggae.bujubanton.json"

#create empty array for song characteristics to be added to
features = []

#create class for analysis target objects
class Analysis_Target:
#define class properties
    def __init__(
#basic properties
                self, 
                genre,
                songName,
                lengthInSamples,
                statsLocation,
#characteristic availability check
                mfcc_available,
                rms_available,
                spectralCentroid_available,
                spectralBandwidth_available, 
                spectralContrast_available,
                spectralFlatness_available,
                spectralRolloff_available,
                polyFeatures_available,
                tonalCentroid_available,
                zeroXingRate_available,
                beatTrack_available,
                localPulse_available,
                tempoEstimate_available,
#characteristics actual
                #mfcc x20 frames                
                mfcc1_mean,
                mfcc1_variance,
                mfcc2_mean,
                mfcc2_variance,
                mfcc3_mean,
                mfcc3_variance,
                mfcc4_mean,
                mfcc4_variance,
                mfcc5_mean,
                mfcc5_variance,
                mfcc6_mean,
                mfcc6_variance,
                mfcc7_mean,
                mfcc7_variance,
                mfcc8_mean,
                mfcc8_variance,
                mfcc9_mean,
                mfcc9_variance,
                mfcc10_mean,
                mfcc10_variance,
                mfcc11_mean,
                mfcc11_variance,
                mfcc12_mean,
                mfcc12_variance,
                mfcc13_mean,
                mfcc13_variance,
                mfcc14_mean,
                mfcc14_variance,
                mfcc15_mean,
                mfcc15_variance,
                mfcc16_mean,
                mfcc16_variance,
                mfcc17_mean,
                mfcc17_variance,
                mfcc18_mean,
                mfcc18_variance,
                mfcc19_mean,
                mfcc19_variance,
                mfcc20_mean,
                mfcc20_variance,
                #rms
                rms_mean,
                rms_variance,
                #spectral centroid
                spectralCentroid_mean,
                spectralCentroid_variance,
                #spectral bandwidth
                spectralBandwidth_mean,
                spectralBandwidth_variance,
                #spectral flatness
                spectralFlatness_mean,
                spectralFlatness_variance,
                #spectral contrast
                spectralContrast1_mean,
                spectralContrast1_variance,
                spectralContrast2_mean,
                spectralContrast2_variance,
                spectralContrast3_mean,
                spectralContrast3_variance,
                spectralContrast4_mean,
                spectralContrast4_variance,
                spectralContrast5_mean,
                spectralContrast5_variance,
                spectralContrast6_mean,
                spectralContrast6_variance,
                spectralContrast7_mean,
                spectralContrast7_variance,
                #spectral rolloff
                spectralRolloff_mean,
                spectralRolloff_variance,
                #poly features (for 0d [constants] model, would need to be expanded for multi dimensionality)
                polyFeatures_mean,
                polyFeatures_variance,
                #tonal centroid 6 dimensional
                tonalCentroidD1_mean,
                tonalCentroidD1_variance,
                tonalCentroidD2_mean,
                tonalCentroidD2_variance,
                tonalCentroidD3_mean,
                tonalCentroidD3_variance,
                tonalCentroidD4_mean,
                tonalCentroidD4_variance,
                tonalCentroidD5_mean,
                tonalCentroidD5_variance,
                tonalCentroidD6_mean,
                tonalCentroidD6_variance,
                #zero crossing rate
                xeroXingRate_mean,
                xeroXingRate_variance,
                #beat track difference in values between beats
                beatTrackAvgJumpDiffs_average,
                beatTrackAvgJumpDiffs_variance,
                #beat track array length (alternative indication of tempo)
                beatTrackArrayLength,
                #predominant local pulse including 0 values
                localPulse_mean,
                localPulse_variance,
                #predominant local pulse excluding 0 values
                localPulseZeroless_mean,
                localPulseZeroless_variance,
                #tempo estimate
                tempoEstimate
                ):

#basic properties
        self.genre = genre
        self.songName = songName
        self.lengthInSamples = lengthInSamples
        self.statsLocation = statsLocation
#toggleable characteristic properties which trigger based on config status
        self.mfcc_available = mfcc_available
        self.rms_available = rms_available
        self.spectralCentroid_available = spectralCentroid_available
        self.spectralBandwidth_available = spectralBandwidth_available
        self.spectralContrast_available = spectralContrast_available
        self.spectralFlatness_available = spectralFlatness_available
        self.spectralRolloff_available = spectralRolloff_available
        self.polyFeatures_available= polyFeatures_available
        self.tonalCentroid_available = tonalCentroid_available
        self.zeroXingRate_available = zeroXingRate_available
        self.beatTrack_available = beatTrack_available
        self.localPulse_available = localPulse_available
        self.tempoEstimate_available = tempoEstimate_available
#toggleable characteristic properties which trigger based on config status
        self.mfcc1_mean = mfcc1_mean
        self.mfcc1_variance = mfcc1_variance
        self.mfcc2_mean = mfcc2_mean
        self.mfcc2_variance = mfcc2_variance
        self.mfcc3_mean = mfcc3_mean
        self.mfcc3_variance = mfcc3_variance
        self.mfcc4_mean = mfcc4_mean
        self.mfcc4_variance = mfcc4_variance
        self.mfcc5_mean = mfcc5_mean
        self.mfcc5_variance = mfcc5_variance
        self.mfcc6_mean = mfcc6_mean
        self.mfcc6_variance = mfcc6_variance
        self.mfcc7_mean = mfcc7_mean
        self.mfcc7_variance = mfcc7_variance
        self.mfcc8_mean = mfcc8_mean
        self.mfcc8_variance = mfcc8_variance
        self.mfcc9_mean = mfcc9_mean
        self.mfcc9_variance = mfcc9_variance
        self.mfcc10_mean = mfcc10_mean
        self.mfcc10_variance = mfcc10_variance
        self.mfcc11_mean = mfcc11_mean
        self.mfcc11_variance = mfcc11_variance
        self.mfcc12_mean = mfcc12_mean
        self.mfcc12_variance = mfcc12_variance
        self.mfcc13_mean = mfcc13_mean
        self.mfcc13_variance = mfcc13_variance
        self.mfcc14_mean = mfcc14_mean
        self.mfcc14_variance = mfcc14_variance
        self.mfcc15_mean = mfcc15_mean
        self.mfcc15_variance = mfcc15_variance
        self.mfcc16_mean = mfcc16_mean
        self.mfcc16_variance = mfcc16_variance
        self.mfcc17_mean = mfcc17_mean
        self.mfcc17_variance = mfcc17_variance
        self.mfcc18_mean = mfcc18_mean
        self.mfcc18_variance = mfcc18_variance
        self.mfcc19_mean = mfcc19_mean
        self.mfcc19_variance = mfcc19_variance
        self.mfcc20_mean = mfcc20_mean 
        self.mfcc20_variance = mfcc20_variance
        #rms
        self.rms_mean = rms_mean
        self.rms_variance = rms_variance
        #spectral centroid
        self.spectralCentroid_mean = spectralCentroid_mean
        self.spectralCentroid_variance = spectralCentroid_variance
        #spectral bandwidth
        self.spectralBandwidth_mean = spectralBandwidth_mean
        self.spectralBandwidth_variance = spectralBandwidth_variance
        #spectral flatness
        self.spectralFlatness_mean = spectralFlatness_mean
        self.spectralFlatness_variance = spectralFlatness_variance
        #spectral contrast
        self.spectralContrast1_mean = spectralContrast1_mean
        self.spectralContrast1_variance = spectralContrast1_variance
        self.spectralContrast2_mean = spectralContrast2_mean
        self.spectralContrast2_variance = spectralContrast2_variance
        self.spectralContrast3_mean = spectralContrast3_mean
        self.spectralContrast3_variance = spectralContrast3_variance
        self.spectralContrast4_mean = spectralContrast4_mean
        self.spectralContrast4_variance = spectralContrast4_variance
        self.spectralContrast5_mean = spectralContrast5_mean
        self.spectralContrast5_variance = spectralContrast5_variance
        self.spectralContrast6_mean = spectralContrast6_mean
        self.spectralContrast6_variance = spectralContrast6_variance
        self.spectralContrast7_mean = spectralContrast7_mean
        self.spectralContrast7_variance = spectralContrast7_variance
        #spectral rolloff
        self.spectralRolloff_mean = spectralRolloff_mean
        self.spectralRolloff_variance = spectralRolloff_variance
        #poly features (for 0d [constants] model, would need to be expanded for multi dimensionality)
        self.polyFeatures_mean = polyFeatures_mean
        self.polyFeatures_variance = polyFeatures_variance
        #tonal centroid 6 dimensional
        self.tonalCentroidD1_mean = tonalCentroidD1_mean
        self.tonalCentroidD1_variance = tonalCentroidD1_variance
        self.tonalCentroidD2_mean = tonalCentroidD2_mean
        self.tonalCentroidD2_variance = tonalCentroidD2_variance
        self.tonalCentroidD3_mean = tonalCentroidD3_mean
        self.tonalCentroidD3_variance = tonalCentroidD3_variance
        self.tonalCentroidD4_mean = tonalCentroidD4_mean
        self.tonalCentroidD4_variance = tonalCentroidD4_variance
        self.tonalCentroidD5_mean = tonalCentroidD5_mean
        self.tonalCentroidD5_variance = tonalCentroidD5_variance
        self.tonalCentroidD6_mean = tonalCentroidD6_mean
        self.tonalCentroidD6_variance = tonalCentroidD6_variance
        #zero crossing rate
        self.xeroXingRate_mean = xeroXingRate_mean
        self.xeroXingRate_variance = xeroXingRate_variance
        #beat track difference in values between beats
        self.beatTrackAvgJumpDiffs_average = beatTrackAvgJumpDiffs_average
        self.beatTrackAvgJumpDiffs_variance = beatTrackAvgJumpDiffs_variance
        #beat track array length (alternative indication of tempo)
        self.beatTrackArrayLength = beatTrackArrayLength
        #predominant local pulse including 0 values
        self.localPulse_mean = localPulse_mean
        self.localPulse_variance = localPulse_variance
        #predominant local pulse excluding 0 values
        self.localPulseZeroless_mean = localPulseZeroless_mean
        self.localPulseZeroless_variance = localPulseZeroless_variance
        #tempo estimate
        self.tempoEstimate = tempoEstimate     

json_input_list = glob.glob(pathname = ('%s/*.json' %(json_filepath)))
json_input_list_length = len(json_input_list)
json_list_length_counter = json_input_list_length
counter_secondary = 1
current_json_location = json_input_list[(json_input_list_length - counter_secondary)]

def add_for_analysis():
    current_analysis_location = current_json_location
    with open(current_analysis_location, 'r') as f:
        data = json.load(f)
    current_song_genre = data["genre"]
    current_song_length = data["audioLength"]
    current_song_filename = data["fileName"]
    current_stats_location = current_json_location
    mfcc_available_var = data["mfcc_status"]
    if not mfcc_available_var:
        print('WARNING: mfcc_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    rms_available_var = data["rms_status"]
    if not rms_available_var:
        print('WARNING: rms_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    spectralCentroid_available_var = data["rms_status"]
    if not spectralCentroid_available_var:
        print('WARNING: spectralCentroid_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    spectralBandwidth_available_var = data["spectralBandwidth_status"]
    if not spectralBandwidth_available_var:
        print('WARNING: spectralBandwidth_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    spectralContrast_available_var = data["spectralContrast_status"]
    if not spectralContrast_available_var:
        print('WARNING: spectralContrast_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    spectralFlatness_available_var = data["spectralFlatness_status"]
    if not spectralFlatness_available_var:
        print('WARNING: spectralFlatness_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    spectralRolloff_available_var = data["spectralRolloff_status"]
    if not spectralRolloff_available_var:
        print('WARNING: spectralRolloff_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    polyFeatures_available_var = data["polyFeatures_status"]
    if not polyFeatures_available_var:
        print('WARNING: polyFeatures_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    tonalCentroid_available_var = data["tonalCentroid_status"]
    if not tonalCentroid_available_var:
        print('WARNING: tonalCentroid_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    zeroXingRate_available_var = data["zeroXingRate_status"]
    if not zeroXingRate_available_var:
        print('WARNING: zeroXingRate_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    beatTrack_available_var = data["beatTrack_status"]
    if not beatTrack_available_var:
        print('WARNING: beatTrack_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    localPulse_available_var = data["localPulse_status"]
    if not localPulse_available_var:
        print('WARNING: localPulse_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    tempoEstimate_available_var = data["tempoEstimate_status"]
    if not tempoEstimate_available_var:
        print('WARNING: tempoEstimate_data not available!\nMake sure this data is present in the song \'%s\' located at \'%s\' before continuing...' %(current_song_filename, current_json_location))
    
    #Need further processing before appending
    raw_mfcc_data = data["mfcc"] if mfcc_available_var else None
    raw_rms_data = data["rms"] if rms_available_var else None
    raw_spectralCentroid_data = data["spectralCentroid"] if spectralCentroid_available_var else None
    raw_spectralBandwidth_data = data["spectralBandwidth"] if spectralBandwidth_available_var else None
    raw_spectralContrast_data = data["spectralContrast"] if spectralContrast_available_var else None
    raw_spectralFlatness_data = data["spectralFlatness"] if spectralFlatness_available_var else None
    raw_spectralRolloff_data = data["spectralRolloff"] if spectralRolloff_available_var else None
    raw_polyFeatures_data = data["polyFeatures"] if polyFeatures_available_var else None
    raw_tonalCentroid_data = data["tonalCentroid"] if tonalCentroid_available_var else None
    raw_zeroXingRate_data = data["zeroXingRate"] if zeroXingRate_available_var else None
    raw_beatTrack_data = data["beatTrack"] if beatTrack_available_var else None
    raw_localPulse_data = data["localPulse"] if localPulse_available_var else None
    
    #Already in format to be appended
    formatted_tempoEstimate = data["tempoEstimate"] if tempoEstimate_available_var else None
    #mfccs split array (frames)
    formatted_mfcc1 = raw_mfcc_data[0]
    formatted_mfcc2 = raw_mfcc_data[1]
    formatted_mfcc3 = raw_mfcc_data[2]
    formatted_mfcc4 = raw_mfcc_data[3]
    formatted_mfcc5 = raw_mfcc_data[4]
    formatted_mfcc6 = raw_mfcc_data[5]
    formatted_mfcc7 = raw_mfcc_data[6]
    formatted_mfcc8 = raw_mfcc_data[7]
    formatted_mfcc9 = raw_mfcc_data[8]
    formatted_mfcc10 = raw_mfcc_data[9]
    formatted_mfcc11 = raw_mfcc_data[10]
    formatted_mfcc12 = raw_mfcc_data[11]
    formatted_mfcc13 = raw_mfcc_data[12]
    formatted_mfcc14 = raw_mfcc_data[13]
    formatted_mfcc15 = raw_mfcc_data[14]
    formatted_mfcc16 = raw_mfcc_data[15]
    formatted_mfcc17 = raw_mfcc_data[16]
    formatted_mfcc18 = raw_mfcc_data[17]
    formatted_mfcc19 = raw_mfcc_data[18]
    formatted_mfcc20 = raw_mfcc_data[19]
    #spectral contrast split array (frequency/octave bands)
    formatted_spectralContrast1 = raw_spectralContrast_data[0]
    formatted_spectralContrast2 = raw_spectralContrast_data[1]
    formatted_spectralContrast3 = raw_spectralContrast_data[2]
    formatted_spectralContrast4 = raw_spectralContrast_data[3]
    formatted_spectralContrast5 = raw_spectralContrast_data[4]
    formatted_spectralContrast6 = raw_spectralContrast_data[5]
    formatted_spectralContrast7 = raw_spectralContrast_data[6]
    #tonal centroid split array (musical interval presence)
    formatted_tonalCentroidD1 = raw_tonalCentroid_data[0]
    formatted_tonalCentroidD2 = raw_tonalCentroid_data[1]
    formatted_tonalCentroidD3 = raw_tonalCentroid_data[2]
    formatted_tonalCentroidD4 = raw_tonalCentroid_data[3]
    formatted_tonalCentroidD5 = raw_tonalCentroid_data[4]
    formatted_tonalCentroidD6 = raw_tonalCentroid_data[5]
    #simplify array (where it is 2d but has one element)
    formatted_rms = raw_rms_data[0]
    formatted_spectralCentroid = raw_spectralCentroid_data[0]
    formatted_spectralBandwidth = raw_spectralBandwidth_data[0]
    formatted_spectralFlatness = raw_spectralFlatness_data[0]
    formatted_spectralRolloff = raw_spectralRolloff_data[0]
    formatted_polyFeatures = raw_polyFeatures_data[0]
    formatted_zeroXingRate = raw_zeroXingRate_data[0]
    #get beat track jump differences
    beatTrack_length = len(raw_beatTrack_data)
    beatTrack_counter = 0
    formatted_beatTrack_diffs = []
    while beatTrack_counter < (beatTrack_length - 1):
        formatted_beatTrack_diffs.append(np.absolute((raw_beatTrack_data[beatTrack_counter]) - (raw_beatTrack_data[beatTrack_counter+1])))
        beatTrack_counter += 1
    #beat track length
    formatted_beatTrack_length = len(raw_beatTrack_data)
    #remove 0s from 1 set of local pulse
    formatted_localPulse = raw_localPulse_data
    formatted_localPulseZeroless = list(filter(lambda x: x != 0, raw_localPulse_data))

    parsed_mfcc1_mean = mean(formatted_mfcc1)
    parsed_mfcc2_mean = mean(formatted_mfcc2)
    parsed_mfcc3_mean = mean(formatted_mfcc3)
    parsed_mfcc4_mean = mean(formatted_mfcc4)
    parsed_mfcc5_mean = mean(formatted_mfcc5)
    parsed_mfcc6_mean = mean(formatted_mfcc6)
    parsed_mfcc7_mean = mean(formatted_mfcc7)
    parsed_mfcc8_mean = mean(formatted_mfcc8)
    parsed_mfcc9_mean = mean(formatted_mfcc9)
    parsed_mfcc10_mean = mean(formatted_mfcc10)
    parsed_mfcc11_mean = mean(formatted_mfcc11)
    parsed_mfcc12_mean = mean(formatted_mfcc12)
    parsed_mfcc13_mean = mean(formatted_mfcc13)
    parsed_mfcc14_mean = mean(formatted_mfcc14)
    parsed_mfcc15_mean = mean(formatted_mfcc15)
    parsed_mfcc16_mean = mean(formatted_mfcc16)
    parsed_mfcc17_mean = mean(formatted_mfcc17)
    parsed_mfcc18_mean = mean(formatted_mfcc18)
    parsed_mfcc19_mean = mean(formatted_mfcc19)
    parsed_mfcc20_mean = mean(formatted_mfcc20)
    parsed_mfcc1_variance = variance(formatted_mfcc1)
    parsed_mfcc2_variance = variance(formatted_mfcc2)
    parsed_mfcc3_variance = variance(formatted_mfcc3)
    parsed_mfcc4_variance = variance(formatted_mfcc4)
    parsed_mfcc5_variance = variance(formatted_mfcc5)
    parsed_mfcc6_variance = variance(formatted_mfcc6)
    parsed_mfcc7_variance = variance(formatted_mfcc7)
    parsed_mfcc8_variance = variance(formatted_mfcc8)
    parsed_mfcc9_variance = variance(formatted_mfcc9)
    parsed_mfcc10_variance = variance(formatted_mfcc10)
    parsed_mfcc11_variance = variance(formatted_mfcc11)
    parsed_mfcc12_variance = variance(formatted_mfcc12)
    parsed_mfcc13_variance = variance(formatted_mfcc13)
    parsed_mfcc14_variance = variance(formatted_mfcc14)
    parsed_mfcc15_variance = variance(formatted_mfcc15)
    parsed_mfcc16_variance = variance(formatted_mfcc16)
    parsed_mfcc17_variance = variance(formatted_mfcc17)
    parsed_mfcc18_variance = variance(formatted_mfcc18)
    parsed_mfcc19_variance = variance(formatted_mfcc19)
    parsed_mfcc20_variance = variance(formatted_mfcc20)
    
    parsed_spectralContrast1_mean = mean(formatted_spectralContrast1)
    parsed_spectralContrast2_mean = mean(formatted_spectralContrast2)
    parsed_spectralContrast3_mean = mean(formatted_spectralContrast3)
    parsed_spectralContrast4_mean = mean(formatted_spectralContrast4)
    parsed_spectralContrast5_mean = mean(formatted_spectralContrast5)
    parsed_spectralContrast6_mean = mean(formatted_spectralContrast6)
    parsed_spectralContrast7_mean = mean(formatted_spectralContrast7)
    parsed_spectralContrast1_variance = variance(formatted_spectralContrast1)
    parsed_spectralContrast2_variance = variance(formatted_spectralContrast2)
    parsed_spectralContrast3_variance = variance(formatted_spectralContrast3)
    parsed_spectralContrast4_variance = variance(formatted_spectralContrast4)
    parsed_spectralContrast5_variance = variance(formatted_spectralContrast5)
    parsed_spectralContrast6_variance = variance(formatted_spectralContrast6)
    parsed_spectralContrast7_variance = variance(formatted_spectralContrast7)
    
    parsed_tonalCentroidD1_mean = mean(formatted_tonalCentroidD1)
    parsed_tonalCentroidD2_mean = mean(formatted_tonalCentroidD2)
    parsed_tonalCentroidD3_mean = mean(formatted_tonalCentroidD3)
    parsed_tonalCentroidD4_mean = mean(formatted_tonalCentroidD4)
    parsed_tonalCentroidD5_mean = mean(formatted_tonalCentroidD5)
    parsed_tonalCentroidD6_mean = mean(formatted_tonalCentroidD6)
    parsed_tonalCentroidD1_variance = variance(formatted_tonalCentroidD1)
    parsed_tonalCentroidD2_variance = variance(formatted_tonalCentroidD2)
    parsed_tonalCentroidD3_variance = variance(formatted_tonalCentroidD3)
    parsed_tonalCentroidD4_variance = variance(formatted_tonalCentroidD4)
    parsed_tonalCentroidD5_variance = variance(formatted_tonalCentroidD5)
    parsed_tonalCentroidD6_variance = variance(formatted_tonalCentroidD6)

    parsed_rms_mean = mean(formatted_rms)
    parsed_rms_variance = variance(formatted_rms)
    parsed_spectralCentroid_mean = mean(formatted_spectralCentroid)
    parsed_spectralCentroid_variance = variance(formatted_spectralCentroid)
    parsed_spectralBandwidth_mean = mean(formatted_spectralBandwidth)
    parsed_spectralBandwidth_variance = variance(formatted_spectralBandwidth)
    parsed_spectralFlatness_mean = mean(formatted_spectralFlatness)
    parsed_spectralFlatness_variance = variance(formatted_spectralFlatness)
    parsed_spectralRolloff_mean = mean(formatted_spectralRolloff)
    parsed_spectralRolloff_variance = variance(formatted_spectralRolloff)
    parsed_polyFeatures_mean = mean(formatted_polyFeatures)
    parsed_polyFeatures_variance = variance(formatted_polyFeatures)
    parsed_zeroXingRate_mean = mean(formatted_zeroXingRate)
    parsed_zeroXingRate_variance = variance(formatted_zeroXingRate)

    parsed_beatTrack_diffs_mean = mean(formatted_beatTrack_diffs)
    parsed_beatTrack_diffs_variance = variance(formatted_beatTrack_diffs)
    parsed_beatTrack_length = formatted_beatTrack_length

    parsed_localPulse_mean = mean(formatted_localPulse)
    parsed_localPulse_variance = variance(formatted_localPulse)
    parsed_localPulseZeroless_mean = mean(formatted_localPulseZeroless)
    parsed_localPulseZeroless_variance = variance(formatted_localPulseZeroless)

    #set genre class data to numbers so that machine learning can parse
    parsed_tempoEstimate = formatted_tempoEstimate
    #default value=0 (blues) is temporary placeholder to deal with a bug where models being predicted need a value, this will not exist
    #in final version  
    current_song_genre_number = 0
    if current_song_genre == 'blues':
        current_song_genre_number = 0
    if current_song_genre == 'classical':
        current_song_genre_number = 1
    if current_song_genre == 'country':
        current_song_genre_number = 2
    if current_song_genre == 'disco':
        current_song_genre_number = 3
    if current_song_genre == 'hiphop':
        current_song_genre_number = 4
    if current_song_genre == 'jazz':
        current_song_genre_number = 5
    if current_song_genre == 'metal':
        current_song_genre_number = 6
    if current_song_genre == 'pop':
        current_song_genre_number = 7
    if current_song_genre == 'reggae':
        current_song_genre_number = 8
    if current_song_genre == 'rock':
        current_song_genre_number = 9
    

    parsed_genre = current_song_genre_number
    parsed_length = current_song_length
    parsed_fileName = current_song_filename

    #create song object with all of the characteristics needed
    analysis_target_song = Analysis_Target(
                                          parsed_genre,
                                          parsed_fileName,
                                          parsed_length,
                                          current_json_location,
                                          mfcc_available_var,
                                          rms_available_var,
                                          spectralCentroid_available_var,
                                          spectralBandwidth_available_var, 
                                          spectralContrast_available_var,
                                          spectralFlatness_available_var,
                                          spectralRolloff_available_var,
                                          polyFeatures_available_var,
                                          tonalCentroid_available_var,
                                          zeroXingRate_available_var,
                                          beatTrack_available_var,
                                          localPulse_available_var,
                                          tempoEstimate_available_var,
                                          parsed_mfcc1_mean,
                                          parsed_mfcc1_variance,
                                          parsed_mfcc2_mean,
                                          parsed_mfcc2_variance,
                                          parsed_mfcc3_mean,
                                          parsed_mfcc3_variance,
                                          parsed_mfcc4_mean,
                                          parsed_mfcc4_variance,
                                          parsed_mfcc5_mean,
                                          parsed_mfcc5_variance,
                                          parsed_mfcc6_mean,
                                          parsed_mfcc6_variance,
                                          parsed_mfcc7_mean,
                                          parsed_mfcc7_variance,
                                          parsed_mfcc8_mean,
                                          parsed_mfcc8_variance,
                                          parsed_mfcc9_mean,
                                          parsed_mfcc9_variance,
                                          parsed_mfcc10_mean,
                                          parsed_mfcc10_variance,
                                          parsed_mfcc11_mean,
                                          parsed_mfcc11_variance,
                                          parsed_mfcc12_mean,
                                          parsed_mfcc12_variance,
                                          parsed_mfcc13_mean,
                                          parsed_mfcc13_variance,
                                          parsed_mfcc14_mean,
                                          parsed_mfcc14_variance,
                                          parsed_mfcc15_mean,
                                          parsed_mfcc15_variance,
                                          parsed_mfcc16_mean,
                                          parsed_mfcc16_variance,
                                          parsed_mfcc17_mean,
                                          parsed_mfcc17_variance,
                                          parsed_mfcc18_mean,
                                          parsed_mfcc18_variance,
                                          parsed_mfcc19_mean,
                                          parsed_mfcc19_variance,
                                          parsed_mfcc20_mean,
                                          parsed_mfcc20_variance,
                                          parsed_rms_mean,
                                          parsed_rms_variance,
                                          parsed_spectralCentroid_mean,
                                          parsed_spectralCentroid_variance,
                                          parsed_spectralBandwidth_mean,
                                          parsed_spectralBandwidth_variance,
                                          parsed_spectralFlatness_mean,
                                          parsed_spectralFlatness_variance,
                                          parsed_spectralContrast1_mean,    
                                          parsed_spectralContrast1_variance,
                                          parsed_spectralContrast2_mean,
                                          parsed_spectralContrast2_variance,
                                          parsed_spectralContrast3_mean,
                                          parsed_spectralContrast3_variance,
                                          parsed_spectralContrast4_mean,
                                          parsed_spectralContrast4_variance,
                                          parsed_spectralContrast5_mean,
                                          parsed_spectralContrast5_variance,
                                          parsed_spectralContrast6_mean,
                                          parsed_spectralContrast6_variance,
                                          parsed_spectralContrast7_mean,
                                          parsed_spectralContrast7_variance,
                                          parsed_spectralRolloff_mean,
                                          parsed_spectralRolloff_variance,
                                          parsed_polyFeatures_mean,
                                          parsed_polyFeatures_variance,
                                          parsed_tonalCentroidD1_mean,
                                          parsed_tonalCentroidD1_variance,
                                          parsed_tonalCentroidD2_mean,
                                          parsed_tonalCentroidD2_variance,
                                          parsed_tonalCentroidD3_mean,
                                          parsed_tonalCentroidD3_variance,
                                          parsed_tonalCentroidD4_mean,
                                          parsed_tonalCentroidD4_variance,
                                          parsed_tonalCentroidD5_mean,
                                          parsed_tonalCentroidD5_variance,
                                          parsed_tonalCentroidD6_mean,
                                          parsed_tonalCentroidD6_variance,
                                          parsed_zeroXingRate_mean,
                                          parsed_zeroXingRate_variance,
                                          parsed_beatTrack_diffs_mean,
                                          parsed_beatTrack_diffs_variance,
                                          parsed_beatTrack_length,
                                          parsed_localPulse_mean,
                                          parsed_localPulse_variance,
                                          parsed_localPulseZeroless_mean,
                                          parsed_localPulseZeroless_variance,
                                          parsed_tempoEstimate
                                          )
    #add an array to the features array with the properties of the song
    features.append([
                     analysis_target_song.genre,
                     analysis_target_song.lengthInSamples,           
                     analysis_target_song.mfcc1_mean,
                     analysis_target_song.mfcc1_variance,
                     analysis_target_song.mfcc2_mean,
                     analysis_target_song.mfcc2_variance,
                     analysis_target_song.mfcc3_mean,
                     analysis_target_song.mfcc3_variance,
                     analysis_target_song.mfcc4_mean,
                     analysis_target_song.mfcc4_variance,
                     analysis_target_song.mfcc5_mean,
                     analysis_target_song.mfcc5_variance,
                     analysis_target_song.mfcc6_mean,
                     analysis_target_song.mfcc6_variance,
                     analysis_target_song.mfcc7_mean,
                     analysis_target_song.mfcc7_variance,
                     analysis_target_song.mfcc8_mean,
                     analysis_target_song.mfcc8_variance,
                     analysis_target_song.mfcc9_mean,
                     analysis_target_song.mfcc9_variance,
                     analysis_target_song.mfcc10_mean,
                     analysis_target_song.mfcc10_variance,
                     analysis_target_song.mfcc11_mean,
                     analysis_target_song.mfcc11_variance,
                     analysis_target_song.mfcc12_mean,
                     analysis_target_song.mfcc12_variance,
                     analysis_target_song.mfcc13_mean,
                     analysis_target_song.mfcc13_variance,
                     analysis_target_song.mfcc14_mean,
                     analysis_target_song.mfcc14_variance,
                     analysis_target_song.mfcc15_mean,
                     analysis_target_song.mfcc15_variance,
                     analysis_target_song.mfcc16_mean,
                     analysis_target_song.mfcc16_variance,
                     analysis_target_song.mfcc17_mean,
                     analysis_target_song.mfcc17_variance,
                     analysis_target_song.mfcc18_mean,
                     analysis_target_song.mfcc18_variance,
                     analysis_target_song.mfcc19_mean,
                     analysis_target_song.mfcc19_variance,
                     analysis_target_song.mfcc20_mean,
                     analysis_target_song.mfcc20_variance,
                     analysis_target_song.rms_mean,
                     analysis_target_song.rms_variance,
                     analysis_target_song.spectralCentroid_mean,
                     analysis_target_song.spectralCentroid_variance,
                     analysis_target_song.spectralBandwidth_mean,
                     analysis_target_song.spectralBandwidth_variance,
                     analysis_target_song.spectralFlatness_mean,
                     analysis_target_song.spectralFlatness_variance,
                     analysis_target_song.spectralContrast1_mean,
                     analysis_target_song.spectralContrast1_variance,
                     analysis_target_song.spectralContrast2_mean,
                     analysis_target_song.spectralContrast2_variance,
                     analysis_target_song.spectralContrast3_mean,
                     analysis_target_song.spectralContrast3_variance,
                     analysis_target_song.spectralContrast4_mean,
                     analysis_target_song.spectralContrast4_variance,
                     analysis_target_song.spectralContrast5_mean,
                     analysis_target_song.spectralContrast5_variance,
                     analysis_target_song.spectralContrast6_mean,
                     analysis_target_song.spectralContrast6_variance,
                     analysis_target_song.spectralContrast7_mean,
                     analysis_target_song.spectralContrast7_variance,
                     analysis_target_song.spectralRolloff_mean,
                     analysis_target_song.spectralRolloff_variance,
                     analysis_target_song.polyFeatures_mean,
                     analysis_target_song.polyFeatures_variance,
                     analysis_target_song.tonalCentroidD1_mean,
                     analysis_target_song.tonalCentroidD1_variance,
                     analysis_target_song.tonalCentroidD2_mean,
                     analysis_target_song.tonalCentroidD2_variance,
                     analysis_target_song.tonalCentroidD3_mean,
                     analysis_target_song.tonalCentroidD3_variance,
                     analysis_target_song.tonalCentroidD4_mean,
                     analysis_target_song.tonalCentroidD4_variance,
                     analysis_target_song.tonalCentroidD5_mean,
                     analysis_target_song.tonalCentroidD5_variance,
                     analysis_target_song.tonalCentroidD6_mean,
                     analysis_target_song.tonalCentroidD6_variance,
                     analysis_target_song.xeroXingRate_mean,
                     analysis_target_song.xeroXingRate_variance,
                     analysis_target_song.beatTrackAvgJumpDiffs_average,
                     analysis_target_song.beatTrackAvgJumpDiffs_variance,
                     analysis_target_song.beatTrackArrayLength,
                     analysis_target_song.localPulse_mean,
                     analysis_target_song.localPulse_variance,
                     analysis_target_song.localPulseZeroless_mean,
                     analysis_target_song.localPulseZeroless_variance,
                     analysis_target_song.tempoEstimate
                     ])
    print('Features successfully extracted from json format and added to the ML dataframe.\n')

#create loop to ensure that all data is extracted for the folder where the main training dataset is
while json_list_length_counter > 0:
    print('Now adding next set of data located at \'%s\' to machine learning dataframe. \nThis song is %s of %s to be analysed.' %(current_json_location, counter_secondary, json_input_list_length))
    add_for_analysis()
    json_list_length_counter -= 1
    counter_secondary += 1
    current_json_location = json_input_list[(json_input_list_length - counter_secondary)]

#put data into a dataframe for further processing
featuresdataframe = pd.DataFrame(features, columns=
                          [
                          "genre", 
                          "length_samples", 
                          "mfcc1_mean",
                          "mfcc1_variance",
                          "mfcc2_mean",
                          "mfcc2_variance",
                          "mfcc3_mean",
                          "mfcc3_variance",
                          "mfcc4_mean",
                          "mfcc4_variance",
                          "mfcc5_mean",
                          "mfcc5_variance",
                          "mfcc6_mean",
                          "mfcc6_variance",
                          "mfcc7_mean",
                          "mfcc7_variance",
                          "mfcc8_mean",
                          "mfcc8_variance",
                          "mfcc9_mean",
                          "mfcc9_variance",
                          "mfcc10_mean",
                          "mfcc10_variance",
                          "mfcc11_mean",
                          "mfcc11_variance",
                          "mfcc12_mean",
                          "mfcc12_variance",
                          "mfcc13_mean",
                          "mfcc13_variance",
                          "mfcc14_mean",
                          "mfcc14_variance",
                          "mfcc15_mean",
                          "mfcc15_variance",
                          "mfcc16_mean",
                          "mfcc16_variance",
                          "mfcc17_mean",
                          "mfcc17_variance",
                          "mfcc18_mean",
                          "mfcc18_variance",
                          "mfcc19_mean",
                          "mfcc19_variance",
                          "mfcc20_mean",
                          "mfcc20_variance",
                          "rms_mean",
                          "rms_variance",
                          "spectralCentroid_mean",
                          "spectralCentroid_variance",
                          "spectralBandwidth_mean",
                          "spectralBandwidth_variance",
                          "spectralFlatness_mean",
                          "spectralFlatness_variance",
                          "spectralContrast1_mean",
                          "spectralContrast1_variance",
                          "spectralContrast2_mean",
                          "spectralContrast2_variance",
                          "spectralContrast3_mean",
                          "spectralContrast3_variance",
                          "spectralContrast4_mean",
                          "spectralContrast4_variance",
                          "spectralContrast5_mean",
                          "spectralContrast5_variance",
                          "spectralContrast6_mean",
                          "spectralContrast6_variance",
                          "spectralContrast7_mean",
                          "spectralContrast7_variance",
                          "spectralRolloff_mean",
                          "spectralRolloff_variance",
                          "polyFeatures_mean",
                          "polyFeatures_variance",
                          "tonalCentroidD1_mean",
                          "tonalCentroidD1_variance",
                          "tonalCentroidD2_mean",
                          "tonalCentroidD2_variance",
                          "tonalCentroidD3_mean",
                          "tonalCentroidD3_variance",
                          "tonalCentroidD4_mean",
                          "tonalCentroidD4_variance",
                          "tonalCentroidD5_mean",
                          "tonalCentroidD5_variance",
                          "tonalCentroidD6_mean",
                          "tonalCentroidD6_variance",
                          "xeroXingRate_mean",
                          "xeroXingRate_variance",
                          "beatTrackAvgJumpDiffs_average",
                          "beatTrackAvgJumpDiffs_variance",
                          "beatTrackArrayLength",
                          "localPulse_mean",
                          "localPulse_variance",
                          "localPulseZeroless_mean",
                          "localPulseZeroless_variance",
                          "tempoEstimate"                   
                          ])
# Rename featuresdataframe to something shorter to work with and in case featuresdataframe (raw) is required later one
dataset = featuresdataframe

#get shape of dataset
print(dataset.shape)

#drop genre and length for the non-genre dataset (X)
X = dataset.drop(['genre', 'length_samples'], axis = 1)
dataset['genre'] = preprocessing.LabelEncoder().fit_transform(dataset['genre'])
#add scaler to normalise X set
scaler = preprocessing.MinMaxScaler()
#apply scaling to X
X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
y = dataset['genre']

# Actually split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42, stratify = y)

# Add class names for better interpretation (they had to be referred to as 0-10); 
# This should be implemented better for non GTZAN datasets
classes_names = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
#length = len(X_test[0])
#print(length)

#build neural network
def build_model(hp):
    #sequential means each layer happens one after another and layers can not access each other in a non-linear manner
    #it is the simplest to build
    model = Sequential()
    #make shape compatible with dataset
    model.add(layers.Flatten(input_shape = (88,)))
    #adding layers to the neural network
    model.add(layers.Dense(units = hp.Choice('dense_1', [256,512]), activation= "relu"))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Dense(units = hp.Choice('dense_2',[128,256]), activation= "relu"))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Dense(units = hp.Choice('dense_3',[64,128]), activation= "relu"))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Dense(units = hp.Choice('dense_4',[64,128]), activation= "relu"))
    #final layer has 10 outputs and a softmax activation to get prediction probability
    model.add(layers.Dense(10, activation='softmax'))
    #compile model with sparse categorical crossentropy, again to get prediction probability, and with sparse categorical crossentropy
    #instead of categorical_crossentropy because the categories are integer values instead of being encoded as binary
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    return model

#build a tuner to do a hyperparameter grid search to optimise the model
tuner = keras_tuner.RandomSearch(
                                hypermodel = build_model,
                                objective = "val_acc",
                                max_trials = 5000
                                )
#get summary of the search with the grid search
tuner.search_space_summary()

#run through 500 models (epochs) and test them, with different parameters
tuner.search(x = X_train, y = y_train, epochs=500, validation_data=(X_test, y_test))

#set a filepath for the best model to be saved at
model_output_filepath = 'iteration_best_model.hdf5'

#create a checkpoint function to be able to save best model
checkpoint = ModelCheckpoint(filepath = model_output_filepath, 
                             monitor='val_acc',
                             verbose=1, 
                             save_best_only = True,
                             mode='max')

callbacks = [checkpoint]

#get the top 3 models from the search for best models
top_models = tuner.get_best_models(num_models=10)

#grab the best of the top 3 (currently the other 2 are unused...)
best_model1 = top_models[0]

#console updates
print('5 best models have been identified. Continuing with the best model...')

best_models_history = best_model1.fit(x = X_train, y = y_train, validation_data = (X_test, y_test), epochs = 100, callbacks = callbacks)
print('The best model has been saved as %s' %model_output_filepath)

#load model (not necessary, but test of functionality for later work)
saved_model = load_model('iteration_best_model.hdf5', compile = True)

##option for using non-neural network models
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#lr =  LogisticRegression(solver='liblinear', multi_class='ovr')
#lr.fit(X_train, y_train)

#rf = RandomForestClassifier(random_state=0)
#rf.fit(X_train, y_train)

#dt = DecisionTreeClassifier(random_state=0)
#dt.fit(X_train, y_train)

#knn = KNeighborsClassifier(n_neighbors=8)
#knn.fit(X_train, y_train)

#lda = LinearDiscriminantAnalysis()
#lda.fit(X_train, y_train)

#gnb = GaussianNB()
#gnb.fit(X_train, y_train)

#svm = SVC(gamma = 'auto')
#svm.fit(X_train, y_train)

#saved_model = #change this to get data for different models

#evaluate the model
loss, acc = saved_model.evaluate(X_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#sklearn results evaluation tools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_pred = saved_model.predict(X_test)

#remove next line if analysing non neural network model
y_pred = np.argmax(y_pred, axis=1)

#sklearn results evaluation console report
print(classification_report(y_test, y_pred))

#create confusion matric
cm_final = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)

#put confusion matrix through seaborn to give make it prettier representation of data which prints it also
seaborn.heatmap(cm_final, annot=True, fmt='d');

#show in console the confusion matric for the loaded model
plt.show()

#Print classification report matrix (raw) to console
print(classification_report(y_test, y_pred))

class_predicted = saved_model.predict(X_test)
print(class_predicted)

#generate get max values for predictions
classes = np.argmax(class_predicted, axis = 1)
#prints an array of the predicted classes of the test set, somewhat unnecessary and meaningless, but show model has some distribution between them
print(classes)

#Code for getting weightings
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import cross_val_predict

#classifier = OneVsRestClassifier(
#    make_pipeline(RandomForestClassifier(random_state=42))
#)

#classifier.fit(X_train, y_train)
#y_train_pred = cross_val_predict(classifier, X_train, y_train, cv=3) 
#test0 = classifier.estimators_[0].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 0')
#print(test0)
#test1 = classifier.estimators_[1].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 1')
#print(test1)
#test2 = classifier.estimators_[2].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 2')
#print(test2)
#test3 = classifier.estimators_[3].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 3')
#print(test3)
#test4 = classifier.estimators_[4].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 4')
#print(test4)
#test5 = classifier.estimators_[5].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 5')
#print(test5)
#test6 = classifier.estimators_[6].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 6')
#print(test6)
#test7 = classifier.estimators_[7].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 7')
#print(test7)
#test8 = classifier.estimators_[8].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 8')
#print(test8)
#test9 = classifier.estimators_[9].named_steps['randomforestclassifier'].feature_importances_
#print('these are the feature importances for 9')
#print(test9)

history = saved_model.fit(X_train, y_train,validation_split = 0.3, epochs=50, batch_size=4)
neural_history = history.history['val_acc']

##BELOW CODE GETS AND PLOTS DISTRIBUTIONS FOR 50 TESTS WITH DIFFERENT ALGORITHMS
#from pandas.plotting import scatter_matrix
#from matplotlib import pyplot
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#multitestmodels = []
#multitestmodels.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#multitestmodels.append(('LDA', LinearDiscriminantAnalysis()))
#multitestmodels.append(('KNN', KNeighborsClassifier()))
#multitestmodels.append(('CART', DecisionTreeClassifier()))
#multitestmodels.append(('NB', GaussianNB()))
#multitestmodels.append(('SVM', SVC(gamma='auto')))
#multitestmodels.append(('RF', RandomForestClassifier()))
## evaluate each model in turn
#multitestresults = []
#multitestnames = []
#for name, model in multitestmodels:
#	kfold = StratifiedKFold(n_splits=50, random_state=1, shuffle=True)
#	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#	multitestresults.append(cv_results)
#	multitestnames.append(name)6
#	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#multitestresults.append(neural_history)
#multitestnames.append('NN')

#pyplot.boxplot(multitestresults, labels=multitestnames)
#pyplot.title('Prediction Model Algorithm Validation Accuracy Comparison')
#pyplot.show()
#set the previous function to process data to now target the song to be predicted6
current_json_location = predict_this_filepath
#clear the previous list of features used for training
features.clear()
#add song to be predicted for analysis
add_for_analysis()

#current_json_location = "D:\\Coding\\AMGR\ML\\MachineLearningV0.1 - Copy\\predict_these\\jazz.11.json"
#add_for_analysis()
#current_json_location = "D:\\Coding\\AMGR\ML\\MachineLearningV0.1 - Copy\\predict_these\\jazz.12.json"
#add_for_analysis()
#current_json_location = "D:\\Coding\\AMGR\ML\\MachineLearningV0.1 - Copy\\predict_these\\jazz.13.json"
#add_for_analysis()
#current_json_location = "D:\\Coding\\AMGR\ML\\MachineLearningV0.1 - Copy\\predict_these\\jazz.14.json"
#add_for_analysis()
#current_json_location = "D:\\Coding\\AMGR\ML\\MachineLearningV0.1 - Copy\\predict_these\\reggae.11.json"
#add_for_analysis()
#current_json_location = "D:\\Coding\\AMGR\ML\\MachineLearningV0.1 - Copy\\predict_these\\reggae.12.json"
#add_for_analysis()
#current_json_location = "D:\\Coding\\AMGR\ML\\MachineLearningV0.1 - Copy\\predict_these\\reggae.13.json"
#add_for_analysis()
#current_json_location = "D:\\Coding\\AMGR\ML\\MachineLearningV0.1 - Copy\\predict_these\\reggae.14.json"
#add_for_analysis()
#put it in a dataframe as before
predict_me = pd.DataFrame(features, columns=
                          [
                          "genre", 
                          "length_samples", 
                          "mfcc1_mean",
                          "mfcc1_variance",
                          "mfcc2_mean",
                          "mfcc2_variance",
                          "mfcc3_mean",
                          "mfcc3_variance",
                          "mfcc4_mean",
                          "mfcc4_variance",
                          "mfcc5_mean",
                          "mfcc5_variance",
                          "mfcc6_mean",
                          "mfcc6_variance",
                          "mfcc7_mean",
                          "mfcc7_variance",
                          "mfcc8_mean",
                          "mfcc8_variance",
                          "mfcc9_mean",
                          "mfcc9_variance",
                          "mfcc10_mean",
                          "mfcc10_variance",
                          "mfcc11_mean",
                          "mfcc11_variance",
                          "mfcc12_mean",
                          "mfcc12_variance",
                          "mfcc13_mean",
                          "mfcc13_variance",
                          "mfcc14_mean",
                          "mfcc14_variance",
                          "mfcc15_mean",
                          "mfcc15_variance",
                          "mfcc16_mean",
                          "mfcc16_variance",
                          "mfcc17_mean",
                          "mfcc17_variance",
                          "mfcc18_mean",
                          "mfcc18_variance",
                          "mfcc19_mean",
                          "mfcc19_variance",
                          "mfcc20_mean",
                          "mfcc20_variance",
                          "rms_mean",
                          "rms_variance",
                          "spectralCentroid_mean",
                          "spectralCentroid_variance",
                          "spectralBandwidth_mean",
                          "spectralBandwidth_variance",
                          "spectralFlatness_mean",
                          "spectralFlatness_variance",
                          "spectralContrast1_mean",
                          "spectralContrast1_variance",
                          "spectralContrast2_mean",
                          "spectralContrast2_variance",
                          "spectralContrast3_mean",
                          "spectralContrast3_variance",
                          "spectralContrast4_mean",
                          "spectralContrast4_variance",
                          "spectralContrast5_mean",
                          "spectralContrast5_variance",
                          "spectralContrast6_mean",
                          "spectralContrast6_variance",
                          "spectralContrast7_mean",
                          "spectralContrast7_variance",
                          "spectralRolloff_mean",
                          "spectralRolloff_variance",
                          "polyFeatures_mean",
                          "polyFeatures_variance",
                          "tonalCentroidD1_mean",
                          "tonalCentroidD1_variance",
                          "tonalCentroidD2_mean",
                          "tonalCentroidD2_variance",
                          "tonalCentroidD3_mean",
                          "tonalCentroidD3_variance",
                          "tonalCentroidD4_mean",
                          "tonalCentroidD4_variance",
                          "tonalCentroidD5_mean",
                          "tonalCentroidD5_variance",
                          "tonalCentroidD6_mean",
                          "tonalCentroidD6_variance",
                          "xeroXingRate_mean",
                          "xeroXingRate_variance",
                          "beatTrackAvgJumpDiffs_average",
                          "beatTrackAvgJumpDiffs_variance",
                          "beatTrackArrayLength",
                          "localPulse_mean",
                          "localPulse_variance",
                          "localPulseZeroless_mean",
                          "localPulseZeroless_variance",
                          "tempoEstimate"                   
                          ])
#get new X value(s) from the song to be predicted's features and drop genre and lgneth
X2 = predict_me.drop(['genre', 'length_samples'], axis = 1)
#rescale the song to be predicted according to the scaling of the original dataset in order for it to be able to be properly interpreted
X3 = pd.DataFrame(scaler.transform(X2), columns = X2.columns)
#get a prediction for the model
finale = saved_model.predict(X3)

##next couple lines are alternatives and useful for getting single predictions but need fixed
#finale2 = np.argmax(saved_model.predict(scaled_predict_me), axis=-1)
#output_prediction = np.argmax(output_prediction, axis=1)    

#print final prediction
print(finale)




#console update
print('Script complete. Any messages following this are GPU information or Keras ending type errors that will not effect the model.')