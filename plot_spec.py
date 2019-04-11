#!/usr/bin/env python
# coding: utf-8


from librosa import load
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import scipy as scipy
from scipy.signal import lfilter
import IPython.display as ipd
from math import pi
from math import tan
from math import pow
import seaborn as sns
from numpy import sin, cos, pi
import sys
import csv

from cycler import cycler

fs = 44100
# NFFT=0
# integ= 'slow'
#
#
# if integ == 'slow':
#     NFFT= 65436
# elif integ == 'fast':
#     NFFT= 8192





def plot_spectrogram(the_file, durata, P, integ):

    plt.figure(figsize=(20, 5))
    fig, ax = plt.subplots(figsize=(20, 5))

    NFFT=8192


    fft_size = NFFT
    sr=44100

    frame = NFFT/4


    ax = plt.gca()

    the_transform_mag = np.abs(librosa.core.stft(the_file, n_fft=fft_size, hop_length=None, win_length=None, window='hann'))
    librosa.display.specshow(librosa.amplitude_to_db(the_transform_mag,ref=-90),cmap='gray_r',sr=44100, y_axis='log',x_axis='frames')
    ax.xaxis.set_major_locator(plt.MultipleLocator(sr/frame*10))

    def format_func(value, tick_number):



        return int((value * frame)/fs)
        # return int(value/fs)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))


    plt.xlabel('Seconds')

    plt.title(P +'_'+ str(NFFT))

    # print("hey you")

    fig.tight_layout()


    plt.savefig(P+'_'  + str(NFFT)+'_'+ str(durata) +'.eps', dpi=150)


    # plt.show()



def main(files):

    try:
        for f in files:



            # fileloc =   '/Users/amilo/Desktop/karl/participants/251119_P1/yourfolder/P01_D-mono_A.wav'
            # filebase = fileloc[:-10]
            print('executing on: ', f)

            filebase = f[:-4]
            print('root is = ', filebase)

            mono, fs = load(f, sr=44100, mono=True)
            # left, fs = load(filebase+'left_A.wav', sr=44100, mono=True)
            # righ, fs = load(filebase+'righ_A.wav', sr=44100, mono=True)
            # oops, fs = load(filebase+'oops_A.wav', sr=44100, mono=True)

            durata = len(mono)-1
            # durata = fs*9*60


            # partic = 'P01_D-mono_A.wav'
            partic = f[-16:-4]


            print(partic)

            # array1= sum_bin_energy(normalize(magnitude(mono[0:durata])))
            # array2= sum_bin_energy(normalize(magnitude(left[0:durata])))
            # array3= sum_bin_energy(normalize(magnitude(righ[0:durata])))
            # array4= sum_bin_energy(normalize(magnitude(oops[0:durata])))

            # plotlines1(array2, durata, partic, integ, 'm')

            # plotlines3(array2, array3, array4, durata, partic, integ, 's')
            plot_spectrogram(mono, durata, partic, 'fast')


        return 0
    except:
        return 1
    # sys.exit()

if __name__ == "__main__":
    files = sys.argv[1:] # slices off the first argument (executable itself)
    exit (main(files))
    # main(files)
    # print(files)



