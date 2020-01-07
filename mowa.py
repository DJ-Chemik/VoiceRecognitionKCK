import numpy as np
import numpy.fft as fft
import sys
import librosa
import pandas
from scipy.signal.windows import blackmanharris

FILE_TO_ANALYZE = sys.argv[1]
wavSample = 44100
file, sample_rate = librosa.load(FILE_TO_ANALYZE)#, sr=wavSample)

partSize = 4096*4
parts = []
for i in range(0, len(file), partSize):
    parts.append(file[i:i + partSize])
for part in parts:
    windowed = part * blackmanharris(len(part))
    part = windowed

localFreqs = []

for part in parts:
    # Spectrum from FFT
    fftPart = fft.fft(part, partSize)
    fftPart = fftPart[:len(fftPart) // 2]
    fftPart = np.abs(fftPart)

    # Harmonic Product Spectrum
    stepsNumber = 5
    stepSize = len(fftPart) // stepsNumber
    hpsResult = fftPart[:stepSize]
    for i in range(1, stepsNumber):
        hpsResult = hpsResult * fftPart[::i][stepSize]

    #Basic Frequency
    minimum=0

    start = int((minimum/sample_rate) * partSize)
    maximum = start
    for i in range(start+1, len(hpsResult)):
        if hpsResult[i] > hpsResult[maximum]:
            maximum=i
    localFreqs.append((maximum / partSize) * sample_rate)


fr = np.median(localFreqs)
if fr < 170:
    print("M")
else:
    print("K")