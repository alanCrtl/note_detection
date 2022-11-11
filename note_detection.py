import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
from numpy.fft import fft
from numpy.fft import fftfreq
grad = np.gradient
abs = np.abs
sign = np.sign

"""
PART 1:
------ 
GOAL: audio of a note or a chord, use fourier transform to recognise notes played
- draw graph
- figure out how to draw amplitude spectrum
- save frequencies based on highest amplitudes with a threshold rule
- distance algorithm to figure out note correspondance

PART 2:
------
0.3 - analyse complexe différente: juste avec de pics de volume (vol[i] - vol[i-1])

GOAL: chunk analysis
1 - noise reduction gate, high pass filter
2 - catch the start of notes/chords by checking volume increase (derivative of abs(data))
find local minima of gradient(movingavg(left)) correspond to note playing
from this analysis only consider wide enough rectangles and volume past a certain level

3 - sample the audio at the start of notes/chords and before the next one
4 - chord analysis on the sample
5 - result: list of single notes or chords played through time
6 - try : input: (Notes, Times) output: Sheet Music
5.9 - tempo (check interval between note starts)

https://python-course.eu/applications-python/musical-scores-with-python.php
https://www.google.com/search?q=python+draw+on+sheet+music&oq=python+draw+on+sheet+music+&aqs=chrome..69i57j33i160l5j33i22i29i30j33i15i22i29i30.6604j0j7&sourceid=chrome&ie=UTF-8

"""

notes = [
    ['C0', 16.35, []],
    ['C#0/Db0', 17.32, []],
    ['D0', 18.35, []],
    ['D#0/Eb0', 19.45, []],
    ['E0', 20.60, []],
    ['F0', 21.83, []],
    ['F#0/Gb0', 23.12, []],
    ['G0', 24.50, []],
    ['G#0/Ab0', 25.96, []],
    ['A0', 27.50, []], # start of piano
    ['A#0/Bb0', 29.14, []],
    ['B0', 30.87, []],
    ['C1', 32.70, []],
    ['C#1/Db1', 34.65, []],
    ['D1', 36.71, []],
    ['D#1/Eb1', 38.89, []],
    ['E1', 41.20, []],
    ['F1', 43.65, []],
    ['F#1/Gb1', 46.25, []],
    ['G1', 49.00, []],
    ['G#1/Ab1', 51.91, []],
    ['A1', 55.00, []],
    ['A#1/Bb1', 58.27, []],
    ['B1', 61.74, []],
    ['C2', 65.41, []],
    ['C#2/Db2', 69.30, []],
    ['D2', 73.42, []],
    ['D#2/Eb2', 77.78, []],
    ['E2', 82.41, []],
    ['F2', 87.31, []],
    ['F#2/Gb2', 92.50, []],
    ['G2', 98.00, []],
    ['G#2/Ab2', 103.83, []],
    ['A2', 110.00, []],
    ['A#2/Bb2', 116.54, []],
    ['B2', 123.47, []],
    ['C3', 130.81, []],
    ['C#3/Db3', 138.59, []],
    ['D3', 146.83, []],
    ['D#3/Eb3', 155.56, []],
    ['E3', 164.81, []],
    ['F3', 174.61, []],
    ['F#3/Gb3', 185.00, []],
    ['G3', 196.00, []],
    ['G#3/Ab3', 207.65, []],
    ['A3', 220.00, []],
    ['A#3/Bb3', 233.08, []],
    ['B3', 246.94, []],
    ['C4', 261.63, []],
    ['C#4/Db4', 277.18, []],
    ['D4', 293.66, []],
    ['D#4/Eb4', 311.13, []],
    ['E4', 329.63, []],
    ['F4', 349.23, []],
    ['F#4/Gb4', 369.99, []],
    ['G4', 392.00, []],
    ['G#4/Ab4', 415.30, []],
    ['A4', 440.00, []],
    ['A#4/Bb4', 466.16, []],
    ['B4', 493.88, []],
    ['C5', 523.25, []],
    ['C#5/Db5', 554.37, []],
    ['D5', 587.33, []],
    ['D#5/Eb5', 622.25, []],
    ['E5', 659.25, []],
    ['F5', 698.46, []],
    ['F#5/Gb5', 739.99, []],
    ['G5', 783.99, []],
    ['G#5/Ab5', 830.61, []],
    ['A5', 880.00, []],
    ['A#5/Bb5', 932.33, []],
    ['B5', 987.77, []],
    ['C6', 1046.50, []],
    ['C#6/Db6', 1108.73, []],
    ['D6', 1174.66, []],
    ['D#6/Eb6', 1244.51	, []],
    ['E6', 1318.51, []],
    ['F6', 1396.91, []],
    ['F#6/Gb6', 1479.98, []],
    ['G6', 1567.98, []],
    ['G#6/Ab6', 1661.22, []],
    ['A6', 1760.00	, []],
    ['A#6/Bb6', 1864.66, []],
    ['B6', 1975.53	, []],
    ['C7', 2093.00, []],
    ['C#7/Db7', 2217.46, []],
    ['D7', 2349.32, []],
    ['D#7/Eb7', 2489.02, []],
    ['E7', 2637.02, []],
    ['F7', 2793.83, []],
    ['F#7/Gb7 ', 2959.96, []],
    ['G7', 3135.96, []],
    ['G#7/Ab7', 3322.44, []],
    ['A7', 3520.00, []],
    ['A#7/Bb7', 3729.31, []],
    ['B7', 3951.07, []],
    ['C8', 4186.01, []], # end of piano
    ['C#8/Db8', 4434.92, []],
    ['D8', 4698.63, []],
    ['D#8/Eb8', 4978.03, []],
    ['E8', 5274.04, []],
    ['F8', 5587.65, []],
    ['F#8/Gb8', 5919.91, []],
    ['G8', 6271.93, []],
    ['G#8/Ab8', 6644.88	, []],
    ['A8', 7040.00, []],
    ['A#8/Bb8', 7458.62, []],
    ['B8', 7902.13, []],
]

# ====================== start of definitions ====================== 
# returns the frequencies that have amplitude above threshold
# THRESHOLD : volume threshold
# amp : array of the amplitude
# freq : array of the frequencies
# FJUMP : number of iteration to skip to avoid capturing same frequency multiple times
def freq_of_good_amplitude(THRESHOLD, amp, freq, FJUMP):
	L = []
	i = 0
	# if amplitude > threshold => save frequency (jump the peak)
	while i < len(amp):
		if amp[i] > THRESHOLD:
			L.append(freq[i])	
			i += FJUMP # skip a peak of amplitude
		i += 1

	return(L)

# returns the note names corresponding to the frequencies in input
# freq : array of frequencies
def chord_recognition(freq):
	print(f'{len(freq)} frequencies to be matched with notes\n')
	if len(freq)==0:
		return

	Lnotes = []
	Lnotesnames = []
	redundancy = 0 
	# for each frequency
	for j in range(len(freq)):
		print(f'Searching for note corresponding to frequency: {freq[j]:.2f}')
		# initialise to note C0
		smallestdist = np.abs(notes[0][1] - freq[j])
		smallestdist_indx = 0

		# simply go through the note frequencies and store the lowest distance and 
		# the corresponding index
		for i in range(len(notes)):
			dist = np.abs(notes[i][1] - freq[j])
			if dist < smallestdist:
				smallestdist = dist
				smallestdist_indx = i

		Lnotes.append(smallestdist_indx)
		# add the note to the array if it isn't there yet
		if notes[smallestdist_indx][0] not in Lnotesnames:
			Lnotesnames.append(notes[smallestdist_indx][0])
		# otherwise increment redundancy, goal is redundancy=0
		else:
			redundancy += 1
			
	redundancystr = f'redundancy={redundancy}'
	
	# printing results
	for k in range(len(freq)):	
		print(f'\nnote recognised: {notes[Lnotes[k]][0]} (freq: {notes[Lnotes[k]][1]:.2f} Hz)\nis the closest note to freq: {freq[k]:.2f} Hz,\nwith a distance of {np.abs(notes[Lnotes[k]][1] - freq[k]):.2f}')
	print(f'\n{Lnotesnames} {redundancystr}\n{len(Lnotesnames)} notes')

	return(Lnotesnames)

# creates the moving average of the past {order} datapoints
# data : signal array
# order : number of datapoints you look back to make an average (2 means avg between current and 1 back)
def moving_average(data, order=2):
	L = []
	length = len(data)

	# we can't make an average on the first ones < order
	# so we set them equal to the avg of themselves
	sum0=0
	for k in range(order-1):
		sum0 += data[k]/(order-1)
	for w in range(order-1):
		L.append(sum0)

	# calculate the full sum once
	sum = 0
	for j in range(order):
			sum += data[j]
	sum = sum/order
	L.append(sum)

	# then smartly iterate, to avoid redundant operations
	for i in range(order, length):
		sum = sum + (data[i]/order) - data[i-order]/order
		L.append(sum)

	return(L)

# looking around (left and right) if values are mostly pos or neg
# if the average is pos then add 1 to the list, if neg add 0
# data : signal array
# order : is length at which you look around left and right (=1 means 1 right, 1 left)
# scaler : is to scale values for nice display on graph
def moving_signed_indicator(data, order=1, scaler=1):
	L = []
	N = len(data)
	order = 2 * order # I don't want to think about odd numbers

	# fill up the beginning
	sum0=0
	for k in range((order//2)):
		sum0 += data[k]
	for w in range((order//2)):
		L.append(max(0, sign(sum0)*scaler))

	# calculate the full sum once
	sum = 0
	for j in range((order//2)):
			# print(f'data[j]:{data[j]}, data[order//2+1+j]:{data[order//2+1+j]}')
			sum += (data[j] + data[order//2+1+j])
	L.append(max(0, sign(sum)*scaler) )
		
	# then smartly iterate, to avoid redundant operations
	for i in range(order//2+1, N-order//2):
		sum += (data[i-1] + data[i+order//2]) - (data[i] + data[i-(order//2)-1])
		L.append(max(0, sign(sum)*scaler) )

	# fill up the end
	sum0=0
	for k2 in range(N-1,N-1-(order//2),-1):
		sum0 += data[k2]
	for w2 in range((order//2)):
		L.append(max(0, sign(sum0)*scaler))

	return(L)

# set volume to 0 if it's below a percentage of maxvolume
# thold : between 0 and 1 makes the most sense
# data : signal array
def remove_lowvolume(data, thold=0.1):
	maxvol = np.max(data)
	for i in range(len(data)):
		if abs(data[i]) < thold * maxvol:
			data[i] = 0
	return(data)

# a sample is a start and an end where a note is predicted to 
# have been played
def sample_around_note(msi, dt):
	samples_times = []
	start = -1
	end = -1
	for i in range(len(msi)):
		# rectangle search
		if msi[i] == 1 and start == -1:
			start = i * dt # start time		
		if start != -1 and end == -1 and msi[i] == 0:
			end = i * dt # end time
			duration = (end - start) # duration in s
			samples_times.append([start,end,duration])
			start = -1
			end = -1

	# removing samples too short in duration
	toPop = []
	for j in range(len(samples_times)):
		# criteria on duration in ms
		if samples_times[j][2] * 1000 <= 20: 
			toPop.append(samples_times[j])
	truesamples_times = [s for s in samples_times if s not in toPop ]

	# TODO: remove rectangle corresponding to low volume
	# printing press
	for k in range(len(truesamples_times)):
		print(f'duration of bloc {k}: {"{:.2f}".format(truesamples_times[k][2] * 1000)} ms')

	return(truesamples_times)

# from signal sample return the chords/notes found
def complex_sample_analysis(data, sample_times , dt, THRESHOLDMULT, FJUMP):
	song = []
	fig, plot = plt.subplots(len(sample_times))

	# for every sample of audio
	for v in range(len(sample_times)):
		duration = sample_times[v][2] # duration of sample in sec
		N = int(duration // dt) # number of datapoints corresponding to the sample
		start = sample_times[v][0] # starting timestamp in s
		start_point = int(start // dt) # corresponding start point in the data
		sample = data[start_point: start_point + N] # sampling the actual data

		# filter out low frequencies
		b, a = signal.butter(5, 50*dt, 'hp')
		sample = signal.filtfilt(b, a, sample) 

		# fourier
		fourier = fft(sample)
		fourier_mag = np.abs(fourier)/N 

		# puella magi madoka magica
		# fstep = 1/T # 1/N*dt comme dans np.fft.fftfreq
		# freqscale = (N-1)*fstep
		# f = np.linspace(0, freqscale, N)

		# alternative
		f = fftfreq(N, dt) 
		
		# adjustments for plotting (https://youtu.be/O0Y8FChBaFU)
		f_plot = f[0:int(N/2)]
		fourier_mag_plot = 2 * fourier_mag[0:int(N/2)]
		fourier_mag_plot[0] = fourier_mag_plot[0] / 2

		# computing the amplitude threshold
		THRESHOLD = np.max(fourier_mag_plot) * THRESHOLDMULT
		print(f'\n\n===================\nTHRESHOLD = {THRESHOLD}')

		# et pouf ça marche
		plot[v].plot(f_plot, fourier_mag_plot)
		plot[v].set_title("Magnitude Spectrum via Fourier")
		plot[v].axhline(THRESHOLD, color='red')
		plot[v].set_xlim(left=-1000,right=5000)

		# frequencies that have amplitude above threshold are sent to be identified
		freqs_of_good_amp = freq_of_good_amplitude(THRESHOLD, fourier_mag_plot, f_plot, FJUMP)  
		Lnotesnames = chord_recognition(freqs_of_good_amp) # figure out closest notes
		song.append(Lnotesnames)

	return(song)

# analysis on the whole file
def simple_sample_analysis(data, N, dt, THRESHOLDMULT, FJUMP):
	song = []

	# fourier
	fourier = fft(data)
	fourier_mag = np.abs(fourier)/N 

	# puella magi madoka magica
	# fstep = 1/T # 1/N*dt comme dans np.fft.fftfreq
	# freqscale = (N-1)*fstep
	# f = np.linspace(0, freqscale, N)

	# alternative
	f = fftfreq(N, dt) 
	
	# adjustments for plotting (https://youtu.be/O0Y8FChBaFU)
	f_plot = f[0:int(N/2)]
	fourier_mag_plot = 2 * fourier_mag[0:int(N/2)]
	fourier_mag_plot[0] = fourier_mag_plot[0] / 2

	# computing the amplitude threshold
	THRESHOLD = np.max(fourier_mag_plot) * THRESHOLDMULT
	print(f'------\nTHRESHOLD = {THRESHOLD}')

	# et pouf ça marche
	plt.figure()
	plt.plot(f_plot, fourier_mag_plot)
	plt.title("Magnitude Spectrum via Fourier")
	plt.axhline(THRESHOLD, color='red')
	plt.xlim(left=-1000,right=5000)

	# expectation for 4chords.wav file
	# expected_notes = [293.66, 130.81, 196.00, 261.63, 440.00, 369.99, 98.00, 164.81,  246.94, 329.63]
	# for ex in range(len(expected_notes)):
	# 	plt.axvline(expected_notes[ex], color='green', alpha=0.3)
	
	# frequencies that have amplitude above threshold are sent to be identified
	freqs_of_good_amp = freq_of_good_amplitude(THRESHOLD, fourier_mag_plot, f_plot, FJUMP)  
	Lnotesnames = chord_recognition(freqs_of_good_amp) # figure out closest notes
	song.append(Lnotesnames)

	return(song)

# ====================== main ====================== 
# THRESHOLDMULT : multiplier for the amplitude threshold of note recognition
# FJUMP : to skip peaks corresponding to a note; to avoid redundancy.
# GRAPHS : allows for graphs to be displayed
# SA : stands for 'simple analysis' meaning you fourier the all audio at once
def main(filename, GRAPHS=0, THRESHOLDMULT=0.5, FJUMP=0, CA=0):
	# preparing the data
	# =============================================================
	# data: 2D ndarray of audio data, fs: sample frequency
	data, fs = sf.read(filename, dtype='float32')

	# extract left and right audio channel, we'll use only left
	if len(np.shape(data))==2:
		left = data[:,0]
		right = data[:,1]
	# incase there is only 1 channel of audio
	else:
		left = data
		right = data

	left = left / np.max(left) # scaling volume
	left = remove_lowvolume(left, 0.2) # low volume removal

	N = len(data) # number of datapoints
	dt = 1/fs # datapoints time interval
	T = N*dt # datapoints * time interval = Length of Audio 
	t = np.linspace(0, T, N) # time vector	

	# plotting the signal
	# =============================================================
	fig, (g0, g1) = plt.subplots(2,1)
	g0.set_title("Signal on left channel")
	g0.plot(t, left)

	g1.set_title("Signal on right channel")
	g1.plot(t, right)

	# complex signal analysis; per chunk analysis of the signal
	# =============================================================
	# goal is to catch the local minimas of the moving average of the gradient
	# which corresponds to a peak in volume, so we know a note/chord is played
	# around that area
	# TODO: maybe I should just look at volume peaks...
	if CA:
		MAORDER=6000 # moving average order
		MSIORDER=3000 # moving signed indicator order
		# gradient of signal
		figg, (a0, a1) = plt.subplots(2, 1)
		a0.set_title("Signal on left channel in absolute value")
		a0.set_xlabel("time (sec)")
		a0.plot(t, abs(left))

		# yes
		ma = moving_average(abs(left), MAORDER)
		gradma = grad(ma)
		mamsi = moving_signed_indicator(gradma, MSIORDER, np.max(left))
		mamsi2 = moving_signed_indicator(gradma, MSIORDER, np.max(gradma))
		
		# plots
		a0.plot(t, ma, alpha=0.5, color='red')
		a0.plot(t, mamsi, alpha=0.4, color='green')
		# size of the window of search for the moving signed indicator
		a0.axvline(T//2 + 2 * MSIORDER * dt, alpha=0.4, color='orange') 
		a0.axvline(T//2 - 2 * MSIORDER * dt, alpha=0.4, color='orange') 

		# plots
		a1.set_title("gradient of ma(abs(left))")
		a1.plot(t, gradma)
		a1.plot(t, mamsi2, alpha=0.4, color='green')
		# size of the window of search for the moving signed indicator
		a1.axvline(T//2 + 2 * MSIORDER * dt, alpha=0.4, color='orange') 
		a1.axvline(T//2 - 2 * MSIORDER * dt, alpha=0.4, color='orange') 

	# Start of note/chord analysis for each note/chord detected
	# =============================================================
	# tries to cut the audio into sample where each sample corresponds to a
	# single note/chord being played and then do the simple analysis on these
	# samples
	if CA:
		samples_times = sample_around_note(mamsi, dt)
		song = complex_sample_analysis(left, samples_times, dt, THRESHOLDMULT, FJUMP)
	# simple analysis of the entire audio
	else:
		song = simple_sample_analysis(left, N, dt, THRESHOLDMULT, FJUMP)


	print(f"------\nsong : {song}")
	print(f'------\n {(time.time() - start_time):.2f} sec')

	if GRAPHS==1:
		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("filename", help="wav file to do note detection on")
	parser.add_argument("-g", "--graphs", metavar="n", help="show graphs (default=0) (int)", type=int, default=0)
	parser.add_argument("-tmult", "--thresholdMult", metavar="n", help="multiplier to the average amplitude that will establish the threshold to be considered a note; is percentage of max amplitude (default=0.5) (float)", type=float, default=0.5)
	parser.add_argument("-fj", "--freqjump", metavar="n", help="frequency jumps to be considered same frequency, so as not to pick every frequency around a peak (default=10) (int)", type=int, default=10)
	parser.add_argument("-ca","--complexanalysis", metavar="n", help="between multiple chord analysis or single note/chord analysis (default=0) (int)", type=int, default=0)

	args = parser.parse_args()
	start_time = time.time()

	main(args.filename, args.graphs, args.thresholdMult, args.freqjump, args.complexanalysis)

