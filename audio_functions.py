
# imports
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib.ticker import MaxNLocator
import numpy as np

from pydub import AudioSegment
import librosa
import librosa.display

import requests
from io import BytesIO


from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis
from scipy.stats import entropy

# ===========================
## Functions


#----------------------------------------------------------------------
def signal_remove_blank_space(y,sr,  y_crop_initial_buffer_time=0.25):
  # finds onset of sound and crops the file prior to that with a buffer

    # Get real onset
    onset_frames_real = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        backtrack=False,  # Don't backtrack to earlier points, focus on the main onset
        pre_max=20,       # Spread out local maxima search, increase to focus on larger onsets
        post_max=20,      # Spread out the local maxima window
        pre_avg=100,      # Smoothing window to ignore smaller onsets
        post_avg=100,     # Same for post average
        delta=0.5         # Larger threshold for detecting stronger onsets
    )

    onset_time_real = librosa.frames_to_time(onset_frames_real, sr=sr)

    # If onset_time_real is an array, ensure it's a scalar by extracting the first element
    onset_time_real = onset_time_real[0]

    onset_frames_all = librosa.onset.onset_detect(y=y, sr=sr,) # ie will include small blips in the noise
    onset_times_all = librosa.frames_to_time(onset_frames_all, sr=sr)

    # Find the onset times that are greater than the target onset value
    onsets_after = onset_times_all[onset_times_all > onset_time_real]

    # Get the smallest onset time after the target onset
    if len(onsets_after) > 0:
        next_onset_after_real = np.min(onsets_after)
    else:
      next_onset_after_real = len(y) / sr # ie the original duration

    # Convert time to sample indices
    start_sample = int((onset_time_real-y_crop_initial_buffer_time) * sr)
    end_sample = int(next_onset_after_real * sr)

    truncated_y = y[start_sample:end_sample]

    return truncated_y

#----------------------------------------------------------------------
def create_wav_from_m4a(m4a_filename_no_ext, folder_path):

   # creates a wav file from an m4a audio file of the same name
   #TODO make generic to multiple formats and accept file name with file extension

    # Change directory to the desired folder
    os.chdir(folder_path)

    # Create full m4a path with extension
    m4a_file_path = os.path.join(folder_path, f"{m4a_filename_no_ext}.m4a")

    # load m4a audio
    audio = AudioSegment.from_file(m4a_file_path, format="m4a")

    # Create full wav path with extension from just the name
    wav_file_path = os.path.join(folder_path, f"{m4a_filename_no_ext}.wav")

    # Convert .m4a to .wav, check if doesn't already exists
    if not os.path.isfile(wav_file_path):

        audio.export(wav_file_path, format="wav")
        print(f"New wav file saved to: {wav_file_path}")

#----------------------------------------------------------------------
def calc_attack_time(y, sr):
    # Attack time is the time to reach peak amplitude
    envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))  # Use onset strength as an envelope
    peak_time = np.argmax(envelope)  # Find the peak index
    attack_time = peak_time / sr  # Convert index to time

    return attack_time*1000 # ms

#----------------------------------------------------------------------
# Generate fft
def generate_fft(y, sr):
    # generate fft
    N = len(y)
    T = 1.0 / sr
    yf = np.fft.fft(y)
    #The frequency axis xf is generated using np.linspace to go from 0 Hz to the Nyquist frequency (sr / 2),
    #which corresponds to the highest frequency that can be represented in the FFT result.
    #The range is divided into N//2 values because the FFT result is symmetric, so we only need to consider the first half (positive frequencies).
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    return xf, yf

#----------------------------------------------------------------------
def normalise_fft(yf):
    # Normalise y-frequency
    N = len(yf)
    # Get the magnitude of the FFT - magnitude of each frequency component given by the absolute value of the complex number
    #Since the FFT result is symmetric, we take only the first half (yf[:N//2]), corresponding to positive frequencies.
    yf_magnitude = np.abs(yf[:N//2])

    # Normalize magnitudes to percentages
    total_energy = np.sum(yf_magnitude)

    yf_normalised = (yf_magnitude / total_energy) * 100  # Now in percentage of total energy

    return yf_magnitude, total_energy, yf_normalised

#----------------------------------------------------------------------
# Function to format frequency labels on energy bar chart
def format_freq(low, high):
    if low >= 1000:
        low_label = f'{low // 1000}k'
    else:
        low_label = str(low)

    if high >= 1000:
        high_label = f'{high // 1000}k'
    else:
        high_label = str(high)

    return f'{low_label}-{high_label}'

#----------------------------------------------------------------------
def find_freq_peaks(xf, yf, peak_threshold=0.1, distance=200, prominence=0.1):
    # Find peaks in the frequency domain. returns the frequencies and their indicies
    # parameters used to control the sensitivity
    N = len(yf)
    # Get the magnitude of the FFT - magnitude of each frequency component given by the absolute value of the complex number
    #Since the FFT result is symmetric, we take only the first half (yf[:N//2]), corresponding to positive frequencies.
    yf_magnitude = np.abs(yf[:N//2])

    yf_magnitude_max = np.max(yf_magnitude)

    # Distance - only peaks separated by at least this many data points will be considered as distinct.
                  # increase or decrease this value depending on how coarse you want the peak detection to be.
    # Prominence - Ensures only peaks that stand out significantly from surrounding values are detected.
                  # Higher values will filter out even more subtle peaks.
    peaks, _ = find_peaks(yf_magnitude, height=yf_magnitude_max * peak_threshold, # ie peak at least this %
                distance=distance, prominence=prominence)

    peak_frequencies = xf[peaks]

    return peaks, peak_frequencies
#----------------------------------------------------------------------
def get_top_peaks(yf_normalised, peak_frequencies, peaks, top_freq_number=None):
    # returns a sorted list of the frequencies with the largest peaks
    # returns a shortened list as per top_freq_number
    # also returns the normalised amplitude
    # o/p can be used for annotating the freq plot

    # If top_freq_number is not provided, include all peaks
    if top_freq_number is None or top_freq_number > len(peaks):
        top_freq_number = len(peaks)  # Include all peaks if not specified or larger than available peaks

    # Sort all peak magnitudes and corresponding frequencies in descending order based on the magnitudes
    sorted_indices = np.argsort(yf_normalised[peaks])[::-1]  # Sort all peaks by their magnitude in descending order

    # Apply the sorted indices to both frequencies and magnitudes
    sorted_frequencies = peak_frequencies[sorted_indices]
    sorted_magnitudes = yf_normalised[peaks][sorted_indices]

    # Now, get the top 'n' values after sorting
    top_frequencies_sorted = sorted_frequencies[:top_freq_number]  # Take the top n sorted frequencies
    top_magnitudes_sorted = sorted_magnitudes[:top_freq_number]  # Take the top n sorted magnitudes

    return top_frequencies_sorted, top_magnitudes_sorted

#----------------------------------------------------------------------
def plot_normalized_spectrum(xf, yf_normalised, frequencies_for_annotation, magnitudes_for_annotation, max_freq=12000):
    # Plot the normalized frequency spectrum with an upper freq limit
    plt.figure(figsize=(10, 4))
    plt.plot(xf, yf_normalised)

    # Annotate the top frequencies on the plot
    for i in range(len(frequencies_for_annotation)):
        plt.annotate(f'{frequencies_for_annotation[i]:.0f} hz',
                    (frequencies_for_annotation[i], magnitudes_for_annotation[i] ),
                    textcoords="offset points",xytext=(5,5), fontsize=9, color='black')

    if max_freq is None or max_freq > max(xf):
        max_freq = max(xf)

    plt.title(f'Frequency spectrum (max {max_freq/1000} kHz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

#----------------------------------------------------------------------
def calc_energy_in_freq_bands(y, sr, freq_bands = [(20, 60), (60, 250), (250, 500), (500, 2000), (2000, 4000),
              (4000, 6000), (6000, 12000), (12000, 20000)]):
    # Calculate energy in each key frequency band

    # Perform STFT to get frequency domain representation
    D = np.abs(librosa.stft(y))  # Magnitude of the STFT
    frequencies = librosa.fft_frequencies(sr=sr)

    # Calculate energy in each frequency band
    band_energies = []
    for (low, high) in freq_bands:
        # Find index range for this frequency band
        band_idx = np.where((frequencies >= low) & (frequencies <= high))[0]

        # Sum the magnitude in this band over time
        band_energy = np.sum(D[band_idx, :], axis=0)

        # Aggregate energy across time (optional)
        total_band_energy = np.sum(band_energy)
        band_energies.append(total_band_energy)

    # Normalize the energies
    total_energy = np.sum(band_energies)
    normalized_band_energies = [energy / total_energy for energy in band_energies]

    return normalized_band_energies

#----------------------------------------------------------------------
def compute_dominant_frequencies(y, sr, n_fft=2048, hop_length=512, top_n=1):
    """
    Compute and track the dominant frequency over time using STFT.

    Parameters:
    - y (np.ndarray): Audio time series.
    - sr (int): Sample rate of y.
    - n_fft (int): Length of the FFT window.
    - hop_length (int): Number of samples between successive frames.
    - top_n (int): The number of top dominant frequencies to track per time frame.

    Returns:
    - dominant_freqs (np.ndarray): Array of dominant frequencies (Hz) for each time frame.
    - times (np.ndarray): Time stamps corresponding to each frame.
    """

    # Perform STFT to get the time-varying frequency spectrum
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Get the frequency values corresponding to the FFT bins
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Initialize list to hold dominant frequencies
    dominant_freqs = []

    # For each time frame, find the top_n dominant frequencies
    for i in range(D.shape[1]):
        # Get the magnitude spectrum for this frame
        spectrum = D[:, i]

        # Find indices of the top_n largest values in the spectrum
        top_indices = np.argsort(spectrum)[-top_n:][::-1]

        # Get the corresponding frequencies for these indices
        dominant_frequencies = frequencies[top_indices]

        # Store the dominant frequency (or frequencies)
        dominant_freqs.append(dominant_frequencies)

    # Convert the list of dominant frequencies to a NumPy array
    dominant_freqs = np.array(dominant_freqs)

    # Compute the time stamps for each frame
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)

    return dominant_freqs, times


#----------------------------------------------------------------------
def assign_line_styles(values, thresholds=[0.02, 0.05, 0.07, 0.1]):
    """
    Function that takes a list of values and assigns line styles ('solid', 'dotted', 'dashed', 'dashdot')
    based on threshold values.

    Parameters:
    - values (list of float): Input list of numeric values.
    - thresholds (list of float): Threshold values to assign line styles. Default thresholds are [0.05, 0.06, 0.07, 0.1].

    Returns:
    - line_styles (list of str): List of line styles corresponding to each value.
    """
    line_styles = []
   # plotly['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

    for value in values:
        if value < thresholds[0]:
            line_styles.append('longdashdot')
        elif value < thresholds[1]:
            line_styles.append('dot')
        elif value < thresholds[2]:
            line_styles.append('dashdot')
        elif value < thresholds[3]:
            line_styles.append('dash')
        else:
            line_styles.append('solid')  # Fallback to solid for values above the highest threshold

    return line_styles

#----------------------------------------------------------------------
def bandpass_filter(signal, sr, low, high, order=4):
    """
    Apply a Butterworth bandpass filter to the signal.

    Parameters:
    - signal (np.ndarray): Input audio signal.
    - sr (int): Sample rate.
    - low (float): Low cutoff frequency in Hz.
    - high (float): High cutoff frequency in Hz.
    - order (int): Order of the filter.

    Returns:
    - filtered_signal (np.ndarray): Bandpass filtered signal.
    """
    nyquist = 0.5 * sr
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = butter(order, [low_norm, high_norm], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

#----------------------------------------------------------------------
def y_remove_nans(y):
    """
    Ensure the audio buffer contains no NaN or infinite values.

    Parameters:
    - y (np.ndarray): Input audio signal.

    Returns:
    - y_clean (np.ndarray): Cleaned audio signal without NaN or inf values.
    """
    # Replace NaN values with 0
    y = np.nan_to_num(y, nan=0.0)

    # Replace inf and -inf with large finite values (or set them to 0 if you prefer)
    y = np.where(np.isinf(y), 0.0, y)

    return y

#----------------------------------------------------------------------
def compute_energy_decay_per_band(y, sr, freq_bands, n_fft=2048, hop_length=512, order=4):
    """
    Compute the energy decay over time for each specified frequency band.

    Parameters:
    - y (np.ndarray): Audio time series.
    - sr (int): Sample rate of y.
    - freq_bands (list of tuples): List of (low_freq, high_freq) in Hz.
    - n_fft (int): Length of the FFT window.
    - hop_length (int): Number of samples between successive frames.
    - order (int): Order of the Butterworth filter.

    Returns:
    - energy_decay (list of np.ndarray): Energy over time for each frequency band.
    - times (np.ndarray): Time stamps corresponding to each frame.
    """

    energy_decay = []

    for (low, high) in freq_bands:
        # Ensure frequency band is within valid range
        if low < 0 or high > sr / 2:
            raise ValueError(f"Frequency band ({low}-{high} Hz) is out of valid range (0-{sr/2} Hz).")
        if low >= high:
            raise ValueError(f"Invalid frequency band: low frequency {low} Hz is not less than high frequency {high} Hz.")

        # Apply bandpass filter
        filtered_y = bandpass_filter(y, sr, low, high, order=order)

        # ensure no NaNs or inf
        filtered_y = y_remove_nans(filtered_y)

        # Compute the STFT of the band-filtered signal
        D = np.abs(librosa.stft(filtered_y, n_fft=n_fft, hop_length=hop_length))

        # Calculate energy (squared magnitude) for each frame
        band_energy = np.sum(D**2, axis=0)

        # Store energy decay for this band
        energy_decay.append(band_energy)

    # Compute time stamps for each frame
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)

    return energy_decay, times

#----------------------------------------------------------------------

def calc_decay_time(y, sr, percent_peak=0.1):
    # Calculate Decay Rate
    # Decay rate is calculated as the time from peak to when the amplitude falls below a threshold (e.g., 10% of the peak)


    # Step 1: Rectify the waveform by taking the absolute value
    rectified_signal = np.abs(y)

    # Step 2: Detect the start of the real signal based on a threshold
    amplitude_threshold = 0.01  # Define a small threshold to detect signal onset
    signal_start_index = np.argmax(rectified_signal > amplitude_threshold)

    # Trim the signal by ignoring the dead time
    trimmed_signal = rectified_signal[signal_start_index:]

    # Step 3: Calculate the envelope using the maximum value across moving bins
    def moving_max(signal, window_size):
        return np.maximum.reduceat(signal, np.arange(0, len(signal), window_size))

    # Define the window size in samples
    window_size = int(sr * 0.025)  # eg 0.025 = 25ms

    # Apply the moving max function to the trimmed signal
    envelope = moving_max(trimmed_signal, window_size)

    # Create a time vector for the envelope that matches the time resolution of the trimmed signal
    envelope_time = np.linspace(signal_start_index / sr, len(trimmed_signal) / sr + signal_start_index / sr, len(envelope))

    # Step 4: Smooth the envelope using a moving average
    def moving_average(signal, window_size):
        return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

    # Define the smoothing window size
    smoothing_window = 3  # low value for less smoothing

    # Apply the moving average smoothing to the envelope
    smoothed_envelope = moving_average(envelope, smoothing_window)

    # Step 5: Find the X% decay point
    initial_peak_value = np.max(smoothed_envelope)
    decay_threshold = percent_peak * initial_peak_value

    # Find the first point where the envelope falls below the 10% threshold
    decay_index = np.argmax(smoothed_envelope < decay_threshold)
    decay_time = envelope_time[decay_index]  # Convert to time in seconds

    return decay_time, initial_peak_value


#----------------------------------------------------------------------
def progressive_range(lst):
    # to find range span of top frequencies
    if len(lst) == 0:
        return []

    ranges = [0]  # The range of the first element is 0 by definition
    for i in range(1, len(lst)):
        current_range = np.max(lst[:i+1]) - np.min(lst[:i+1])  # Calculate range for first i+1 elements
        ranges.append(current_range)

    return ranges

#----------------------------------------------------------------------
   #----------------------------------------------------------------------

def get_frequency_band(frequency, freq_bands_hz_name):

    for low, high, name in freq_bands_hz_name:
        if low <= frequency < high:
            return name
    return "Out of range"

#----------------------------------------------------------------------
# Function to assign frequency bands (as provided before)
def assign_frequency_bands(frequencies, freq_bands):
    band_labels = [
        "Sub-Bass",
        "Bass",
        "Low Midrange",
        "Midrange",
        "Upper Midrange",
        "Presence",
        "Brilliance",
        "Air"
    ]

    # Initialize an empty list to store the bands
    frequency_band_assignment = []

    # Iterate through the list of frequencies
    for freq in frequencies:
        # Assign frequency to a band
        assigned_band = None
        for i, (low, high) in enumerate(freq_bands):
            if low <= freq < high:
                assigned_band = band_labels[i]
                break

        # If frequency does not fit into any band, append "Out of Range"
        if assigned_band is None:
            assigned_band = "Out of Range"

        frequency_band_assignment.append(assigned_band)

    return frequency_band_assignment




#----------------------------------------------------------------------
def get_significant_frequencies(yf, xf, target_percentage):
    """
    Identify the frequencies that contribute to a specific percentage of the total spectrum energy.

    Parameters:
    yf (numpy array): FFT magnitudes (positive frequencies only)
    xf (numpy array): Frequencies corresponding to the FFT magnitudes
    target_percentage (float): The target percentage of the total energy (e.g., 0.20 for 20%)

    Returns:
    numpy array: Frequencies that contribute to the target percentage of the total energy
    """
    # Get the magnitude of the FFT
    N = len(yf)
    yf_magnitude = np.abs(yf[:N//2])

    # Total energy of the spectrum
    total_energy = np.sum(yf_magnitude)

    # Sort frequencies by their magnitude in descending order
    sorted_indices = np.argsort(yf_magnitude)[::-1]
    sorted_magnitudes = yf_magnitude[sorted_indices]
    sorted_frequencies = xf[sorted_indices]

    # Compute the cumulative energy of sorted magnitudes
    cumulative_energy = np.cumsum(sorted_magnitudes)

    # Find the index where cumulative energy crosses the target energy
    target_energy = total_energy * target_percentage
    index = np.where(cumulative_energy >= target_energy)[0][0]

    # Return the frequencies that contribute to the target percentage of energy
    significant_frequencies = sorted_frequencies[:index+1]

    return significant_frequencies

#----------------

# Define the function to get the highest significant frequency
def get_max_significant_frequency(yf, xf, target_percentage):
    """
    Identify the highest frequency that contributes to a specific percentage of the total spectrum energy.

    Parameters:
    yf (numpy array): FFT magnitudes (positive frequencies only)
    xf (numpy array): Frequencies corresponding to the FFT magnitudes
    target_percentage (float): The target percentage of the total energy (e.g., 0.20 for 20%)

    Returns:
    float: The highest frequency that contributes to the target percentage of the total energy
    """
    # Get the magnitude of the FFT for the positive frequencies
    N = len(yf)
    yf_magnitude = np.abs(yf[:N//2])  # Only keep positive frequencies

    # Total energy of the spectrum
    total_energy = np.sum(yf_magnitude)

    # Sort frequencies by their magnitude in descending order
    sorted_indices = np.argsort(yf_magnitude)[::-1]
    sorted_magnitudes = yf_magnitude[sorted_indices]
    sorted_frequencies = xf[sorted_indices]

    # Compute the cumulative energy of sorted magnitudes
    # simplistically energy corresponds to the amplitude
    cumulative_energy = np.cumsum(sorted_magnitudes)

    # Find the index where cumulative energy crosses the target energy
    target_energy = total_energy * target_percentage
    index = np.where(cumulative_energy >= target_energy)[0][0]

    # Return the highest frequency that contributes to the target percentage of energy
    return sorted_frequencies[index]



def plot_time_waveform(y, sr, initial_peak_value, decay_point, max_time_for_plotting=6, height=400, width=600, title="Waveform"):
    fig = go.Figure()

    # Generate time values for the waveform
    x_vals = np.linspace(0, len(y) / sr, num=len(y))

    fig.add_trace(go.Scatter(x=x_vals, y=y, mode='lines', line=dict(color='black', width=1),
                             name="Waveform", showlegend=True),
                  )

    # Add vertical line for decay time
    fig.add_shape(type='line',
                  x0=decay_point, y0=min(y), x1=decay_point, y1=max(y),
                  line=dict(color="blue", dash="dash"),
                  name="Decay Time",
                  )

    # Add horizontal line for peak value
    fig.add_shape(type='line',
                  x0=0, y0=initial_peak_value, x1=max_time_for_plotting, y1=initial_peak_value,
                  line=dict(color="grey", dash="dash"),
                  name="Peak Value",
                  )

    # Update layout
    fig.update_xaxes(title_text="Time (s)", range=[0, max_time_for_plotting])
    fig.update_yaxes(title_text="Amplitude")

    fig.update_xaxes(showgrid=True, gridwidth=0.7, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=0.7, gridcolor='lightgrey')

    fig.update_xaxes(showline=True, linecolor='black', linewidth=2)
    fig.update_yaxes(showline=True, linecolor='black', linewidth=2)

    # Add a center justified title, legend, and set the background to white, with custom height and width
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,  # Center justification
            xanchor='center',
            font=dict(size=18)
        ),
        legend=dict(
            x=0.9, y=1,
            traceorder="normal",
            font_size=10,
            bgcolor="White",  # Legend background color
            bordercolor="Black",
            borderwidth=1
        ),
        plot_bgcolor='white',  # Background color of the plot area
        paper_bgcolor='white',  # Background color of the entire figure
        height=height,  # Custom height
        width=width    # Custom width
    )


    fig.show()
