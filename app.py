import streamlit as st
import audio_functions

#=================
    


st.title("Cymbal analysis")
st.write("This app loads cymbal audio files and presents their key features")


st.header("Select audio clip", divider="gray")

# Define file options in a dropdown menu
files = {
    "Ride": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-08-28 20in2786g (Sa ride) - stick - crash.wav",
    "Crash": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-09-28 16in1143g (Sa crash) - stick - crash.wav",
    "Splash": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-09-28 12in419g (Sa splash) - stick - crash.wav",
}

# Dropdown menu for file selection
file_choice = st.selectbox("Choose a file to load:", options=list(files.keys()))

# Load the selected file
if file_choice:
    file_url = files[file_choice]
    
    
response = requests.get(file_url)
if response.status_code == 200:
    audio_bytes = BytesIO(response.content)
        
    # Play the audio file in Streamlit
    st.write('Play audio file')
    st.audio(audio_bytes, format="audio/wav")
    #st.write('file loaded ok')
else:
    st.error("Failed to load the audio file.")

# Display a "Play" button
#if st.button("Play Audio"):
    # Play the audio file in Streamlit
    #st.audio(audio_bytes, format="audio/wav")


# Load wav file with librosa
# y is time series, sr is sample rate
y, sr = librosa.load(audio_bytes, sr=None)

# Crop the waveform to remove blank space at either end
y_original = y.copy()
y = signal_remove_blank_space(y,sr)


#--------------------------

#==============================================================
# Process waveform
# - initial calculations and overall metrics
#==============================================================

# Define frequency band ranges, names and colours
freq_bands_hz_name = [
        (20, 60, "Sub-Bass"),
        (60, 250, "Bass"),
        (250, 500, "Low Midrange"),
        (500, 2000, "Midrange"),
        (2000, 4000, "Upper Midrange"),
        (4000, 6000, "Presence"),
        (6000, 12000, "Brilliance"),
        (12000, 20000, "Air")
    ]

freq_bands = [(low,high) for low,high,name in freq_bands_hz_name]
freq_band_names = [name for low,high,name in freq_bands_hz_name]
freq_band_colours = ['black', 'red','indigo', 'orange', 'blue', 'green', 'yellow','violet']


#----------------------------------------------------------------------
## Generate frequency domain waveform

# Generate fft
xf, yf = generate_fft(y, sr)

# Normalise the freq spectra
yf_magnitude, total_energy, yf_normalised = normalise_fft(yf)
normalized_band_energies = calc_energy_in_freq_bands(y, sr, freq_bands)


# Sort frequencies by their magnitude in descending order
sorted_indices = np.argsort(yf_magnitude)[::-1]
sorted_magnitudes = yf_magnitude[sorted_indices]
sorted_frequencies = xf[sorted_indices]


#----------------------------------------------------------------------
# Find freq peaks
peaks, peak_frequencies = find_freq_peaks(xf, yf, peak_threshold=0.1, distance=100, prominence=0.5)

# Sort the list of frequencies and their normalised amplitude to identify the largest ones
top_frequencies_sorted, top_magnitudes_sorted = get_top_peaks(yf_normalised, peak_frequencies, peaks, top_freq_number=None)

# Determine the range of frequencies spanned by progressive top peaks
# ie the ranges between the min and max freq of the top largest peaks
top_frequency_ranges = progressive_range(top_frequencies_sorted)


#----------------------------------------------------------------------
metrics = {} # initialise metrics store

# Overall signal energy
metrics['overall_energy'] = total_energy

# calc freq metrics
metrics['highest_top_freq'] = np.max(top_frequencies_sorted)
metrics['lowest_top_freq'] = np.min(top_frequencies_sorted)

# generate list of freq of 5 top peaks
for p in range(5):
    metrics[f"top_freq_{p+1}"] = top_frequencies_sorted[p]

# generate list of names of 5 top peaks
for p in range(5):
    metrics[f"top_freq_{p+1}_band"] = get_frequency_band(top_frequencies_sorted[p],freq_bands_hz_name)


# calc range of top frequency peaks
metrics['top_3_freq_range'] = max(top_frequencies_sorted[:6]) - min(top_frequencies_sorted[:6])
metrics['top_5_freq_range'] = max(top_frequencies_sorted[:4]) - min(top_frequencies_sorted[:4])


# calc the number of freq
#that contribute to top % energy
metrics['num_freq_in_top_5%_energy'] = get_significant_frequencies(yf, xf, 0.05)
metrics['num_freq_in_top_10%_energy']  = get_significant_frequencies(yf, xf, 0.1)
metrics['num_freq_in_top_20%_energy'] = get_significant_frequencies(yf, xf, 0.2)
metrics['num_freq_in_top_50%_energy'] = get_significant_frequencies(yf, xf, 0.5)


# Calculate attack and decay time of time waveform
#metrics['attack_time'] = calc_attack_time(y, sr)
decay_time_10pc, initial_peak_value = calc_decay_time(y, sr, percent_peak=0.1)
decay_time_50pc, initial_peak_value = calc_decay_time(y, sr, percent_peak=0.5)

metrics['decay_time_10pc'] = decay_time_10pc
metrics['decay_time_50pc'] = decay_time_50pc

# centroid - overall center of mass
overall_spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
metrics['overall_spectral_centroid'] = overall_spectral_centroid
metrics[f"overall_spectral_centroid_band"] = get_frequency_band(overall_spectral_centroid,freq_bands_hz_name)

# bandwidth - oveall spread of frequencies
metrics['overall_spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

# flatness - how noiselike it is
metrics['overall_spectral_flatness']  = np.mean(librosa.feature.spectral_flatness(y=y))

# zero_crossing rate - how noiselike it is
metrics['overall_zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))

# onset strength - how strongly sound energy changes over time
metrics['onset_strength']= np.mean(librosa.onset.onset_strength(y=y, sr=sr))


#------------------------
# Peaks range and variance
# Simple measures of peak spread


# Define the number of top frequency peaks
freq_spectrum_top_peaks_num = 20

# determine the freq of the largest amplitude peaks
top_freqs = np.array(top_frequencies_sorted[:freq_spectrum_top_peaks_num])

# Calculate range of top amplitude peaks
metrics['top_peak_range'] = np.max(top_freqs) - np.min(top_freqs)

# Calculate variance of top peaks
metrics['top_peak_variance'] = np.var(top_freqs)

# Calculate inter-peak distances
# distances between consecutive peaks
# ie how evenly peaks are spread out across freq spectrum
# nb the top peaks should be in freq order to make sense

inter_peak_distances = np.diff(np.sort(top_freqs))

# Mean and standard deviation of inter-peak distances
metrics["inter_top_peak_distance_mean"] = np.mean(inter_peak_distances)
metrics["inter_top_peak_distance_std"] = np.std(inter_peak_distances)

# Range & variance after logarithmic transformation
# Transform freq to logs to account for logarithmic nature

log_top_freqs = np.log2(top_freqs)

# Calculate range and variance on the log-transformed frequencies
metrics["log_top_peaks_range"] = np.max(log_top_freqs) - np.min(log_top_freqs)
metrics["log_top_peaks_variance"] = np.var(log_top_freqs)

# kurtosis of top frequency peaks
metrics["top_peaks_kurtosis"] = kurtosis(top_freqs)



#--------------------------

# How random the peak distribution is
# Min = 0 (all peaks in same place)
# Max = log(len(yf_normalised))

# Calculate spectral entropy
#spectral_entropy = entropy(prob_distribution)
metrics["overall_spectral_entropy"] = entropy(yf_normalised)

# Count the number of this number of top peaks in each band
top_peaks_across_bands_num = 20 # ie how many of these are in each band

peaks_per_band = []
for band in freq_bands:
    lower_bound, upper_bound = band
    count_in_band = sum((lower_bound <= peak <= upper_bound) for peak in top_frequencies_sorted[:top_peaks_across_bands_num])
    peaks_per_band.append(count_in_band)

metrics["peaks_per_band"] = {band:peak for band,peak in zip (freq_band_names,peaks_per_band)}






st.header("Time-based charts", divider="gray")

# Generate sample data for plotting
x = np.linspace(0, 10, 100)

# Create a 2x2 grid of columns
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Plot 1
with col1:
    fig, ax = plt.subplots()
    ax.plot(range(len(xf)), sorted_frequencies)
    ax.set_title("sorted freq")
    st.pyplot(fig)

# Plot 2
with col2:

  
    #fig, ax = plt.subplots()
    #ax.plot(x, np.cos(x))
   # ax.set_title("Cosine Wave")
    
    # Define target percentages to explore
    target_percentages = np.linspace(0.05, 0.50,100)  # eg from 5% to 95%

    # Collect the highest significant frequency for each target percentage
    max_frequencies = [get_max_significant_frequency(yf, xf, p) for p in target_percentages]

    # Plot the results
    fig, ax = plt.subplots()
    st.write('The frequencies which contribute to the top 50% energy')
    ax.plot(target_percentages,max_frequencies)
    ax.set_xlabel('Target Percentage of Total Energy')
    ax.set_ylabel('Significant Frequencies (Hz)')
    ax.set_title('Significant Frequencies vs. Target Percentage of Total Energy')
    st.pyplot(fig)

# Plot 3
with col3:
    fig, ax = plt.subplots()
    ax.plot(x, np.tan(x))
    ax.set_title("Tangent Wave")
    st.pyplot(fig)

# Plot 4
with col4:
    fig, ax = plt.subplots()
    ax.plot(x, np.sinh(x))
    ax.set_title("Hyperbolic Sine Wave")
    st.pyplot(fig)

st.header("Frequency-based charts", divider="gray")



# Generate and display multiple plots in a vertical layout
#for i in range(4):
 #   fig, ax = plt.subplots()
#    ax.plot(x, np.sin(x + i))
 #   ax.set_title(f"Plot {i+1}")
#    st.pyplot(fig)




with st.expander("View First Row of Plots"):
    col5, col6 = st.columns(2)
  
    with col5:
        # Plot 1
        fig, ax = plt.subplots()
        ax.plot(x, np.sin(x))
        st.pyplot(fig)
      
    

    with col6:
        # Plot 2
        fig, ax = plt.subplots()
        ax.plot(x, np.cos(x))
        st.pyplot(fig)
