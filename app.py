#==============================================================
# imports
#==============================================================
import streamlit as st # needed in app

from audio_functions import *

# imports needed directly in this file
import requests
from io import BytesIO
import plotly.graph_objects as go
import librosa

import urllib.parse
from PIL import Image

#==============================================================
# Start streamlit code
#==============================================================

# Define file options in a dropdown menu
files_audio = {
    "Ride": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-08-28 20in2786g (Sa ride) - stick - crash.wav",
    "Crash": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-09-28 16in1143g (Sa crash) - stick - crash.wav",
    "Splash": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-09-28 12in419g (Sa splash) - stick - crash.wav",
}

files_image = {
    "Ride": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-15 20in 2786g (Sa ride).jpg",
    "Crash": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-15 16in 1143g (Sa crash).jpg",
    "Splash": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-15 12in 419g (Sa splash) (2).jpg",
}

# Replace spaces with %20
files_image_no_spaces = {
    key: urllib.parse.quote(value, safe=":/") for key, value in files_image.items()
}

st.title("Cymbal analysis")
st.write('Piers Walker 2024 [https://github.com/pierswalker71](https://github.com/pierswalker71)')
intro_text = f'''
Hi, welcome to my cymbal analyser.

You can select an audio file of one of the cymbals from my drumkit being hit. The audio file will be  
analysed in both time and frequency and you will be presented with charts that portray the sound in intreresting and informative ways.
'''
st.write(intro_text)

st.header("Load cymbal audio file", divider="gray")



# Dropdown menu for file selection
st.write('Choose an audio wav file to load.')
st.write(f"There are currently {len(files_audio)} different cymbal sounds to choose from ({", ".join(files_audio.keys())}).")

file_choice = st.selectbox("", options=list(files_audio.keys()))

# Load the selected file
if file_choice:
    st.markdown(f'<p style="text-align:center; font-size:20px;">You have chosen the {file_choice}</p>', unsafe_allow_html=True)
    file_url_audio = files_audio[file_choice]
    file_url_image = files_image_no_spaces[file_choice]

col1, col2 = st.columns(2)

with col1:
    response = requests.get(file_url_image)
    if response.status_code == 200:

        # Show the image
        st.write('**Here is what the cymbal looks like:**')

        # Fetch and open the image
        response = requests.get(file_url_image)
        image = Image.open(BytesIO(response.content))

        # Resize the image (maintaining aspect ratio)
        new_width = 300
        aspect_ratio = image.height / image.width
        new_height = int(new_width * aspect_ratio)
        resized_image = image.resize((new_width, new_height))

        # Display image
        st.image(resized_image, caption=file_choice)

    else:
        st.error("Failed to load the image.")

with col2:
    response = requests.get(file_url_audio)
    if response.status_code == 200:
        audio_bytes = BytesIO(response.content)

        # Play the audio file in Streamlit
        st.write('**Here is what the cymbal sounds like:**')
        st.audio(audio_bytes, format="audio/wav")
    else:
        st.error("Failed to load the audio file.")


# Load data

# Load wav file with librosa
# y is time series, sr is sample rate
y, sr = librosa.load(audio_bytes, sr=None)

# Crop the waveform to remove blank space at either end
y_original = y.copy()
y = signal_remove_blank_space(y,sr)

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

#----------------------------------------------------
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

#==============================================================
# **** Generate plots ****
#==============================================================

st.header("How long the cymbal rings for", divider="gray")

# Key stats

st.write(f"It takes {decay_time_50pc:.1f}s for the sound to decay to 50% its initial maximum amplitude.")

text = f"It takes <b>{decay_time_10pc:.1f}s</b> for the sound to decay to 10% its initial maximum amplitude."
st.markdown(f'<p style="color:blue; background-color:yellow; font-size:15px; font-weight:normal;">{text}</p>', unsafe_allow_html=True)

#--------------------------------------------------------------
# Plot 1
#--------------------------------------------------------------
with st.expander("Audio waveform",expanded=True):

    fig = go.Figure()
    max_time_for_plotting=10
    decay_point=decay_time_10pc

    # Generate time values for the waveform
    x_vals = np.linspace(0, len(y) / sr, num=len(y))

    fig.add_trace(go.Scatter(x=x_vals, y=y, mode='lines', line=dict(color='black', width=1),
                             name="Waveform", showlegend=True),
                  )

    # Add vertical line for decay time
    # Add vertical line for decay time as a scatter trace
    fig.add_trace(go.Scatter(
       x=[decay_point, decay_point],
        y=[min(y), max(y)],
        mode='lines',
        line=dict(color="blue", dash="dash"),
        name="Decay Time"
    ))

    # Add horizontal line for peak value as a scatter trace
    fig.add_trace(go.Scatter(
        x=[0, max_time_for_plotting],
        y=[initial_peak_value, initial_peak_value],
        mode='lines',
        line=dict(color="grey", dash="dash"),
        name="Peak Value"
    ))

    # Update layout
    fig.update_layout(title_text="Audio waveform in time", title_x=0.5, title_xanchor='center')
    fig.update_xaxes(title_text="Time (s)", range=[0, max_time_for_plotting])
    fig.update_yaxes(title_text="Amplitude")
    fig.update_layout(height=400, width=700)
    
    fig.update_layout(
        legend=dict(
            x=0.9, y=1,
            traceorder="normal",
            font_size=10,
            bgcolor="White",  # Legend background color
            bordercolor="Black",
            borderwidth=1
        ),
        xaxis=dict(
        title_font=dict(color="black"), tickfont=dict(color="black"),
        showline=True, linecolor='black', linewidth=1,
        gridwidth=0.7, gridcolor='lightgrey',
        ),
        yaxis=dict(
        title_font=dict(color="black"), tickfont=dict(color="black"),
        showline=True, linecolor='black', linewidth=1,
        gridwidth=0.7, gridcolor='lightgrey',
        ),
        plot_bgcolor='white',  # Background color of the plot area
        paper_bgcolor='white',  # Background color of the entire figure
    )
    
    st.write("Here you can see the amplitude of the audio waveform as it changes in time.")
    st.plotly_chart(fig)


#==============================================================
st.header("The key frequencies", divider="gray")
#==============================================================

# Key stats

st.markdown("The top peak frequencies are:")

text = f'''
<b>{metrics[f"top_freq_1"]:.0f}Hz</b> ({metrics[f"top_freq_1_band"]}) /n
<b>{metrics[f"top_freq_2"]:.0f}Hz</b> ({metrics[f"top_freq_2_band"]})
'''
st.markdown(f'<p style="color:blue; background-color:yellow; font-size:15px; font-weight:normal;">{text}</p>', unsafe_allow_html=True)

st.write(f"{metrics["top_freq_1"]:.0f} ({metrics["top_freq_1_band"]})")

text = f"The key frequencies are <b>{decay_time_10pc:.1f}s</b>."
st.markdown(f'<p style="color:blue; background-color:yellow; font-size:15px; font-weight:normal;">{text}</p>', unsafe_allow_html=True)

#--------------------------------------------------------------
# Plot 2
#--------------------------------------------------------------



#--------------------------------------------------------------
# Plot 3
#--------------------------------------------------------------
with st.expander("Frequency spectrum",expanded=True):
    fig = go.Figure()
    # Define some necessary constants for freq spectrum plot
    max_freq = 6000
    freq_spectrum_plot_top_peaks_num = 5
    
    # Add the frequency domain (FFT) plot to the second subplot
    fig.add_trace(go.Scatter(x=xf, y=yf_normalised, mode='lines', line=dict(color='black', width=1), name="Spectrum"),
                  )
    
    # Annotate the top frequencies on the plot
    for i in range(len(top_frequencies_sorted[:freq_spectrum_plot_top_peaks_num])):
        fig.add_trace(go.Scatter(x=[top_frequencies_sorted[i]], y=[top_magnitudes_sorted[i]],
                                 text=[f'{top_frequencies_sorted[i]:.0f}'],
                                 mode='text', textposition="top right", showlegend=False, textfont=dict(color='black', size=10)),
                     )
    
    # Add colored backgrounds for each frequency band
    for (lower_bound, upper_bound), color in zip(freq_bands, freq_band_colours):
        fig.add_shape(type='rect', x0=lower_bound, x1=upper_bound, y0=0, y1=1,
                      xref='x1', yref='paper', fillcolor=color, opacity=0.5, layer="below", line_width=0,
        )
    
    
    #fig.layout.annotations[1].text = f"Spectrum (peak: {round(metrics['top_freq_1'])} Hz. Top 5 range: {round(metrics['top_5_freq_range'])} Hz)" 

    fig.update_layout(title_text="Frequency spectrum", title_x=0.5, title_xanchor='center')
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, max_freq])
    fig.update_yaxes(title_text="Amplitude")
    fig.update_yaxes(range=[0, max(yf_normalised)*1.05])
    fig.update_layout(height=400, width=700)

    fig.update_layout(     
        xaxis=dict(
        title_font=dict(color="black"), tickfont=dict(color="black"),
        showline=True, linecolor='black', linewidth=1,
        gridwidth=0.7, gridcolor='lightgrey',
        ),
        yaxis=dict(
        title_font=dict(color="black"), tickfont=dict(color="black"),
        showline=True, linecolor='black', linewidth=1,
        gridwidth=0.7, gridcolor='lightgrey',
        ),
        plot_bgcolor='white',  
        paper_bgcolor='white', 
    )

    st.write("Here you can see the frequency spectrum of the audio waveform.")
    st.write("The peaks show the frequencies which are present most strongly in the sound.")
    st.plotly_chart(fig)


#--------------------------------------------------------------
# Plot 4
#--------------------------------------------------------------
# Compute energy decay for each frequency band

with st.expander("Energy in each frequency band",expanded=True):
    fig = go.Figure()
    energy_decay, times = compute_energy_decay_per_band(y, sr, freq_bands)
    
    # Define line widths and line styles based on thresholds
    line_widths = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]  # Example
    line_styles = assign_line_styles(normalized_band_energies, thresholds=[0.03, 0.06, 0.07, 0.1])
    
    # Add energy decay traces for each frequency band to the subplot
    for i, (low, high) in enumerate(freq_bands):
        label = f"{low}-{high} Hz"
        fig.add_trace(
            go.Scatter(
                x=times,
                y=energy_decay[i],
                mode='lines',
                line=dict(color=freq_band_colours[i], width=line_widths[i], dash=line_styles[i]),
                name=label,
                showlegend=True if i == 0 else False
            ),
    
        )
    
    fig.update_layout(title_text="The energy in each frequency band", title_x=0.5, title_xanchor='center')
    fig.update_xaxes(title_text='Time (s)', range=[0, 3])
    fig.update_yaxes(title_text='Energy')
    fig.update_yaxes(range=[0, np.max([np.max(arr) for arr in energy_decay])*1.05])
    fig.update_layout(height=400, width=700)

    fig.update_layout(     
        xaxis=dict(
        title_font=dict(color="black"), tickfont=dict(color="black"),
        showline=True, linecolor='black', linewidth=1,
        gridwidth=0.7, gridcolor='lightgrey',
        ),
        yaxis=dict(
        title_font=dict(color="black"), tickfont=dict(color="black"),
        showline=True, linecolor='black', linewidth=1,
        gridwidth=0.7, gridcolor='lightgrey',
        ),
        plot_bgcolor='white',  
        paper_bgcolor='white'
    )

 
    # Add a legend
    fig.update_layout(
        legend=dict(x=1, y=1, traceorder="normal", font_size=10)
    )

    st.write("Here you can see the amount energy in each of the key frequency bands.")
    st.plotly_chart(fig)
    
#--------------------------------------------------------------
# Plot 5
#--------------------------------------------------------------
with st.expander("Significant frequencies",expanded=True):
    
    # Define target percentages to explore
    target_percentages = np.linspace(0.05, 0.50,100)  # eg from 5% to 95%

    # Collect the highest significant frequency for each target percentage
    max_frequencies = [get_max_significant_frequency(yf, xf, p) for p in target_percentages]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
    x=target_percentages,
    y=max_frequencies,
    mode='lines+markers',  # Shows both lines and points
    name='Significant Frequencies',
    line=dict(color='black'),
    marker=dict(color='black') 
    ))

    fig.update_layout(title_text="The significant frequencies which contribute to the total audio energy", title_x=0.5, title_xanchor='center')
    fig.update_xaxes(title_text="Proportion of total energy")
    fig.update_yaxes(title_text="Significant frequencies (Hz)")
    fig.update_yaxes(range=[0, max(max_frequencies) * 1.1])
    fig.update_layout(height=400, width=700)
    
    fig.update_layout(     
        xaxis=dict(
        title_font=dict(color="black"), tickfont=dict(color="black"),
        showline=True, linecolor='black', linewidth=1,
        gridwidth=0.7, gridcolor='lightgrey',
        ),
        yaxis=dict(
        title_font=dict(color="black"), tickfont=dict(color="black"),
        showline=True, linecolor='black', linewidth=1,
        gridwidth=0.7, gridcolor='lightgrey',
        ), 
        plot_bgcolor='white',  
        paper_bgcolor='white'
    )
    
    st.write("Here you can see the frequencies which contribute to the top 50% of the total energy in the audio signal.")
    st.plotly_chart(fig)

#==============================================================
st.header("?? charts", divider="gray")
#==============================================================

