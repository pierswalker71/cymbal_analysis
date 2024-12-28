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
    "Sabian 20in 2786g ride [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-08-28 20in2786g (Sa ride) - stick - crash.wav",
    "Sabian 20in 2786g ride [tap]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-09-28 20in2786g (Sa ride) - stick - tap.wav",    
    "Sabian 16in 1143g crash [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-09-28 16in1143g (Sa crash) - stick - crash.wav",
    "Sabian 12in 419g splash [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-09-28 12in419g (Sa splash) - stick - crash.wav",
    "Istanbul 14in 1188g hihat [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-09-28 14in 1188g (Istanbul hi hat) - stick - crash.wav",    
    "Pearl 14in 1182g hihat [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-08-28 14in1182g (pearl tophat) - stick - crash.wav",
    "Diamond 14in 900g hihat [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-08-28 14in900g (diamond bottomhat) - stick - crash.wav",
     "4in 61g broach [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-22 NA 4in 61g Broach [crash].wav",
}

files_image = {
    "Sabian 20in 2786g ride [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-15 20in 2786g (Sa ride).jpg",
    "Sabian 20in 2786g ride [tap]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-15 20in 2786g (Sa ride).jpg",
    "Sabian 16in 1143g crash [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-15 16in 1143g (Sa crash).jpg",
    "Sabian 12in 419g splash [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-15 12in 419g (Sa splash) (2).jpg",
    "Istanbul 14in 1188g hihat [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-22 Istanbul 14in 1188g hihat (crash).jpg",
    "Pearl 14in 1182g hihat [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-22 14in 1182g (pearl hh).jpg",
    "Diamond 14in 900g hihat [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-22 14in 900g (diamond hh).jpg",
    "4in 61g broach [crash]": "https://raw.githubusercontent.com/pierswalker71/cymbal_analysis/main/2024-12-22 61g 4in broach.jpg",
}

# Replace spaces with %20
files_image_no_spaces = {
    key: urllib.parse.quote(value, safe=":/") for key, value in files_image.items()
}

st.title('🥁 Cymbal analysis')
st.write('Piers Walker 2024 ([https://github.com/pierswalker71](https://github.com/pierswalker71))')
intro_text = f'''
:tada: **Welcome to the Cymbal Sound Analyser!**

Select an audio file of one of my drumkit cymbals being struck either as a tap or a crash, and this tool will analyse the sound in both the time and frequency domains. 
It will show you key metrics and visualisations that will reveal unique characteristics of the cymbal's sound.

**Section info**
'''
st.write(intro_text)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎵 1. Selected cymbal", 
                                        "⏱️ 2. Time", 
                                        "📊 3. Frequency stats",
                                        "🎼 4. Overall frequency",
                                        "📉 5. Frequency over time"])

with tab1:
    st.write(":point_right: This is where you see and hear the selected cymbal.")
with tab2:
    st.write(":point_right: This section shows how the amplitude of the sound changes in time.")
with tab3:
    st.write(":point_right: This section shows metrics relating to the key frequency components of the audio.")
with tab4:
    st.write(":point_right: This section shows charts of the frequencies found in the entire audio clip.")
with tab5:
    st.write(":point_right: This section shows how the key frequencies change throughout the duration of the audio clip.")    

#==============================================================
st.sidebar.title("Audio selection")
# Dropdown menu for file selection
st.sidebar.write('Choose an audio wav file to load.')
file_choice = st.sidebar.selectbox("Select", options=list(files_audio.keys()))
#==============================================================


#==============================================================
st.header("", divider="orange")
st.header("Selected cymbal", divider="gray")
#==============================================================


st.write("Click the < icon at the top left of the app :arrow_upper_left: to select the audio.")
#st.write(f"There are currently {len(files_audio)} different cymbal sounds to choose from: {", ".join(files_audio.keys())}.")
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
        (20, 60, "Sub-bass"),
        (60, 250, "Bass"),
        (250, 500, "Low midrange"),
        (500, 2000, "Midrange"),
        (2000, 4000, "Upper midrange"),
        (4000, 6000, "Presence"),
        (6000, 12000, "Brilliance"),
        (12000, 20000, "Air")
    ]

freq_bands = [(low,high) for low,high,name in freq_bands_hz_name]
freq_band_names = [name for low,high,name in freq_bands_hz_name]
freq_band_colours = ['black', 'red','indigo', 'orange', 'blue', 'green', 'yellow','violet']
freq_band_labels = [format_freq(low, high) for (low, high) in freq_bands]

#----------------------------------------------------------------------
## Generate frequency domain waveform

# Generate fft
xf, yf, yf_db = generate_fft(y, sr)

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
metrics['top_3_freq_range'] = max(top_frequencies_sorted[:3]) - min(top_frequencies_sorted[:3])
metrics['top_5_freq_range'] = max(top_frequencies_sorted[:5]) - min(top_frequencies_sorted[:5])

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


# Significant frequencies
# Define target percentages to explore
significant_frequency_target_percentages = np.linspace(0.05, 0.50,100)  # eg from 5% to 95%

# Collect the highest significant frequency for each target percentage
significant_frequency_max_frequencies = [get_max_significant_frequency(yf, xf, p) for p in significant_frequency_target_percentages]

metrics["significant_frequency_max_frequencies_mean"] = np.mean(significant_frequency_max_frequencies)
metrics["significant_frequency_max_frequencies_median"] = np.median(significant_frequency_max_frequencies)
metrics["significant_frequency_max_frequencies_min"] = np.min(significant_frequency_max_frequencies)
metrics["significant_frequency_max_frequencies_max"] = np.max(significant_frequency_max_frequencies)


#==============================================================
# **** Generate plots ****
#==============================================================


#==============================================================
st.header("Time characteristics", divider="gray")
#==============================================================

# Key stats
st.write("**How long it takes for the sound to die down:**")

#text = f"It takes <b>{decay_time_50pc:.1f}s</b> for the sound to decay to 50% its initial peak value."
#st.markdown(f'<p style="color:blue; background-color:lightyellow; font-size:15px; font-weight:normal;">{text}</p>', unsafe_allow_html=True)

text = f'''
It takes <b>{decay_time_50pc:.1f}s</b> for the sound to decay to 50% its initial peak value 
and <b>{decay_time_10pc:.1f}s</b> for the sound to decay to 10% its initial peak value.'''
st.markdown(f'<p style="color:blue; background-color:lightyellow; font-size:15px; font-weight:normal;">{text}</p>', unsafe_allow_html=True)

#==============================================================
#--------------------------------------------------------------
# Plot 1
#--------------------------------------------------------------
#==============================================================
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
    fig.update_layout(height=400, width=700)
    
    st.write("Here you can see the amplitude of the audio waveform as it changes in time.")
    st.plotly_chart(fig)


#==============================================================
st.header("Frequency stats", divider="gray")
#==============================================================

# Key stats

Top_energy_bands = sorted(enumerate(normalized_band_energies), key=lambda x: x[1], reverse=True)[:5]
# Extract indices and values
Top_energy_bands_indices, Top_energy_bands_values = zip(*Top_energy_bands)

st.markdown("**Key frequency stats:**")
text = f'''
1. The overall spectral centroid is <b>{metrics['overall_spectral_centroid']:,.0f}Hz</b> ({metrics[f"overall_spectral_centroid_band"]})  <br>
2. The top frequency is <b>{metrics[f"top_freq_1"]:,.0f}Hz</b> ({metrics[f"top_freq_1_band"]}) <br>
3. Top frequency band is <b>{freq_band_names[Top_energy_bands_indices[0]]} ({freq_band_labels[Top_energy_bands_indices[0]]}Hz)</b>  <br>
4. The median frequency that contributes to 50% of the total signal energy is <b>{metrics["significant_frequency_max_frequencies_median"]:,.0f}Hz</b>  <br>
5. The overall spread of frequencies is <b>{metrics['overall_spectral_bandwidth']:,.0f}Hz</b>   <br>                                                                                                     
'''
st.markdown(f'<p style="color:blue; background-color:lightyellow; font-size:15px; font-weight:normal; padding:10px; border-radius:5px;">{text}</p>', unsafe_allow_html=True)


st.markdown("**Most of the sound energy is concentrated in the following 5 bands:**")

text = f'''
1. <b>{freq_band_names[Top_energy_bands_indices[0]]} ({freq_band_labels[Top_energy_bands_indices[0]]}Hz)</b> ({100*Top_energy_bands_values[0]:.0f}% total energy) <br>
2. <b>{freq_band_names[Top_energy_bands_indices[1]]} ({freq_band_labels[Top_energy_bands_indices[1]]}Hz)</b> ({100*Top_energy_bands_values[1]:.0f}% total energy) <br>
3. <b>{freq_band_names[Top_energy_bands_indices[2]]} ({freq_band_labels[Top_energy_bands_indices[2]]}Hz)</b> ({100*Top_energy_bands_values[2]:.0f}% total energy) <br>
4. <b>{freq_band_names[Top_energy_bands_indices[3]]} ({freq_band_labels[Top_energy_bands_indices[3]]}Hz)</b> ({100*Top_energy_bands_values[3]:.0f}% total energy) <br>
5. <b>{freq_band_names[Top_energy_bands_indices[4]]} ({freq_band_labels[Top_energy_bands_indices[4]]}Hz)</b> ({100*Top_energy_bands_values[4]:.0f}% total energy) <br>
'''
st.markdown(f'<p style="color:blue; background-color:lightyellow; font-size:15px; font-weight:normal;">{text}</p>', unsafe_allow_html=True)


st.markdown("**The top 5 peak frequencies are:**")
text = f'''
1. <b>{metrics[f"top_freq_1"]:,.0f}Hz</b> ({metrics[f"top_freq_1_band"]}) <br>
2. <b>{metrics[f"top_freq_2"]:,.0f}Hz</b> ({metrics[f"top_freq_2_band"]}) <br>
3. <b>{metrics[f"top_freq_3"]:,.0f}Hz</b> ({metrics[f"top_freq_3_band"]}) <br>
4. <b>{metrics[f"top_freq_4"]:,.0f}Hz</b> ({metrics[f"top_freq_4_band"]}) <br>
5. <b>{metrics[f"top_freq_5"]:,.0f}Hz</b> ({metrics[f"top_freq_5_band"]})
'''
st.markdown(f'<p style="color:blue; background-color:lightyellow; font-size:15px; font-weight:normal;">{text}</p>', unsafe_allow_html=True)

st.markdown("**The range of the top peak frequencies are:**")
text = f'''
Range of top 3 peaks is <b>{metrics['top_3_freq_range']:,.0f}Hz</b> 
({(100 * metrics['top_3_freq_range'] / metrics['top_freq_1']):,.0f}% of top peak value)<br>
Range of top 5 peaks is <b>{metrics['top_5_freq_range']:,.0f}Hz</b> 
({(100 * metrics['top_5_freq_range'] / metrics['top_freq_1']):,.0f}% of top peak value)
'''
st.markdown(f'<p style="color:blue; background-color:lightyellow; font-size:15px; font-weight:normal; padding:10px; border-radius:5px;">{text}</p>', unsafe_allow_html=True)


st.markdown("**Other frequency stats:**")
text = f'''
1. The spectral flatness is <b>{metrics['overall_spectral_flatness']:,.1e}</b> and the zero crossing rate is <b>{metrics['overall_zero_crossing_rate']:,.1e}</b> (how noise-like the sound is)<br>
2. The onset strength is <b>{metrics['onset_strength']:,.1e}</b> (how strongly sound energy changes over time)  <br>                                                                                                        
'''
st.markdown(f'<p style="color:blue; background-color:lightyellow; font-size:15px; font-weight:normal; padding:10px; border-radius:5px;">{text}</p>', unsafe_allow_html=True)


#==============================================================
st.header("Overall frequency characteristics", divider="gray")
#==============================================================


#==============================================================
#--------------------------------------------------------------
# Plot 2
#--------------------------------------------------------------
#==============================================================
with st.expander("Frequency band colour scheme", expanded=False):
    fig = go.Figure()
    
    # Add rectangles (bars) for each frequency band with color
    for name, color, label in zip(freq_band_names, freq_band_colours, freq_band_labels):
        fig.add_trace(
            go.Bar(
                x=[label],
                y=[0.5],  # Smaller height for the bars
                marker=dict(color=color, opacity=0.7),
                name=name
            )
        )
    
    # Update layout to style the plot
    fig.update_layout(
        title=dict(
            text="Frequency band colour scheme",
            font=dict(color="black", size=16)  # Title in black
        ),
        xaxis=dict(
            title="Frequency bands (Hz)",
            title_font=dict(color="black"),  # X-axis title in black
            tickfont=dict(color="black"),    # X-axis tick labels in black
            tickangle=45                     # Rotate x-axis labels
        ),
        yaxis=dict(
            visible=False,  # Hide y-axis
            range=[0, 0.6]
        ),
        legend=dict(
            title=dict(text="Frequency band names", font=dict(color="black")),  
            font=dict(color="black") 
        ),
        plot_bgcolor="white",  # White plot background
        paper_bgcolor="white"  # White overall background
    )
    
    fig.update_layout(height=400, width=700)
    st.plotly_chart(fig, use_container_width=True)


#==============================================================
#--------------------------------------------------------------
# Plot 3
#--------------------------------------------------------------
#==============================================================
with st.expander("Frequency bands", expanded=True):

    fig = go.Figure()

    # Bar chart for normalized energy distribution
    fig.add_trace(
        go.Bar(
            x=freq_band_labels,
            y=normalized_band_energies,
            marker=dict(color=freq_band_colours),
            name="Energy",
            opacity=0.7,
            showlegend=True
        )
    )

    # Count the number of peaks in each band
    top_peaks_across_bands_num = 20

    # Overlay the number of peaks on the secondary y-axis (stalks with round tops)
    x_values = freq_band_labels
    y_values = peaks_per_band

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            line=dict(color='black'),
            marker=dict(symbol='star', size=10),
            name=f'Number of peaks in top {top_peaks_across_bands_num}',
        )
    )

    # Adding vertical lines to represent the stalks
    for i in range(len(x_values)):
        fig.add_trace(
            go.Scatter(
                x=[x_values[i], x_values[i]],
                y=[0, y_values[i]],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            )
        )

    fig.update_layout(title_text="Energy and number of peaks in each frequency band", title_x=0.5, title_xanchor='center')

    # Set the y-axis for both axes to start at zero and configure dual y-axes
    fig.update_layout(
        yaxis=dict(
            title='Proportion of energy',
            title_font=dict(color="black"), tickfont=dict(color="black"),
            showline=True, linecolor='red', linewidth=1,
            gridwidth=0.3, gridcolor='rgba(255,153,153,0.7)',
            range=[0, None]
        ),
        yaxis2=dict(
            title='Number of peaks',
            title_font=dict(color="black"), tickfont=dict(color="black"),
            showline=True, linecolor='black', linewidth=1,
            gridwidth=0.7, gridcolor='lightgrey',
            overlaying='y',
            side='right',
            range=[0, None]
        ),
        xaxis=dict(
            title='Frequency Band (Hz)',
            title_font=dict(color="black"), tickfont=dict(color="black"),
            showline=True, linecolor='black', linewidth=1,
            gridwidth=0.7, gridcolor='lightgrey',
        ),
        
        legend=dict(
            x=0.85,
            y=1,
            traceorder="normal",
            font_size=10
        ),
        plot_bgcolor='white',  
        paper_bgcolor='white', 
    )
    fig.update_layout(height=400, width=700)

    # Assign the Scatter trace for peaks to use the secondary y-axis
    fig.data[1].update(yaxis='y2')
    for i in range(len(x_values)):
        fig.data[2 + i].update(yaxis='y2')  # Update the stalks to use the secondary y-axis
    
    st.write("Here you can see the amount of energy and the number of the peak frequencies in each of the audio bands.")

    text = f'''
        The bands containing the most energy are: 
        {freq_band_names[Top_energy_bands_indices[0]]} ({freq_band_labels[Top_energy_bands_indices[0]]}Hz) ({100*Top_energy_bands_values[0]:.0f}% total energy),
        {freq_band_names[Top_energy_bands_indices[1]]} ({freq_band_labels[Top_energy_bands_indices[1]]}Hz) ({100*Top_energy_bands_values[1]:.0f}% total energy),
        {freq_band_names[Top_energy_bands_indices[2]]} ({freq_band_labels[Top_energy_bands_indices[2]]}Hz) ({100*Top_energy_bands_values[2]:.0f}% total energy) 
          '''
    st.write(f"{text}.")


    # Find bands with largest number of peaks
    #metrics["peaks_per_band"] = {band:peak for band,peak in zip (freq_band_names,peaks_per_band)} 

    max_values_with_indices = [(i, arr.max()) for i, arr in enumerate(peaks_per_band)]
    sorted_max_values = sorted(max_values_with_indices, key=lambda x: x[1], reverse=True)
    peaks_per_band_top_indices = [index for index, _ in sorted_max_values[:3]] # hard code 3

    peaks_per_band_top_peak_num = [peaks_per_band[i] for i in peaks_per_band_top_indices]
    peaks_per_band_top_freq_band_names = [freq_band_names[i] for i in peaks_per_band_top_indices]
    peaks_per_band_top_freq_band_labels = [freq_band_labels[i] for i in peaks_per_band_top_indices]
    
    peaks_per_band_top_freq_band_combined_strings = [
        f"{name} ({label}Hz) ({peaks} peaks)"
        for peaks, name, label in zip(peaks_per_band_top_peak_num, peaks_per_band_top_freq_band_names, peaks_per_band_top_freq_band_labels)
    ]
    combined_string = ", ".join(peaks_per_band_top_freq_band_combined_strings)
    st.write(f"The bands with the most number of top 20 peaks are: {combined_string}.")
    
    st.plotly_chart(fig, use_container_width=True)


#==============================================================
#--------------------------------------------------------------
# Plots 4 and 5
#--------------------------------------------------------------
#==============================================================

with st.expander("High amplitude frequencies",expanded=True):
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

    st.write("Here you can see the frequency spectrum of the audio waveform. The peaks show the frequencies which are present most strongly in the sound.")
    text = f'''
    The top 5 peak frequencies are:
    {metrics[f"top_freq_1"]:,.0f}Hz ({metrics[f"top_freq_1_band"]}), 
    {metrics[f"top_freq_2"]:,.0f}Hz ({metrics[f"top_freq_2_band"]}), 
    {metrics[f"top_freq_3"]:,.0f}Hz ({metrics[f"top_freq_3_band"]}), 
    {metrics[f"top_freq_4"]:,.0f}Hz ({metrics[f"top_freq_4_band"]}), 
    {metrics[f"top_freq_5"]:,.0f}Hz ({metrics[f"top_freq_5_band"]}). 
    '''
    st.write(f"{text}")
    
    st.plotly_chart(fig)

# Chart - log log plot
    
    fig = go.Figure()

    # Add log FFT trace
    fig.add_trace(go.Scatter(x=xf, y=yf_db, mode='lines', line=dict(color='black', width=1), name='FFT'),)

    # Add a shape for the grey background under frequency 20
    fig.add_shape(
        type="rect",
        x0=0, x1=20, y0=0, y1=1,
        xref="x", yref="paper",
        fillcolor="lightgrey", 
        opacity=0.5,
        layer="below",
        line_width=0
    )


    # Add colored backgrounds for each frequency band
    for (lower_bound, upper_bound), color in zip(freq_bands, freq_band_colours):
        fig.add_shape(type='rect', x0=lower_bound, x1=upper_bound, y0=0, y1=1,
                      xref='x1', yref='paper', fillcolor=color, opacity=0.5, layer="below", line_width=0,
        )
    
    fig.update_layout(title_text="Frequency spectrum (log-log)", title_x=0.5, title_xanchor='center')
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Amplitude (dB)")
    fig.update_xaxes(range=[np.log10(1), np.log10(40000)]) # hard coded x limits
    fig.update_layout(height=400, width=700)

    fig.update_layout(     
        xaxis=dict(
        type="log", #Set the x-axis to a logarithmic scale
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
    st.plotly_chart(fig)
    
#==============================================================   
#--------------------------------------------------------------
# Plot 6
#--------------------------------------------------------------
#==============================================================

with st.expander("High energy frequencies",expanded=True):
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
    x=significant_frequency_target_percentages,
    y=significant_frequency_max_frequencies,
    mode='lines+markers',  # Shows both lines and points
    name='Significant Frequencies',
    line=dict(color='black'),
    marker=dict(color='black') 
    ))

    # Add colored backgrounds for each frequency band
    for (lower_bound, upper_bound), color in zip(freq_bands, freq_band_colours):
        fig.add_shape(type='rect', y0=lower_bound, y1=upper_bound, x0=0, x1=1,
                      yref='y1', xref='paper', fillcolor=color, opacity=0.5, layer="below", line_width=0,
        )

    fig.update_layout(title_text="The significant frequencies which contribute to the total audio energy", title_x=0.5, title_xanchor='center')
    fig.update_xaxes(title_text="Proportion of total energy (from 0 to 50%)")
    fig.update_yaxes(title_text="Significant frequencies (Hz)")
    fig.update_yaxes(range=[0, max(significant_frequency_max_frequencies) * 1.1])
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
    
    st.write("Here you can see the frequencies which cumultatively add up to the top 50% of the total energy in the audio signal.")

    text = f'''
        The mean of these values is {metrics["significant_frequency_max_frequencies_mean"]:,.0f}Hz
        and the median is {metrics["significant_frequency_max_frequencies_median"]:,.0f}Hz.
        The minimum value is {metrics["significant_frequency_max_frequencies_min"]:,.0f}Hz 
        and the maximum value is {metrics["significant_frequency_max_frequencies_max"]:,.0f}Hz
         '''
    st.write(f"{text}")
    
    st.plotly_chart(fig)


#==============================================================
st.header("Frequency characteristics over time", divider="gray")
#==============================================================


#==============================================================    
#--------------------------------------------------------------
# Plots 7 and 8
#--------------------------------------------------------------
#==============================================================
# Compute energy decay for each frequency band

# Get energy-time profile for each band 
energy_decay, times = compute_energy_decay_per_band(y, sr, freq_bands)

# Find bands with largest energy peaks
max_values_with_indices = [(i, arr.max()) for i, arr in enumerate(energy_decay)]
sorted_max_values = sorted(max_values_with_indices, key=lambda x: x[1], reverse=True)
energy_decay_top_indices = [index for index, _ in sorted_max_values[:3]] # hard code 3

energy_decay_top_freq_band_names = [freq_band_names[i] for i in energy_decay_top_indices]
energy_decay_top_freq_band_labels = [freq_band_labels[i] for i in energy_decay_top_indices]

energy_decay_top_freq_band_combined_strings = [
    f"{name} ({label})"
    for name, label in zip(energy_decay_top_freq_band_names, energy_decay_top_freq_band_labels)
]

with st.expander("High energy frequency bands",expanded=True):
    fig = go.Figure()
  
    # Define line widths and line styles based on thresholds
    line_widths = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5] 
    line_styles = assign_line_styles(normalized_band_energies, thresholds=[0.03, 0.06, 0.07, 0.1]) # non dot for top 3
    
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
                showlegend=True #if i == 0 else False
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

    st.write("Here you can see the amount energy in each of the key frequency bands and how this changes over time.")
    st.write(f"The top 3 bands with the largest peaks are {", ".join(energy_decay_top_freq_band_combined_strings)}.") 
    st.plotly_chart(fig)

    # Normalised energy

    normalized_energy_decay = [energy / np.max(energy) if np.max(energy) > 0 else np.zeros_like(energy) for energy in energy_decay] # Normalization
    smoothed_energy_decay = [gaussian_filter1d(energy, sigma=2) for energy in normalized_energy_decay]  # Smoothing
    line_styles = assign_line_styles(normalized_band_energies, thresholds=[0.05, 0.06, 0.07, 0.1])
    line_widths = [1.5 for _ in freq_bands]  # Uniform line widths
    times = np.linspace(0, 10, 1000)  # Example time array
    max_time_for_plotting = np.max(times)  # Maximum time for plotting
    
    fig = go.Figure()
    
    # Plot each frequency band as a smoothed, normalized energy decay curve
    for i, (low, high) in enumerate(freq_bands):
        label = f"{low}-{high} Hz"
        fig.add_trace(
            go.Scatter(
                x=times,
                y=smoothed_energy_decay[i],
                mode='lines',
                line=dict(color=freq_band_colours[i], width=line_widths[i], dash=line_styles[i]),
                name=label
            )
        )
    
    # Apply log scaling to the y-axis
    fig.update_yaxes(type="log")

    fig.update_layout(title_text="The normalised energy decay over time", title_x=0.5, title_xanchor='center')
    fig.update_xaxes(title_text='Time (s)', range=[0, max_time_for_plotting])
    fig.update_yaxes(title_text='Normalised energy (log)')
    fig.update_layout(height=400, width=700)
    
    # Formatting axes and grid
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
    
    st.plotly_chart(fig, use_container_width=True)

#==============================================================    
#--------------------------------------------------------------
# Plot 9
#--------------------------------------------------------------
#==============================================================
with st.expander("High amplitude frequencies and fundamental pitches over time",expanded=True):

    top_n = 1
    dominant_freqs, times = compute_dominant_frequencies(y, sr, n_fft=2048, hop_length=128, top_n=top_n)

    # Adjust dominant_freqs: Ensure it's 2D, even if top_n = 1
    if len(dominant_freqs.shape) == 1:  # If it's a 1D array
        dominant_freqs = dominant_freqs[:, np.newaxis]  # Add an extra dimension

    # Plot the dominant frequency over time
    colors = ['black', 'white', 'darkgrey', 'lightgrey', 'whitesmoke']

    fig = go.Figure()
    
    # Add coloured backgrounds for frequency bands
    for (lower_bound, upper_bound), color in zip(freq_bands, freq_band_colours):
        fig.add_shape(
            type='rect',
            x0=0, x1=times[-1], # all the way to the end
            y0=lower_bound, y1=upper_bound,
            fillcolor=color,
            opacity=0.5,
            layer="below",  # Add this line to ensure shapes are below traces
            line_width=0
        )
    
    # Reverse the order of prominant frequencies for plotting
    for i in reversed(range(dominant_freqs.shape[1])):  # Iterate through columns
        fig.add_trace(
            go.Scatter(
                x=times,
                y=dominant_freqs[:, i],  # Select column i
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),  # Cycle colors if needed
                name=f'Prominent Freq {i+1}'
            )
        )
    
     

    fig.update_layout(title_text="Prominant frequencies over time", title_x=0.5, title_xanchor='center')
    fig.update_xaxes(title_text='Time (s)', showgrid=True, gridcolor='lightgrey') #range=[0, max_time_for_plotting]
    ylims = [0, np.nanmax(dominant_freqs)]
    fig.update_yaxes(title_text='Frequency (Hz)', range=ylims, showgrid=True, gridcolor='lightgrey')
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

    st.write("Here you can see how the peak frequencies evolve over time. The most prominent frequency peaks are identified within small, overlapping time windows, and this process is repeated continuously throughout the duration of the audio file")

    # frequency stats

    st.write(f"The mean frequency across the time span is {np.mean(dominant_freqs):,.0f}Hz and the median is {np.median(dominant_freqs):,.0f}Hz.")

    from sklearn.cluster import KMeans

    col_amp1, col_amp2 = st.columns(2)

    with col_amp2:
        n_clusters = st.number_input('choose number of clusters', min_value=2, max_value=6,value=3)
        
    with col_amp1:
        # Reshape data for clustering
        data = dominant_freqs.reshape(-1, 1)
        # Apply KMeans clustering to identify stable groups
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        centroids = kmeans.cluster_centers_
        centroid_values = ", ".join(f"{value:,.0f}Hz" for value in centroids.flatten())
        st.write(f"The top {n_clusters} frequency clusters: {centroid_values}")

    st.plotly_chart(fig, use_container_width=True)

#==============================================================    
#--------------------------------------------------------------
# Plot 10
#--------------------------------------------------------------
#==============================================================
    fig = go.Figure()
    
    # Compute pitches
    pitches, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitches = np.array(pitches)
    pitches_times = librosa.frames_to_time(np.arange(len(pitches)), sr=sr)
    
    # Plot fundamental pitches as markers
    fig.add_trace(
        go.Scatter(
            x=pitches_times,
            y=pitches,
            mode='markers',
            marker=dict(symbol='star', color='black', size=6),
            name='Fundamental pitches (Hz)'
        )
    )

    # Add coloured backgrounds for frequency bands
    for (lower_bound, upper_bound), color in zip(freq_bands, freq_band_colours):
        fig.add_shape(
            type='rect',
            x0=0, x1=times[-1], # all the way to the end
            y0=lower_bound, y1=upper_bound,
            fillcolor=color,
            opacity=0.5,
            layer="below",  # Add this line to ensure shapes are below traces
            line_width=0
        )
    
    fig.update_layout(title_text="Fundamental pitches over time", title_x=0.5, title_xanchor='center')
    fig.update_xaxes(title_text='Time (s)', showgrid=True, gridcolor='lightgrey') #range=[0, max_time_for_plotting]
    ylims = [0, np.nanmax(pitches)]
    fig.update_yaxes(title_text='Frequency (Hz)', range=ylims, showgrid=True, gridcolor='lightgrey')
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

    st.write("""Here you can see if any fundamental pitches are detected (i.e. the lowest frequency when harmonic frequencies are present as multiples of it).""")
    
    
    # Reshape data for clustering
    valid_pitches = pitches[~np.isnan(pitches)]

    st.write(f"The mean fundamental pitch across the time span is {np.mean(valid_pitches):,.0f}Hz and the median is {np.mean(valid_pitches):,.0f}Hz.")

    data = valid_pitches.reshape(-1, 1)

    col_fun1, col_fun2 = st.columns(2)

    with col_fun2:
        n_clusters = st.number_input('choose number of clusters', min_value=2, max_value=6,value=2)

    with col_fun1:     
        if len(valid_pitches) >= n_clusters:
            # Reshape data for clustering
            data = valid_pitches.reshape(-1, 1)
        
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
            centroids = kmeans.cluster_centers_
        
            centroid_values = ", ".join(f"{value:,.0f}Hz" for value in centroids.flatten())
            st.write(f"The top {n_clusters} fundamental frequency clusters are {centroid_values}")

    st.plotly_chart(fig, use_container_width=True)
