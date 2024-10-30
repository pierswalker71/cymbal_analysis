
# imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
import numpy as np

from pydub import AudioSegment
import librosa
import librosa.display

#import io
#from IPython.display import Audio

from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis
from scipy.stats import entropy




st.title("Cymbal analysis")
st.write("This app loads cymbal audio files and presents their key features")

# Generate sample data for plotting
x = np.linspace(0, 10, 100)

# Create a 2x2 grid of columns
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Plot 1
with col1:
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x))
    ax.set_title("Sine Wave")
    st.pyplot(fig)

# Plot 2
with col2:
    fig, ax = plt.subplots()
    ax.plot(x, np.cos(x))
    ax.set_title("Cosine Wave")
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





# Generate and display multiple plots in a vertical layout
for i in range(4):
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x + i))
    ax.set_title(f"Plot {i+1}")
    st.pyplot(fig)




with st.expander("View First Row of Plots"):
    col1, col2 = st.columns(2)
    with col1:
        # Plot 1
        fig, ax = plt.subplots()
        ax.plot(x, np.sin(x))
        st.pyplot(fig)

    with col2:
        # Plot 2
        fig, ax = plt.subplots()
        ax.plot(x, np.cos(x))
        st.pyplot(fig)
