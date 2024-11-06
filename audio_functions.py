import librosa
import librosa.display

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
