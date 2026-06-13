import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import librosa
import librosa.display

# -------------------- HELPER FUNCTIONS --------------------

def preemphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def hamming_window(frames, frame_length):
    return frames * np.hamming(frame_length)

def power_spectrum(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    return pow_frames

def mel_filterbank(sample_rate, NFFT, nfilt=26):
    mel_min = 0
    mel_max = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(mel_min, mel_max, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return fbank

def extract_mfcc(signal, sample_rate, frame_size=0.025, frame_stride=0.01, 
                 NFFT=512, nfilt=26, num_ceps=13, alpha=0.97):
    emphasized_signal = preemphasis(signal, alpha)
    frames = framing(emphasized_signal, sample_rate, frame_size, frame_stride)
    frame_length = int(round(frame_size * sample_rate))
    frames = hamming_window(frames, frame_length)
    pow_frames = power_spectrum(frames, NFFT)
    fbank = mel_filterbank(sample_rate, NFFT, nfilt)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)]
    return mfcc, pow_frames, fbank, frames

# -------------------- STREAMLIT APP --------------------

# Unique student information
st.sidebar.info("Submitted by: Sachi Bhagat | Roll No: 2311401205")

# App title
st.title("MFCC Feature Extraction System for Speaker Recognition")

# Initialize session state
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = None

# Question selection
st.header("Select a Research Question:")
col1, col2 = st.columns(2)

with col1:
    if st.button("1. Spectrogram vs. MFCC Interpretation"):
        st.session_state.selected_question = 1
    if st.button("2. Frame Size and Overlap Analysis"):
        st.session_state.selected_question = 2

with col2:
    if st.button("3. Mel Filter Bank Customization"):
        st.session_state.selected_question = 3
    if st.button("4. Sampling Rate Adjustment Study"):
        st.session_state.selected_question = 4

# Upload audio
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

# Main logic
if uploaded_file is not None:
    signal, sample_rate = librosa.load(uploaded_file, sr=None)
    if len(signal.shape) > 1:
        signal = librosa.to_mono(signal)
    duration = len(signal) / sample_rate
    st.write(f"Sample Rate: {sample_rate} Hz | Duration: {duration:.2f} sec")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sample_rate, ax=ax)
    ax.set_title("Time Domain Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    if st.session_state.selected_question == 1:
        st.header("Spectrogram vs. MFCC Interpretation")
        frame_size = st.sidebar.slider("Frame size (s)", 0.01, 0.05, 0.025)
        frame_stride = st.sidebar.slider("Frame stride (s)", 0.005, 0.02, 0.01)
        nfilt = st.sidebar.slider("Number of Mel filters", 10, 40, 26)

        mfcc, _, _, _ = extract_mfcc(signal, sample_rate, frame_size, frame_stride, nfilt=nfilt)

        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=nfilt)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title('Mel Spectrogram')
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(mfcc.T, sr=sample_rate, x_axis='time', ax=ax)
        ax.set_title('MFCC Heatmap')
        st.pyplot(fig)

    elif st.session_state.selected_question == 2:
        st.header("Frame Size and Overlap Analysis")
        frame_size = st.sidebar.slider("Frame size (ms)", 10, 50, 25)
        frame_overlap = st.sidebar.slider("Frame overlap (%)", 10, 90, 50)
        nfilt = st.sidebar.slider("Number of Mel filters", 10, 40, 26)

        frame_size_sec = frame_size / 1000
        frame_stride_sec = frame_size_sec * (1 - frame_overlap / 100)

        mfcc, _, _, _ = extract_mfcc(signal, sample_rate, frame_size_sec, frame_stride_sec, nfilt=nfilt)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(mfcc.T, sr=sample_rate, x_axis='time', ax=ax)
        ax.set_title(f'MFCC (Frame: {frame_size}ms, Overlap: {frame_overlap}%)')
        st.pyplot(fig)

    elif st.session_state.selected_question == 3:
        st.header("Mel Filter Bank Customization")
        nfilt = st.sidebar.slider("Number of Mel filters", 10, 40, 26)
        frame_size = st.sidebar.slider("Frame size (s)", 0.01, 0.05, 0.025)
        frame_stride = st.sidebar.slider("Frame stride (s)", 0.005, 0.02, 0.01)

        mfcc, _, fbank, _ = extract_mfcc(signal, sample_rate, frame_size, frame_stride, nfilt=nfilt)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(fbank, aspect='auto', origin='lower')
        ax.set_title(f'Mel Filter Bank ({nfilt} filters)')
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(mfcc.T, sr=sample_rate, x_axis='time', ax=ax)
        ax.set_title(f'MFCC Heatmap ({nfilt} filters)')
        st.pyplot(fig)

    elif st.session_state.selected_question == 4:
        st.header("Sampling Rate Adjustment Study")
        new_sr = st.sidebar.selectbox("Target Sample Rate", [8000, 16000, 22050, 44100], index=1)
        frame_size = st.sidebar.slider("Frame size (s)", 0.01, 0.05, 0.025)
        frame_stride = st.sidebar.slider("Frame stride (s)", 0.005, 0.02, 0.01)
        nfilt = st.sidebar.slider("Number of Mel filters", 10, 40, 26)

        resampled_signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=new_sr)

        st.write(f"Original SR: {sample_rate} Hz | New SR: {new_sr} Hz")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        librosa.display.waveshow(signal, sr=sample_rate, ax=ax1)
        ax1.set_title("Original Signal")
        librosa.display.waveshow(resampled_signal, sr=new_sr, ax=ax2)
        ax2.set_title("Resampled Signal")
        st.pyplot(fig)

        mfcc_orig, _, _, _ = extract_mfcc(signal, sample_rate, frame_size, frame_stride, nfilt=nfilt)
        mfcc_resampled, _, _, _ = extract_mfcc(resampled_signal, new_sr, frame_size, frame_stride, nfilt=nfilt)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        librosa.display.specshow(mfcc_orig.T, sr=sample_rate, x_axis='time', ax=ax1)
        ax1.set_title("Original MFCC")
        librosa.display.specshow(mfcc_resampled.T, sr=new_sr, x_axis='time', ax=ax2)
        ax2.set_title("Resampled MFCC")
        st.pyplot(fig)

elif st.session_state.selected_question is not None:
    st.warning("Please upload a WAV file to proceed.")
