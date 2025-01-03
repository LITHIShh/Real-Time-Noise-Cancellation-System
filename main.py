import streamlit as st
import numpy as np
import pyaudio
import webrtcvad
import os
import time
import cv2

# Constants for audio processing
CHUNK = 1024  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate in Hz

# Noise cancellation function
def noise_cancellation(input_audio, mode="single_speaker", reduction_level=0.8):
    """
    Perform noise cancellation on the input audio.

    :param input_audio: Numpy array of audio samples.
    :param mode: 'single_speaker' or 'multiple_speakers'.
    :param reduction_level: Strength of noise reduction (0 to 1).
    :return: Processed audio samples.
    """
    if mode == "single_speaker":
        noise_profile = np.mean(input_audio)  # Estimate noise as mean
    elif mode == "multiple_speakers":
        noise_profile = np.median(input_audio)  # Estimate noise as median
    else:
        raise ValueError("Invalid mode. Choose 'single_speaker' or 'multiple_speakers'.")

    processed_audio = input_audio - reduction_level * noise_profile
    return np.clip(processed_audio, -32768, 32767).astype(np.int16)

# Streamlit app using streamlit-webrtc
def main():
    st.set_page_config(
        page_title="Noise Cancellation",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎧 Real-Time Noise Cancellation System using streamlit-webrtc")

    st.sidebar.title("🔧 Controls")
    st.sidebar.markdown(
        """Control the real-time noise cancellation process using the buttons below."""
    )

    # Persistent state for buttons
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    mode = st.sidebar.radio("Select Mode", ("single_speaker", "multiple_speakers"))
    reduction_level = st.sidebar.slider(
        "Noise Reduction Level", 0.0, 1.0, 0.8, 0.1
    )

    # Start/Stop buttons
    start_button = st.sidebar.button("▶️ Start Noise Cancellation")
    stop_button = st.sidebar.button("⏹ Stop Noise Cancellation")

    if start_button:
        st.session_state.is_running = True

    if stop_button:
        st.session_state.is_running = False

    # WebRTC Callback for streaming
    def audio_callback(frame):
        if st.session_state.is_running:
            input_audio = np.frombuffer(frame.data, dtype=np.int16)
            processed_audio = noise_cancellation(input_audio, mode, reduction_level)
            return processed_audio.tobytes()
        return frame.data

    rtc_config = webrtcvad.VadConfig()
    rtc_config.mode = webrtcvad.VadNormal  # Use normal VAD mode for noise detection

    rtc_context = webrtcvad.Vad(rtc_config)

    st.write("🔊 Noise cancellation is running...")

    # Streamlit-webrtc
    webrtc_ctx = webrtc.Context(
        audio_config=webrtc.AudioConfig(channels=CHANNELS, sample_rate=RATE),
        audio_processor_factory=lambda: webrtcvad.VadProcessor(rtc_context, audio_callback),
    )
    
    if webrtc_ctx.running():
        st.write("🎵 Real-time audio processing with noise cancellation in progress...")

    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_audio.wav")

    # Save audio file
    def save_audio(frames):
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        st.success(f"✅ Processed audio saved to {output_file}")

    # Playback the saved audio
    if os.path.exists(output_file) and not st.session_state.is_running:
        st.subheader("🎵 Processed Audio Playback")
        with open(output_file, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")
            st.success("Audio playback ready. Use the controls above to listen.")

if __name__ == "__main__":
    main()
