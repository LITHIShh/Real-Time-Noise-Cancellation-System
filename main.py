import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import os
import time

# Constants for audio processing
CHUNK = 1024  # Number of audio frames per buffer
FORMAT = np.int16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate in Hz

# Noise cancellation function
def noise_cancellation(input_audio, mode="single_speaker", reduction_level=0.8):
    if mode == "single_speaker":
        noise_profile = np.mean(input_audio)
    elif mode == "multiple_speakers":
        noise_profile = np.median(input_audio)
    else:
        raise ValueError("Invalid mode. Choose 'single_speaker' or 'multiple_speakers'.")

    processed_audio = input_audio - reduction_level * noise_profile
    return np.clip(processed_audio, -32768, 32767).astype(np.int16)

# Function to find valid audio devices
def find_valid_device():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Check if the device has input channels
            return i  # Return the first valid input device ID
    return None  # No valid device found

def find_output_device():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:  # Check if the device has output channels
            return i  # Return the first valid output device ID
    return None  # No valid device found

# Streamlit app
def main():
    st.set_page_config(
        page_title="Noise Cancellation",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎧 Real-Time Noise Cancellation System")

    st.sidebar.title("🔧 Controls")
    st.sidebar.markdown(
        """Control the real-time noise cancellation process using the buttons below."""
    )

    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    mode = st.sidebar.radio("Select Mode", ("single_speaker", "multiple_speakers"))
    reduction_level = st.sidebar.slider(
        "Noise Reduction Level", 0.0, 1.0, 0.8, 0.1
    )
    start_button = st.sidebar.button("▶️ Start Noise Cancellation")
    stop_button = st.sidebar.button("⏹ Stop Noise Cancellation")

    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_audio.wav")

    if start_button:
        st.session_state.is_running = True

    if stop_button:
        st.session_state.is_running = False

    if st.session_state.is_running:
        st.write("🔊 Noise cancellation is running...")

        frames = []  # Initialize frames here to avoid UnboundLocalError
        input_device_id = find_valid_device()  # Get a valid input device ID
        output_device_id = find_output_device()  # Get a valid output device ID

        if input_device_id is not None and output_device_id is not None:
            st.write(f"Using input device ID: {input_device_id} and output device ID: {output_device_id}")

            def callback(indata, outdata, frames, time, status):
                if status:
                    st.error(status)
                input_audio = indata[:, 0]
                processed_audio = noise_cancellation(
                    input_audio, mode=mode, reduction_level=reduction_level
                )
                outdata[:, 0] = processed_audio
                frames.append(processed_audio.tobytes())

            with sd.Stream(callback=callback, channels=CHANNELS, samplerate=RATE,
                            device=(input_device_id, output_device_id)):
                while st.session_state.is_running:
                    time.sleep(0.01)

            with wave.open(output_file, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames))
                st.success(f"✅ Processed audio saved to {output_file}")
        else:
            if input_device_id is None:
                st.error("No valid audio input devices found.")
            if output_device_id is None:
                st.error("No valid audio output devices found.")

    if os.path.exists(output_file) and not st.session_state.is_running:
        st.subheader("🎵 Processed Audio Playback")
        with open(output_file, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")
            st.success("Audio playback ready. Use the controls above to listen.")

if __name__ == "__main__":
    main()
