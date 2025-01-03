import streamlit as st
import pyaudio
import numpy as np
import wave
import os
import time

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

# Function to list audio devices
def list_audio_devices():
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        devices.append((i, info["name"]))
    p.terminate()
    return devices

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

    # Persistent state for buttons
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    mode = st.sidebar.radio("Select Mode", ("single_speaker", "multiple_speakers"))
    reduction_level = st.sidebar.slider(
        "Noise Reduction Level", 0.0, 1.0, 0.8, 0.1
    )

    # Display audio device options
    devices = list_audio_devices()
    input_device = st.sidebar.selectbox(
        "Select Input Device", devices, format_func=lambda x: f"{x[1]}"
    )
    output_device = st.sidebar.selectbox(
        "Select Output Device", devices, format_func=lambda x: f"{x[1]}"
    )

    start_button = st.sidebar.button("▶️ Start Noise Cancellation")
    stop_button = st.sidebar.button("⏹ Stop Noise Cancellation")

    if start_button:
        st.session_state.is_running = True

    if stop_button:
        st.session_state.is_running = False

    # Output directory for processed audio
    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_audio.wav")

    if st.session_state.is_running:
        st.write("🔊 Noise cancellation is running...")

        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                input_device_index=input_device[0],
                output_device_index=output_device[0],
            )

            frames = []

            try:
                while st.session_state.is_running:
                    raw_data = stream.read(CHUNK, exception_on_overflow=False)
                    input_audio = np.frombuffer(raw_data, dtype=np.int16)
                    processed_audio = noise_cancellation(
                        input_audio, mode=mode, reduction_level=reduction_level
                    )
                    stream.write(processed_audio.tobytes())
                    frames.append(processed_audio.tobytes())

                    # Allow Streamlit to update the UI
                    time.sleep(0.01)
            except Exception as e:
                st.error(f"Error during streaming: {e}")
            finally:
                stream.stop_stream()
                stream.close()
        except Exception as e:
            st.error(f"Error initializing audio stream: {e}")
        finally:
            p.terminate()

        # Save the processed audio
        if frames:
            wf = wave.open(output_file, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
            wf.close()
            st.success(f"✅ Processed audio saved to {output_file}")

    # Playback the saved audio
    if os.path.exists(output_file) and not st.session_state.is_running:
        st.subheader("🎵 Processed Audio Playback")
        with open(output_file, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")
            st.success("Audio playback ready. Use the controls above to listen.")

if __name__ == "__main__":
    main()
