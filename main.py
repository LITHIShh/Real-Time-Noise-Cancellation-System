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
    """
    if mode == "single_speaker":
        noise_profile = np.mean(input_audio)  # Estimate noise as mean
    elif mode == "multiple_speakers":
        noise_profile = np.median(input_audio)  # Estimate noise as median
    else:
        raise ValueError("Invalid mode. Choose 'single_speaker' or 'multiple_speakers'.")

    processed_audio = input_audio - reduction_level * noise_profile
    return np.clip(processed_audio, -32768, 32767).astype(np.int16)

# Streamlit app
def main():
    st.set_page_config(page_title="Noise Cancellation", page_icon="🎧")

    st.title("🎧 Real-Time Noise Cancellation System")

    # Sidebar controls
    mode = st.sidebar.radio("Mode", ["single_speaker", "multiple_speakers"])
    reduction_level = st.sidebar.slider("Reduction Level", 0.0, 1.0, 0.8, 0.1)

    start_button = st.sidebar.button("Start")
    stop_button = st.sidebar.button("Stop")

    # Ensure `is_running` exists in the session state
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    # Manage start and stop states
    if start_button:
        st.session_state.is_running = True
    if stop_button:
        st.session_state.is_running = False

    # Initialize frames to ensure no UnboundLocalError
    frames = []

    # Directory for saving processed audio
    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_audio.wav")

    if st.session_state.is_running:
        st.write("🔊 Noise cancellation is running...")

        try:
            # Set up PyAudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
            )

            try:
                while st.session_state.is_running:
                    raw_data = stream.read(CHUNK, exception_on_overflow=False)
                    input_audio = np.frombuffer(raw_data, dtype=np.int16)
                    processed_audio = noise_cancellation(
                        input_audio, mode=mode, reduction_level=reduction_level
                    )
                    stream.write(processed_audio.tobytes())
                    frames.append(processed_audio.tobytes())

                    # Allow Streamlit UI to update
                    time.sleep(0.01)
            except Exception as stream_error:
                st.error(f"Streaming error: {stream_error}")
            finally:
                stream.stop_stream()
                stream.close()
        except Exception as init_error:
            st.error(f"Initialization error: {init_error}")
        finally:
            p.terminate()

        # Save frames if any data was recorded
        if frames:
            try:
                with wave.open(output_file, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b"".join(frames))
                st.success(f"Processed audio saved to {output_file}")
            except Exception as save_error:
                st.error(f"Error saving audio: {save_error}")
    else:
        st.write("🔇 Noise cancellation is stopped.")

    # Play back processed audio
    if os.path.exists(output_file) and not st.session_state.is_running:
        st.subheader("🎵 Processed Audio Playback")
        with open(output_file, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")

if __name__ == "__main__":
    main()
