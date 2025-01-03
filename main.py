import streamlit as st
import pyaudio
import numpy as np
import wave
import os

# Constants for audio processing
CHUNK = 1024  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate in Hz

# Noise cancellation function
def noise_cancellation(input_audio, noise_profile, reduction_level=0.8):
    """
    Perform noise cancellation on the input audio.

    :param input_audio: Numpy array of audio samples.
    :param noise_profile: Cached noise profile (mean or median).
    :param reduction_level: Strength of noise reduction (0 to 1).
    :return: Processed audio samples.
    """
    processed_audio = input_audio - reduction_level * noise_profile
    return np.clip(processed_audio, -32768, 32767).astype(np.int16)

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

    if start_button:
        st.session_state.is_running = True

    if stop_button:
        st.session_state.is_running = False

    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_audio.wav")

    if st.session_state.is_running:
        st.write("🔊 Noise cancellation is running...")

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=True,
            frames_per_buffer=CHUNK,
        )

        frames = []
        noise_profile = None

        try:
            for _ in range(int(RATE / CHUNK * 10)):  # Limit runtime for testing
                raw_data = stream.read(CHUNK, exception_on_overflow=False)
                input_audio = np.frombuffer(raw_data, dtype=np.int16)

                # Calculate noise profile once or periodically
                if noise_profile is None or _ % 10 == 0:
                    if mode == "single_speaker":
                        noise_profile = np.mean(input_audio)
                    elif mode == "multiple_speakers":
                        noise_profile = np.median(input_audio)

                processed_audio = noise_cancellation(
                    input_audio, noise_profile, reduction_level=reduction_level
                )
                stream.write(processed_audio.tobytes())
                frames.append(processed_audio.tobytes())

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open(output_file, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
            wf.close()

            st.success(f"✅ Processed audio saved to {output_file}")

    if os.path.exists(output_file) and not st.session_state.is_running:
        st.subheader("🎵 Processed Audio Playback")
        with open(output_file, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")
            st.success("Audio playback ready. Use the controls above to listen.")

if __name__ == "__main__":
    main()
