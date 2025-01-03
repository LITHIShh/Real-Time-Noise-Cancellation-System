# Real-Time Noise Cancellation System

## Overview
This project implements a **real-time noise cancellation system** using Python and Streamlit. The system processes audio in two distinct scenarios:

1. **Single Speaker Scenario**: Enhances the audio of one primary speaker while treating other voices and background noises as interference to minimize.
2. **Multiple Speaker Scenario**: Preserves multiple speakers' voices and filters out environmental noise (e.g., white noise, workplace background noise, or vehicle noise).

The system allows users to control noise cancellation parameters, process real-time audio, and playback processed audio via a web-based interface.

---

## Features
- **Real-Time Audio Processing**: Processes audio input in real time using PyAudio.
- **Noise Cancellation Modes**:
  - *Single Speaker*: Isolates one speaker by using the mean noise profile.
  - *Multiple Speakers*: Preserves all speakers' voices and reduces background noise using the median noise profile.
- **Dynamic Controls**: Adjustable noise reduction level and mode selection via Streamlit.
- **Audio Playback**: Save and playback processed audio directly from the web interface.
- **Interactive UI**: Simple and intuitive controls to manage noise cancellation.

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install streamlit pyaudio numpy
   ```

3. Run the application:
   ```bash
   streamlit run noise_cancellation.py
   ```

4. Open the URL displayed in the terminal to access the web interface.

---

## Usage

1. Launch the Streamlit app as described above.
2. Use the **sidebar controls** to:
   - Select the noise cancellation mode (`single_speaker` or `multiple_speakers`).
   - Adjust the noise reduction level (0.0 to 1.0).
   - Start or stop the noise cancellation process.
3. Once the process is complete, playback the processed audio using the interface.

---

## Code Structure

- **`noise_cancellation.py`**: Main script for real-time audio processing and the Streamlit web interface.
- **Noise Cancellation Functionality**:
  - The `noise_cancellation` function performs noise reduction based on the selected mode.
  - Uses numpy to calculate and subtract the noise profile from the audio input.
- **Streamlit Integration**:
  - UI elements for mode selection, parameter adjustment, and process controls.
  - Audio playback functionality using `st.audio`.
- **Output Directory**: Processed audio is saved in the `output_audio` folder.

---

## Requirements

- Python 3.7+
- Streamlit
- PyAudio
- Numpy

---

## Troubleshooting

- **Microphone Permissions**:
  Ensure the application has permissions to access the microphone.
- **Audio Overflows**:
  If an `exception_on_overflow` error occurs, adjust the buffer size or ensure the system is not overloaded.
- **Playback Issues**:
  If the processed audio does not play, verify the `output_audio` directory and check the audio file.

---

## Future Enhancements

1. **Advanced Noise Cancellation**:
   Implement spectral gating or Wiener filtering for improved noise reduction.
2. **Visualization**:
   Add visualizations of waveforms for input and processed audio.
3. **Cross-Platform Compatibility**:
   Test and optimize the application for Windows, macOS, and Linux.
4. **Performance Optimization**:
   Optimize the real-time processing to reduce latency.

---

## Acknowledgments
- **PyAudio**: For audio stream handling.
- **Streamlit**: For building an interactive web application.
- **Numpy**: For efficient numerical computations.

---



