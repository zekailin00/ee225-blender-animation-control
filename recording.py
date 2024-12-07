import pyaudio
import wave

def record_audio(filename="output.wav", duration=5, sample_rate=16000, channels=1):
    # Audio format
    FORMAT = pyaudio.paInt16
    CHUNK = 1024

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=FORMAT,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=CHUNK)

    print(f"Recording for {duration} seconds...")

    frames = []

    # Record in chunks
    for _ in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the audio data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {filename}")

if __name__ == "__main__":
    record_audio()