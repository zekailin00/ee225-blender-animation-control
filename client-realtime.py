import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue
import time

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config, HubertModel

import socket
import json
from playsound import playsound
import threading

from recording import record_audio
from utils import load_audio, init_torch_device


# Configuration
sample_rate = 16000  # 16 kHz
channels = 1  # Mono audio
chunk_size = 1600  # Number of samples per 0.1 second
window_duration = 2  # Seconds
samples_per_window = int(sample_rate * window_duration)

# Queue for producer-consumer model
audio_queue = Queue(maxsize=10)  # Buffer for 10 windows

def producer():
    """Record audio in real time and produce 2-second windows."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    buffer = np.zeros(samples_per_window, dtype=np.float32)  # Circular buffer

    try:
        frame_timestamp = 0
        while True:
            frame_timestamp = time.time()
            # Read a chunk of audio data
            data = stream.read(chunk_size, exception_on_overflow=False)

            # Convert byte data to numpy array
            audio_samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            # Normalize audio data
            audio_samples /= np.iinfo(np.int16).max

            # Update the circular buffer
            buffer = np.roll(buffer, -chunk_size)
            buffer[-chunk_size:] = audio_samples

            # Send the current window to the queue
            if not audio_queue.full():
                audio_queue.put(buffer.copy())
                end_time = time.time()
                print("Produce buffer:", end_time - frame_timestamp, end_time)
                frame_timestamp = end_time
            else:
                print("Warning: queue is full")
    except KeyboardInterrupt:
        print("\nProducer stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def consumer():
    """Consume audio windows from the queue and visualize them."""
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    x = np.linspace(0, window_duration, samples_per_window)  # Time axis for 2 seconds
    line, = ax.plot(x, np.zeros(samples_per_window), label="Audio Signal")
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, window_duration)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Real-Time Audio Visualization")
    ax.legend()

    try:
        while True:
            # Check if there's new data in the queue
            if not audio_queue.empty():
                buffer = audio_queue.get()

                start_time = time.time()
                send_shapekey(compute_ema(buffer))
                print("\t\t\t\t Consumer cost: ", time.time() - start_time, start_time)


                # Update the plot
                line.set_ydata(buffer)
                fig.canvas.draw()
                fig.canvas.flush_events()
            else:
                time.sleep(0.01)  # Avoid busy-waiting
    except KeyboardInterrupt:
        print("\nConsumer stopped by user.")
    finally:
        plt.ioff()
        plt.show()


class SimpleModel(nn.Module):
    def __init__(self, input_size=768, output_size=30):
        super(SimpleModel, self).__init__()
        # Single linear layer
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # Forward pass: apply linear layer
        return self.linear(x)
    
    def predict(self, x):
        # Get predictions without gradient tracking (useful for inference)
        with torch.no_grad():
            return self.forward(x)

data = {
    "TT": 0.0,
    "TB": 0.0,
    "TD": 0.0,
    "AU22 lipsFunnel": 0.0,
    "AU18 lipsProtude": 0.0,
    "AU23 lipsKiss": 0.0,
    "AU14 jawStrained": 0.0,
    "AU12 smile": 0.0,
    "AU20 lipStrained": 0.0,
    "AU10 upper lip raiser": 0.0,
    "AU16 lower lip depressor": 0.0,
    "AU17 hmmm": 0.0,
    "AU28 lipsBite": 0.0,
    "AU27 JawOpen": 0.0
}

LABEL = {
    "TD": 3 * 0,
    "TB": 3 * 1,
    "BR": 3 * 2,
    "BL": 3 * 3,
    "TT": 3 * 4,
    "UL": 3 * 5,
    "LC": 3 * 6,
    "LL": 3 * 7,
    "LI": 3 * 8,
    "LJ": 3 * 9,
    "x": 0,
    "y": 1,
    "z": 2
}

def compute_ema(buffer):
    start_time = time.time()
    buffer = torch.Tensor(buffer).unsqueeze(0).to(device)
    # print("buffer.shape", buffer.shape)
    feat = model(buffer , output_hidden_states=True)
    x_features = feat.hidden_states[24].detach()[0]
    # print("[features] x_features.shape:", x_features.shape)

    y_pred = []
    if (x_features.shape[0] > 95):
        for i in range(90, 95):
            # print("x_features[i].shape", x_features[i].shape)
            result = ema_predictor(x_features[i])
            # print("result.shape", result.shape)
            y_pred.append(result.detach().cpu().numpy())
    
    y_pred = np.array(y_pred)
    # print("y_pred.shape", y_pred.shape)
    print("\t EMA cost:", time.time() - start_time, start_time)
    return y_pred[0]


def EMA_to_shapekeys(data, EMA):
    start_time = time.time()
    data["TD"]                          = EMA[LABEL["TD"] + LABEL["z"]] / 10.0 * 1.5
    data["TB"]                          = (EMA[LABEL["TB"] + LABEL["z"]] - EMA[LABEL["TD"] + LABEL["z"]] - 8.0) / -10.0 * 1.5
    data["TT"]                          = (EMA[LABEL["TT"] + LABEL["z"]] - EMA[LABEL["TB"] + LABEL["z"]] - EMA[LABEL["TD"] + LABEL["z"]]) / 10.0 * 1.5
    data["AU22 lipsFunnel"]             = 0 #EMA[5*3+1]
    data["AU18 lipsProtude"]            = 0 #EMA[5*3+1]
    data["AU23 lipsKiss"]               = (EMA[LABEL["LC"]] + 4) / 10.0 
    data["AU14 jawStrained"]            = (EMA[LABEL["LC"]] + 3) / -5.0
    data["AU12 smile"]                  = 0 #EMA[5*3+1]
    data["AU20 lipStrained"]            = 0 #EMA[5*3+1]
    data["AU10 upper lip raiser"]       = (EMA[LABEL["UL"] + LABEL["z"]] - 1) / 5.0
    data["AU16 lower lip depressor"]    = (EMA[LABEL["LL"] + LABEL["z"]] - EMA[LABEL["LI"] + LABEL["z"]] - 1) / -5.0
    data["AU17 hmmm"]                   = (EMA[LABEL["UL"] + LABEL["z"]] - EMA[LABEL["LI"] + LABEL["z"]] - 1) / 5.0
    data["AU28 lipsBite"]               = 0 #EMA[5*3+1]
    data["AU27 JawOpen"]                = (EMA[LABEL["LI"] + LABEL["z"]] + 20.0)/ -30.0
    print("\t\t shapekey cost:", time.time() - start_time, start_time)

def send_shapekey(ema):
    start_time = time.time()
    # print("ema.shape", ema.shape)
    EMA_to_shapekeys(data, ema)
    client.sendall(json.dumps(data).encode("utf-8"))
    print("\t\t\t socket cost:", time.time() - start_time, start_time)


host = "localhost"
port = 12345

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

# Load wav2vec model
device = init_torch_device(0)
model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
ema_predictor = torch.load("/Users/zekailin00/Git/SpeechDrivenTongueAnimation/hubert-large-ll60k-model.pth")
model.to(device)
ema_predictor.to(device)
model.eval()
ema_predictor.eval()
print(model)
print(ema_predictor)

producer_thread = Thread(target=producer, daemon=True)
producer_thread.start()
consumer()

client.close()