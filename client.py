
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config, HubertModel
import matplotlib.pyplot as plt

import socket
import json
import time
from playsound import playsound
import threading

from recording import record_audio
from utils import load_audio, init_torch_device

host = "localhost"
port = 12345

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

def run_every_n_milliseconds(n, x, fn):
    # Convert n milliseconds to seconds
    n_seconds = n / 1000.0
    # Calculate the end time
    end_time = time.time() + x
    
    while time.time() < end_time:
        # Start time of the iteration
        start_time = time.time()
        
        # Your task or function here
        print("Running task at", time.time())
        fn(start_time)
        
        # Calculate elapsed time and sleep for the remaining duration
        elapsed = time.time() - start_time
        if elapsed < n_seconds:
            time.sleep(n_seconds - elapsed)

def EMA_to_shapekeys(data, EMA):
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

def plot_ema(y_val, y_pred, channel, sensor_name):
    # Ensure y_val and y_pred have the same shape (slice if needed)
    min_length = min(y_val.shape[0], y_pred.shape[0])
    y_val = y_val[:min_length, :]  # Slicing to match the shape of y_pred if needed
    y_pred = y_pred[:min_length, :]

    # Plot each feature
    num_features = y_val.shape[1]
    plt.figure(figsize=(12, 8))

    for i in channel:
        plt.plot(y_val[:, i], label=f'Actual EMA Pos {i+1}', linestyle='-', alpha=0.7)
        plt.plot(y_pred[:, i], label=f'Pred. EMA Pos {i+1}', linestyle='--', alpha=0.7)

    plt.title("Comparison of Actual vs. Predicted Values for EMA Position " + sensor_name)
    plt.xlabel("Time")
    plt.ylabel("EMA Position (mm)")
    plt.legend()
    plt.show()

# record_audio()

# File path to the sound
file_index = "0015"
sound_file_path = "/Users/zekailin00/Git/SpeechDrivenTongueAnimation/TongueMocapData/wav/"+ file_index +".wav"
feature_path = "/Users/zekailin00/Git/SpeechDrivenTongueAnimation/TongueMocapData/ema/npy/"+ file_index +".npy"
# sound_file_path = "/Users/zekailin00/Git/ee225-blender-animation-control/output.wav"
is_plot_on_matplotlib = False

# Function to play sound
def play_sound_in_background(sound_file):
    playsound(sound_file)
sound_thread = threading.Thread(target=play_sound_in_background, args=(sound_file_path,))

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

with torch.no_grad():
    audio_signal, _ = load_audio(sound_file_path)
    audio_tensor = torch.Tensor(audio_signal).unsqueeze(0).to(device)
    print("[input] audio_tensor.shape:", audio_tensor.shape)

    feat = model(audio_tensor, output_hidden_states=True)
    x_features = feat.hidden_states[24].detach()[0]
    print("[features] x_features.shape:", x_features.shape)

    y_pred = []
    for i in range(x_features.shape[0]):
        result = ema_predictor((x_features[i]))
        y_pred.append(result.detach().cpu().numpy())
    y_pred = np.array(y_pred)
    print("[output] y_pred.shape:", y_pred.shape)

    if is_plot_on_matplotlib:
        sensor_name = ["TD", "TB", "BR", "BL", "TT", "UL", "LC", "LL", "LI", "LJ"]
        y_val = np.load(feature_path)
        for i in range(10):
            plot_ema(y_val, y_pred, [i*3+0, i*3+1, i*3+2], sensor_name[i])
    else:
        ema_streaming_data = y_pred
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect((host, port))

            timeIndex = 0
            def callback(time):
                global timeIndex
                print("time index:", timeIndex)
                if (timeIndex >= ema_streaming_data.shape[0]):
                    quit()
                EMA_to_shapekeys(data, ema_streaming_data[timeIndex])
                timeIndex += 1

                client.sendall(json.dumps(data).encode("utf-8"))

            # Start the thread
            sound_thread.start()
            run_every_n_milliseconds(20, ema_streaming_data.shape[0]/20, callback)
            sound_thread.join()
            client.close()
