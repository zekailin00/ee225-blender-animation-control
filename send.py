import socket
import json
import time
import numpy as np
from playsound import playsound
import threading

host = "localhost"
port = 12345

data = {
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
    data["AU22 lipsFunnel"]             = 0 #EMA[5*3+1]
    data["AU18 lipsProtude"]            = 0 #EMA[5*3+1]
    data["AU23 lipsKiss"]               = (EMA[LABEL["LC"]] + 4) / 10.0 #EMA[5*3+1]
    data["AU14 jawStrained"]            = (EMA[LABEL["LC"]] + 3) / -5.0 #EMA[5*3+1] ???
    data["AU12 smile"]                  = 0 #EMA[5*3+1]
    data["AU20 lipStrained"]            = 0 #EMA[5*3+1]
    data["AU10 upper lip raiser"]       = (EMA[LABEL["UL"] + LABEL["z"]] - 1) / 5.0
    data["AU16 lower lip depressor"]    = (EMA[LABEL["LL"] + LABEL["z"]] - EMA[LABEL["LI"] + LABEL["z"]] - 1) / -5.0
    data["AU17 hmmm"]                   = (EMA[LABEL["UL"] + LABEL["z"]] - EMA[LABEL["LI"] + LABEL["z"]] - 1) / 5.0
    data["AU28 lipsBite"]               = 0 #EMA[5*3+1]
    data["AU27 JawOpen"]                = (EMA[LABEL["LI"] + LABEL["z"]] + 20.0)/ -30.0



# File path to the sound
file_index = "0013"
sound_file_path = "/Users/zekailin00/Git/SpeechDrivenTongueAnimation/TongueMocapData/wav/"+ file_index +".wav"
feature_path = "/Users/zekailin00/Git/SpeechDrivenTongueAnimation/TongueMocapData/ema/npy/"+ file_index +".npy"

y_val = np.load(feature_path)
timeIndex = 0;

# Function to play sound
def play_sound_in_background(sound_file):
    playsound(sound_file)
sound_thread = threading.Thread(target=play_sound_in_background, args=(sound_file_path,))


print("y_val.shape", y_val.shape)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((host, port))

    def callback(time):
        # data['AU22 lipsFunnel'] = (time - int(time))
        global timeIndex
        print("time index:", timeIndex)
        EMA_to_shapekeys(data, y_val[timeIndex])
        timeIndex += 1

        client.sendall(json.dumps(data).encode("utf-8"))

    # Start the thread
    sound_thread.start()
    run_every_n_milliseconds(20, y_val.shape[0]/20, callback)
    sound_thread.join()
    client.close()