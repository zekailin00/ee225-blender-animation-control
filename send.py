import socket
import json

host = "localhost"
port = 12345

data = {
    "object_name": "Cube",
    "location_x": 10.0,
    "location_y": 10.0,
    "location_z": 0.0
}

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((host, port))
    client.sendall(json.dumps(data).encode("utf-8"))
    client.close()