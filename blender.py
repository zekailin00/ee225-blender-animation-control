import bpy
import socket
import threading
import json

# TCP server to listen for animation parameters
class TCPServer:
    def __init__(self, host="localhost", port=12345):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.settimeout(1.0)  # Timeout for non-blocking operations
        self.server.bind((host, port))
        self.server.listen(1)
        self.client = None
        self.running = True

    def accept_client(self):
        print("Waiting for a connection...")
        while self.running:
            try:
                self.client, _ = self.server.accept()
                print("Client connected!")
                self.handle_client()
            except socket.timeout:
                continue

    def handle_client(self):
        while self.running and self.client:
            try:
                data = self.client.recv(1024)
                if data:
                    yield json.loads(data.decode("utf-8"))
                else:
                    print("Client disconnected.")
                    self.client.close()
                    self.client = None
                    break
            except Exception as e:
                print(f"Error handling client: {e}")
                self.client.close()
                self.client = None
                break

    def receive_data(self):
        # Poll the current client connection
        if self.client:
            try:
                data = self.client.recv(1024)
                if data:
                    return json.loads(data.decode("utf-8"))
                else:
                    print("Client disconnected.")
                    self.client.close()
                    self.client = None
            except Exception as e:
                print(f"Error receiving data: {e}")
                self.client.close()
                self.client = None
        return None

    def stop(self):
        self.running = False
        if self.client:
            self.client.close()
            self.client = None
        self.server.close()
        print("TCP server stopped.")

# Blender operator to integrate server
class AnimationUpdaterOperator(bpy.types.Operator):
    """Operator to update animation parameters from a TCP server"""
    bl_idname = "wm.tcp_animation_updater"
    bl_label = "TCP Animation Updater"

    _server = None
    _timer = None

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self._server:
                data = self._server.receive_data()
                if data:
                    self.update_animation(context, data)
        return {'PASS_THROUGH'}

    def update_animation(self, context, data):
        # Update objects or animations based on received data
        obj = bpy.data.objects.get(data.get("object_name"))
        if obj:
            obj.location.x = data.get("location_x", obj.location.x)
            obj.location.y = data.get("location_y", obj.location.y)
            obj.location.z = data.get("location_z", obj.location.z)
        bpy.context.view_layer.update()

    def execute(self, context):
        self._server = TCPServer()
        threading.Thread(target=self._server.accept_client, daemon=True).start()
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        print("TCP Animation Updater started.")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        if self._server:
            self._server.stop()
            self._server = None
        print("TCP server and operator stopped.")
        return {'CANCELLED'}

# Register and unregister functions
def register():
    bpy.utils.register_class(AnimationUpdaterOperator)

def unregister():
    bpy.utils.unregister_class(AnimationUpdaterOperator)

if __name__ == "__main__":
    register()

    # To start the operator:
    bpy.ops.wm.tcp_animation_updater()
