import websocket
import threading
import json
import logging

class NetworkClient:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.logger = logging.getLogger("NetworkClient")
        
        #Thread-safe storage for the latest server response
        self._latest_prediction = 0 
        self._connected = False
        self._lock = threading.Lock() 

    def connect(self):
        """Starts the WebSocket connection in a background thread."""
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        #Run network loop in a daemon thread so it doesn't block the app
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def send_skeleton(self, skeleton_data):
        """Sends skeleton data (List or Numpy) to server."""
        if self._connected and self.ws:
            try:
                # Ensure data is JSON serializable (convert numpy to list)
                if hasattr(skeleton_data, 'tolist'):
                    payload = json.dumps(skeleton_data.tolist())
                else:
                    payload = json.dumps(skeleton_data)
                
                self.ws.send(payload)
            except Exception as e:
                self.logger.error(f"Send failed: {e}")

    def get_latest_prediction(self):
        """Returns the most recent prediction from the server."""
        with self._lock:
            return self._latest_prediction

    def is_connected(self):
        return self._connected

    def close(self):
        if self.ws:
            self.ws.close()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            pred = data.get("prediction", 0)
            with self._lock:
                self._latest_prediction = pred
        except Exception:
            pass

    def _on_open(self, ws):
        self.logger.info("Connected to Server")
        self._connected = True

    def _on_close(self, ws, status, msg):
        self.logger.info("Disconnected from Server")
        self._connected = False

    def _on_error(self, ws, error):
        self.logger.error(f"Network Error: {error}")