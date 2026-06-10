"""MJPEG streaming server for live neural activity visualization.

Modeled on doom-neuron's ``mjpeg_server.py`` pattern: a
``ThreadingHTTPServer`` in a daemon thread serves a single MJPEG
endpoint.  The simulation thread calls :meth:`update_frame` with an
RGB numpy array; clients connected to the HTTP endpoint receive the
latest JPEG-encoded frame.

Usage::

    server = NeuralMJPEGServer(port=12350)
    server.start()
    ...
    server.update_frame(frame_rgb)   # called from simulation loop
    ...
    server.stop()
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np


class NeuralMJPEGServer:
    """MJPEG streaming server for live neural activity visualization."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 12350,
        path: str = "/neural.mjpeg",
    ):
        self.host = host
        self.port = port
        self.path = path

        # Shared state — protected by a lock for thread safety
        self._lock = threading.Lock()
        self._jpeg_bytes: bytes | None = None
        self._timestamp: int = 0

        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- public API ----------------------------------------------------------

    def start(self):
        """Start the HTTP server in a background daemon thread."""
        server_self = self  # capture for the handler closure

        class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        class _MJPEGHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path != server_self.path:
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                last_ts = 0
                while True:
                    try:
                        with server_self._lock:
                            ts = server_self._timestamp
                            data = server_self._jpeg_bytes

                        if ts != last_ts and data is not None:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                            last_ts = ts
                        else:
                            time.sleep(0.01)

                    except (BrokenPipeError, ConnectionResetError):
                        break
                    except Exception:
                        break

            def log_message(self, format, *args):  # noqa: A002
                """Suppress default stderr logging."""
                pass

        self._server = _ThreadingHTTPServer((self.host, self.port), _MJPEGHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="neural-mjpeg",
        )
        self._thread.start()
        print(f"Neural MJPEG stream at http://{self.host}:{self.port}{self.path}")

    def update_frame(self, frame_rgb: np.ndarray):
        """Update the current frame (called from the simulation thread).

        Accepts an RGB numpy array, encodes it to JPEG once, and stores the
        bytes for all connected HTTP clients.

        Args:
            frame_rgb: ``(H, W, 3)`` uint8 array in RGB order.
        """
        import cv2  # lazy: only the dashboard path needs OpenCV

        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = buf.tobytes()

        with self._lock:
            self._jpeg_bytes = jpeg_bytes
            self._timestamp = time.perf_counter_ns()

    def stop(self):
        """Shut down the HTTP server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
