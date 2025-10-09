import logging
import socket


class TCPLogHandler(logging.Handler):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = None
        self.connect()

    def connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=2)
            self.sock.settimeout(2)
        except Exception:
            self.sock = None

    def emit(self, record):
        if self.sock is None:
            self.connect()
        try:
            log_entry = self.format(record) + "\n"
            self.sock.sendall(log_entry.encode('utf-8'))
        except Exception:
            self.sock = None  # Reset the socket to retry later

    def close(self):
        if self.sock:
            self.sock.close()
        super().close()
