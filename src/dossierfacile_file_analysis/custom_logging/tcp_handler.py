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
        self.sock = socket.create_connection((self.host, self.port))

    def emit(self, record):
        try:
            log_entry = self.format(record) + "\n"
            self.sock.sendall(log_entry.encode('utf-8'))
        except Exception:
            self.handleError(record)

    def close(self):
        if self.sock:
            self.sock.close()
        super().close()
