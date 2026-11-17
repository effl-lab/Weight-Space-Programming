"""Network path for a sandbox started with --unshare-net.

Host side: `AllowlistProxy` accepts HTTP CONNECT requests on a unix socket and tunnels only to
hosts in its allowlist; every request is logged with its decision.
Sandbox side: `python -I netproxy.py <unix_socket> <port>` listens on 127.0.0.1:<port> and relays
each connection to the unix socket, so tools that honor HTTPS_PROXY reach the host proxy.
"""
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

BUF = 65536


def _pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while data := a.recv(BUF):
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _relay(a: socket.socket, b: socket.socket) -> None:
    t = threading.Thread(target=_pipe, args=(b, a), daemon=True)
    t.start()
    _pipe(a, b)
    t.join()
    a.close()
    b.close()


class AllowlistProxy:
    def __init__(self, sock_path: Path, allow: set[str], log_path: Path):
        self.sock_path, self.allow, self.log_path = sock_path, allow, log_path
        self.lock = threading.Lock()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(sock_path))
        self.server.listen(64)
        threading.Thread(target=self._serve, daemon=True).start()

    def _log(self, entry: dict) -> None:
        with self.lock, self.log_path.open("a") as f:
            f.write(json.dumps({"t": time.time(), **entry}) + "\n")

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        head = b""
        while b"\r\n\r\n" not in head and len(head) < 8192:
            chunk = conn.recv(4096)
            if not chunk:
                conn.close()
                return
            head += chunk
        line = head.split(b"\r\n", 1)[0].decode("latin-1")
        parts = line.split()
        if len(parts) < 2 or parts[0] != "CONNECT":
            self._log({"request": line[:200], "allowed": False})
            conn.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            conn.close()
            return
        host, _, port = parts[1].rpartition(":")
        allowed = host in self.allow and port == "443"
        self._log({"host": host, "port": port, "allowed": allowed})
        if not allowed:
            conn.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            conn.close()
            return
        try:
            upstream = socket.create_connection((host, 443), timeout=30)
            upstream.settimeout(None)
        except OSError:
            conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            conn.close()
            return
        conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        _relay(conn, upstream)

    def close(self) -> None:
        self.server.close()
        self.sock_path.unlink(missing_ok=True)


def sandbox_relay(sock_path: str, port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(64)
    while True:
        conn, _ = srv.accept()
        up = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        up.connect(sock_path)
        threading.Thread(target=_relay, args=(conn, up), daemon=True).start()


if __name__ == "__main__":
    if os.environ.get("WSP_RELAY_FORK") == "1" and os.fork():
        sys.exit(0)
    sandbox_relay(sys.argv[1], int(sys.argv[2]))
