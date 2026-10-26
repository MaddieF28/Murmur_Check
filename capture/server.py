"""HTTPS server for the capture page (phones need HTTPS for the mic), plus the link the PC worker uses.
Run: python3 capture/server.py, then open the printed URL on a phone on the same Wi-Fi.
"""
import http.server
import json
import re
import ssl
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
REC = ROOT.parent / "recordings"
PC_RESULTS = ROOT.parent / "ml/out/exp_pc"
CERT, KEY = ROOT / "cert.pem", ROOT / "key.pem"  # self-signed certificate for HTTPS
PORT = 8443


def lan_ip():
    for iface in ("en0", "en1"):
        ip = subprocess.run(["ipconfig", "getifaddr", iface], capture_output=True, text=True).stdout.strip()
        if ip:
            return ip
    return "127.0.0.1"


def ensure_cert(ip):
    if CERT.exists() and KEY.exists():
        return
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30",
        "-keyout", str(KEY), "-out", str(CERT), "-subj", "/CN=heart-capture",
        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
    ], check=True, capture_output=True)


def receive_zenodo(chunks_dir, parts):
    """Join the uploaded chunks and unzip only the expected names (the link is public)."""
    import io
    import zipfile
    data = b"".join((chunks_dir / f"chunk_{i:03d}").read_bytes() for i in range(parts))
    out = ROOT.parent / "data/zenodo"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name == "labels.json" or re.fullmatch(r"(WVU|WC2023)/[A-Za-z0-9_.-]+\.wav", name):
                dest = out / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(name))
    for i in range(parts):
        (chunks_dir / f"chunk_{i:03d}").unlink()


class Handler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        if size > 20_000_000:  # 30 s recording is ~3 MB
            self.send_error(413)
            return
        q = parse_qs(urlparse(self.path).query)
        if urlparse(self.path).path == "/pc_result":
            # strip the name so it can't point outside the folder
            name = re.sub(r"[^A-Za-z0-9_-]", "", q.get("name", ["x"])[0])[:80]
            kind = q.get("kind", [""])[0]
            body = self.rfile.read(size)
            dest = PC_RESULTS / name
            dest.mkdir(parents=True, exist_ok=True)
            if re.fullmatch(r"fold\d", kind):
                (dest / f"{kind}.pt").write_bytes(body)
            elif m := re.fullmatch(r"zen_chunk(\d{1,3})", kind):
                (dest / f"chunk_{int(m[1]):03d}").write_bytes(body)
            elif m := re.fullmatch(r"zen_done(\d{1,3})", kind):
                receive_zenodo(dest, int(m[1]))
            elif kind == "log":
                (dest / "log.txt").write_bytes(body[:100_000])
            elif kind in ("row", "scores", "config"):
                json.loads(body)  # only keep valid JSON
                if kind == "row":
                    with open(PC_RESULTS / "results.jsonl", "ab") as f:
                        f.write(body.strip() + b"\n")
                else:
                    (dest / f"{kind}.json").write_bytes(body)
            self.send_response(200)
            self.end_headers()
            return
        # otherwise a phone recording (.wav) or its notes (.json)
        name = re.sub(r"[^A-Za-z0-9_-]", "", q.get("name", ["rec"])[0])[:80]
        ext = ".json" if "meta" in q else ".wav"
        body = self.rfile.read(size)
        REC.mkdir(exist_ok=True)
        (REC / f"{name}{ext}").write_bytes(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"{name}{ext}".encode())


if __name__ == "__main__":
    ip = lan_ip()
    ensure_cert(ip)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    sslctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sslctx.load_cert_chain(CERT, KEY)
    srv.socket = sslctx.wrap_socket(srv.socket, server_side=True)
    print(f"Open on your phone: https://{ip}:{PORT}  (accept the certificate warning)")
    srv.serve_forever()
