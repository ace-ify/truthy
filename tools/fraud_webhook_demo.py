"""
Minimal fraud-ops webhook receiver for Truthy's reporting path (demo / finale prop).

Receives POSTed case bundles, verifies the HMAC signature with the shared key, and prints a
receipt. Lets you demonstrate the end-to-end "report to fraud team" path against a real endpoint.

    python tools/fraud_webhook_demo.py --port 9100 --key demo-sign-key

Then run the API with:
    FRAUD_WEBHOOK_URL=http://127.0.0.1:9100/intake  REPORT_SIGNING_KEY=demo-sign-key  uvicorn api.main:app ...
and POST /api/case/{id}/report -> this receiver logs the case and whether the signature verified.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.governance.bundle import verify_bundle

KEY = ""


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n)
        try:
            bundle = json.loads(body)
        except Exception:  # noqa: BLE001
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"ok":false}'); return
        valid = verify_bundle(bundle, KEY) if KEY else None
        c = bundle.get("case", {})
        print(f"[intake] case={c.get('caseId')} verdict={c.get('verdict')} risk={c.get('riskLevel')} "
              f"mediaSha={str(c.get('mediaSha256'))[:16]}… signature_valid={valid}", flush=True)
        payload = json.dumps({"ok": True, "signatureValid": valid, "received": c.get("caseId")}).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload))); self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # silence default logging
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--key", default="")
    args = ap.parse_args()
    KEY = args.key
    print(f"fraud-ops webhook demo listening on http://127.0.0.1:{args.port}/intake "
          f"(signature check {'ON' if KEY else 'off'})", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
