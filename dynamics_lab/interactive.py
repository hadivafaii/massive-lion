"""Serve the live optimizer kinematics simulator."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dynamics_lab.engine import SimulationState, defaults_payload


STATE = SimulationState()
STATE_LOCK = threading.Lock()
HTML_PATH = Path(__file__).with_name("interactive.html")


class Handler(BaseHTTPRequestHandler):
	def do_GET(self):
		path = urlparse(self.path).path
		if path in {"/", "/interactive.html"}:
			self._send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
			return
		if path == "/api/defaults":
			self._send_json(defaults_payload())
			return
		if path == "/api/state":
			with STATE_LOCK:
				payload = STATE.snapshot(include_landscape=True)
			self._send_json(payload)
			return
		self.send_error(404)

	def do_POST(self):
		path = urlparse(self.path).path
		if path == "/api/reset":
			config = self._read_json()
			try:
				with STATE_LOCK:
					payload = STATE.reset(config)
			except (ValueError, TypeError, KeyError) as error:
				self._send_bytes(json.dumps({"error": str(error)}).encode(), "application/json", status=400)
				return
			self._send_json(payload)
			return
		if path == "/api/step":
			with STATE_LOCK:
				payload = STATE.step()
			self._send_json(payload)
			return
		self.send_error(404)

	def log_message(self, fmt, *args):
		return

	def _read_json(self):
		length = int(self.headers.get("Content-Length", "0"))
		if length == 0:
			return {}
		return json.loads(self.rfile.read(length).decode("utf-8"))

	def _send_json(self, data):
		body = json.dumps(data).encode("utf-8")
		self._send_bytes(body, "application/json")

	def _send_bytes(self, body, content_type, status=200):
		self.send_response(status)
		self.send_header("Content-Type", content_type)
		self.send_header("Content-Length", str(len(body)))
		self.end_headers()
		try:
			self.wfile.write(body)
		except BrokenPipeError:
			pass


def main():
	parser = argparse.ArgumentParser(
		description="Run the optimizer kinematics simulator.")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8011)
	args = parser.parse_args()

	# noinspection PyTypeChecker
	server = ThreadingHTTPServer((args.host, args.port), Handler)
	url = f"http://{args.host}:{args.port}"
	print(f"Serving optimizer kinematics simulator at {url}")
	server.serve_forever()


if __name__ == "__main__":
	main()
