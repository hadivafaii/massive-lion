"""Serve the live optimizer kinematics simulator."""

import argparse
import json
import threading
import webbrowser
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
		if path == "/favicon.svg":
			self._send_bytes(HTML_PATH.with_name("favicon.svg").read_bytes(), "image/svg+xml")
			return
		if path == "/favicon.ico":
			self._send_bytes(HTML_PATH.with_name("favicon.ico").read_bytes(), "image/x-icon")
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
	parser.add_argument("--no-browser", action="store_true",
						help="Serve without opening a browser tab")
	args = parser.parse_args()

	# noinspection PyTypeChecker
	server = ThreadingHTTPServer((args.host, args.port), Handler)
	host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
	url = f"http://{host}:{server.server_port}"
	print(f"Serving Dynamics Lab at {url}", flush=True)
	if not args.no_browser:
		# The socket is already listening. Open off-thread so browser startup
		# cannot prevent the server from handling its first request.
		threading.Thread(target=webbrowser.open_new_tab, args=(url,), daemon=True).start()
	try:
		server.serve_forever()
	finally:
		server.server_close()


if __name__ == "__main__":
	main()
