# Placeholder for a disabled vLLM service.
# Answers health/sleep/wake with 200 so compose dependencies still resolve.
# Uses no GPU memory.
from http.server import BaseHTTPRequestHandler, HTTPServer

OK_PATHS = ("/health", "/sleep", "/wake_up", "/is_sleeping")


class Handler(BaseHTTPRequestHandler):
    def _reply(self):
        ok = self.path.split("?")[0] in OK_PATHS
        self.send_response(200 if ok else 503)
        self.end_headers()
        self.wfile.write(b"disabled" if ok else b"vLLM service disabled by platform setting")

    do_GET = _reply
    do_POST = _reply

    def log_message(self, *args):
        pass


HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
