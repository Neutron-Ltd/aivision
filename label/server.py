import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

if __name__ == '__main__':
    target_dir = os.path.abspath("images")
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found.")
        exit(1)
    
    os.chdir(target_dir)
    print("CORS server live on port 8765. Serving: " + target_dir)
    HTTPServer(("", 8765), CORSRequestHandler).serve_forever()