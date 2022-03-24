"""
Module to prepare and serve data for visualization
"""

import logging
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

LOGGER = logging.getLogger(__name__)

HOST, PORT = "localhost", 8080

WWW_DIR = "/tmp/"

ANALYSIS_PATH = ""


class CustomHTTPHandler(SimpleHTTPRequestHandler):
    """ HTTP Handler to serve report files
    - Gives JSON input data file when get a request of analysis.json
    """
    def handle(self):
        self.directory = WWW_DIR
        super().handle()

    def _set_json_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        """Serve a GET request."""
        if self.path == "/analysis.json":
            self._set_json_headers()
            analysis_file = open(ANALYSIS_PATH, 'rb')
            try:
                self.copyfile(analysis_file, self.wfile)
            finally:
                analysis_file.close()

        else:
            serve_file = self.send_head()
            if serve_file:
                try:
                    self.copyfile(serve_file, self.wfile)
                finally:
                    serve_file.close()


def launch_report_webserver(json_data_file):
    """ Start the server

        - Launch server that serves build folder
        - Gives JSON input data file when get a request of analysis.json
    """

    target_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(target_dir, "application/build")

    # pylint: disable-msg=global-statement
    global WWW_DIR
    WWW_DIR = target_dir

    global ANALYSIS_PATH
    ANALYSIS_PATH = json_data_file

    httpd = HTTPServer((HOST, PORT), CustomHTTPHandler)
    LOGGER.info("Serving ready: http://%s:%s", HOST, PORT)
    httpd.serve_forever()
