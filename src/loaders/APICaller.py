import logging
import os
from pathlib import Path

import requests
from pydantic import HttpUrl

# Warnung unterdrücken wenn SSL-Verification deaktiviert ist
# requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


class MoodleMaintenanceError(Exception):
    """Raised when the Moodle webservice reports that the whole site is down.

    This is a *global* condition (e.g. errorcode "sitemaintenance"): every
    subsequent API call will fail too, so callers must abort the run rather than
    skip the offending module. Skipping would let the ingestion delete existing
    indexed data and replace it with incomplete content during the outage.
    """


# Moodle webservice errorcodes that mean the entire site is unavailable.
# These are NOT per-resource failures, so they must abort the whole run.
_SITE_DOWN_ERRORCODES = {"sitemaintenance", "maintenance", "servicenotavailable"}


class APICaller:
    def __init__(self, url: HttpUrl, params: dict = {}, headers: dict = {}, **kwargs) -> None:
        self.logger = logging.getLogger("loader")
        self.url = url
        self.params = {}
        self.params.update(params)
        self.headers = {}
        self.headers.update(headers)
        self.params.update(kwargs)
        self.response = requests.Response()
        # This fixes very slow requests, IPV6 is not properly supported by the ki-campus.org server
        # https://stackoverflow.com/questions/62599036/python-requests-is-slow-and-takes-very-long-to-complete-http-or-https-request
        requests.packages.urllib3.util.connection.HAS_IPV6 = False

    def get(self, **kwargs):
        # NOTE: requests.get() without timeouts can hang forever. Use conservative defaults.
        # Override via env if needed.
        timeout_connect = float(os.getenv("HTTP_TIMEOUT_CONNECT", "10"))
        timeout_read = float(os.getenv("HTTP_TIMEOUT_READ", "60"))

        # Workaround: dont verify if server certificate is invalid
        # self.response = requests.get(url=self.url, params=self.params, headers=self.headers, verify=False)
        self.response = requests.get(
            url=self.url,
            params=self.params,
            headers=self.headers,
            verify=True,
            timeout=(timeout_connect, timeout_read),
        )
        try:
            self.response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise err

    def getJSON(self, **kwargs) -> dict:
        try:
            self.get(**kwargs)
        except requests.exceptions.HTTPError as err:
            self.logger.warn(f"Failed to retrieve {self.url}")
            raise err
        response_json = self.response.json()
        if isinstance(response_json, dict) and "exception" in response_json:
            errorcode = response_json.get("errorcode", "UnknownError")
            message = response_json.get("message", "No message provided")
            if errorcode in _SITE_DOWN_ERRORCODES:
                # Whole site is down — abort the run instead of failing this one call.
                raise MoodleMaintenanceError(f"{errorcode}: {message}")
            raise Exception(f"{errorcode}: {message}")
        return response_json

    def getText(self, **kwargs) -> str:
        try:
            self.get(**kwargs)
        except requests.exceptions.HTTPError as err:
            self.logger.warn(f"Failed to retrieve {self.url}")
            raise err
        return self.response.text

    def getBuffer(self, **kwargs) -> str:
        self.get(**kwargs)
        return self.response.text

    def getFile(self, filename, tmp_dir):
        local_filename = Path(f"{tmp_dir}/{filename}")
        # TEMPORARY: verify=False wegen abgelaufenem Server-Zertifikat
        with requests.get(self.url, params=self.params, stream=True, verify=False) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)
        return local_filename
