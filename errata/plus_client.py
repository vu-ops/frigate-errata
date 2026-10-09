from __future__ import annotations

import logging
import re

import requests

logger = logging.getLogger(__name__)

PLUS_API_HOST = "https://api.frigate.video"
KEY_RE = re.compile(r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}:[a-z0-9]{40}")


class PlusError(RuntimeError):
    pass


class PlusClient:
    """Minimal client for the Frigate+ model API.

    The key is used only for the life of this object and never persisted.
    """

    def __init__(self, key: str, host: str = PLUS_API_HOST, timeout: float = 30.0):
        self.key = (key or "").strip()
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._token: str | None = None
        self.session = requests.Session()

    @staticmethod
    def validate_key(key: str) -> bool:
        return bool(KEY_RE.fullmatch((key or "").strip()))

    def _authenticate(self) -> None:
        if not self.key:
            raise PlusError("no Frigate+ API key provided")
        if not self.validate_key(self.key):
            raise PlusError("Frigate+ API key is not formatted correctly")
        key_id, secret = self.key.split(":", 1)
        resp = self.session.get(
            f"{self.host}/v1/auth/token", auth=(key_id, secret), timeout=self.timeout
        )
        if resp.status_code != 200:
            raise PlusError(f"Frigate+ auth failed (HTTP {resp.status_code})")
        token = resp.json().get("accessToken")
        if not token:
            raise PlusError("Frigate+ auth returned no access token")
        self._token = token

    def _get(self, path: str) -> requests.Response:
        if self._token is None:
            self._authenticate()
        resp = self.session.get(
            f"{self.host}/v1/{path}",
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=self.timeout,
        )
        if resp.status_code == 401:
            self._token = None
            self._authenticate()
            resp = self.session.get(
                f"{self.host}/v1/{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=self.timeout,
            )
        return resp

    def list_models(self) -> list[dict]:
        resp = self._get("model/list")
        if resp.status_code != 200:
            raise PlusError(f"listing models failed (HTTP {resp.status_code})")
        return (resp.json() or {}).get("list") or []

    def get_model_info(self, model_id) -> dict:
        resp = self._get(f"model/{model_id}")
        if resp.status_code != 200:
            raise PlusError(f"model {model_id} not found (HTTP {resp.status_code})")
        return resp.json() or {}

    def download_model(self, model_id, dest_path: str) -> str:
        resp = self._get(f"model/{model_id}/signed_url")
        if resp.status_code != 200:
            raise PlusError(f"could not get download url for {model_id} (HTTP {resp.status_code})")
        url = (resp.json() or {}).get("url")
        if not url:
            raise PlusError("Frigate+ returned no download url")
        with self.session.get(url, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            tmp = dest_path + ".part"
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
        import os

        os.replace(tmp, dest_path)
        return dest_path
