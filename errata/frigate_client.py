from __future__ import annotations

import json
import logging
import time
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


class FrigateError(RuntimeError):
    pass


class FrigateClient:
    def __init__(self, frigate_cfg: dict):
        self.base = str(frigate_cfg.get("api_url", "http://frigate:5000")).rstrip("/")
        self.timeout = float(frigate_cfg.get("request_timeout", 30))
        self.username = frigate_cfg.get("username") or ""
        self.password = frigate_cfg.get("password") or ""
        self._token: str | None = None
        self.session = requests.Session()

    @property
    def auth_enabled(self) -> bool:
        return bool(self.username and self.password)

    def login(self) -> None:
        if not self.auth_enabled:
            return
        payload = {"username": self.username, "password": self.password}
        candidates = [
            ("data", payload),
            ("data", {"user": self.username, "password": self.password}),
            ("json", payload),
            ("json", {"user": self.username, "password": self.password}),
        ]
        last_error = ""
        for kwarg, body in candidates:
            try:
                resp = self.session.post(
                    f"{self.base}/api/login",
                    timeout=self.timeout,
                    **{kwarg: body},
                )
            except requests.RequestException as exc:
                raise FrigateError(f"frigate login failed: {exc}") from exc
            if resp.status_code == 200:
                token = resp.json().get("access_token")
                if token:
                    self._token = token
                    logger.info("authenticated to frigate as %s", self.username)
                    return
                last_error = "login 200 but no access_token in response"
            else:
                last_error = f"login attempt returned {resp.status_code}"
        raise FrigateError(f"unable to authenticate to frigate: {last_error}")

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base}{path}"
        headers = dict(kwargs.pop("headers", None) or {})
        if self._token:
            headers.setdefault("Authorization", f"Bearer {self._token}")
        timeout = kwargs.pop("timeout", self.timeout)
        resp = self.session.request(
            method, url, headers=headers, timeout=timeout, **kwargs
        )
        if resp.status_code == 401 and self.auth_enabled:
            logger.info("frigate rejected credentials, re-authenticating")
            self._token = None
            self.login()
            headers["Authorization"] = f"Bearer {self._token}"
            resp = self.session.request(
                method, url, headers=headers, timeout=timeout, **kwargs
            )
        return resp

    def version(self) -> str:
        resp = self.request("GET", "/api/version")
        resp.raise_for_status()
        return resp.text.strip().strip('"')

    def events(self, since: float | None, limit: int = 500, before: float | None = None,
               labels: list[str] | None = None) -> list[dict]:
        params: dict = {
            "limit": limit,
            "has_snapshot": "1",
            "in_progress": "0",
            "include_clips": "0",
        }
        if since:
            params["after"] = since
        if before:
            params["before"] = before
        if labels:
            params["labels"] = ",".join(labels)
        resp = self.request("GET", "/api/events", params=params)
        if resp.status_code != 200:
            raise FrigateError(f"GET /api/events returned {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def download_snapshot(self, event_id: str, dest: str) -> bool:
        resp = self.request("GET", f"/api/events/{event_id}/snapshot.jpg?bbox=0")
        if resp.status_code != 200 or not resp.content:
            logger.warning("snapshot for %s unavailable (HTTP %s)", event_id, resp.status_code)
            return False
        part = dest + ".part"
        with open(part, "wb") as fh:
            fh.write(resp.content)
        import os

        os.replace(part, dest)
        return True

    def get_config_text(self) -> str:
        resp = self.request("GET", "/api/config/raw")
        if resp.status_code != 200:
            raise FrigateError(f"GET /api/config/raw returned {resp.status_code}")
        text = resp.text
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                return decoded
        except ValueError:
            pass
        return text

    def save_config(self, text: str, save_option: str = "restart") -> requests.Response:
        resp = self.request(
            "POST",
            f"/api/config/save?save_option={quote(save_option)}",
            data=text.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
        return resp

    def wait_until_ready(self, max_wait: float = 120.0, interval: float = 2.0) -> bool:
        """Poll Frigate until it answers again (used after a save_option=restart)."""
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            try:
                if self.request("GET", "/api/version").status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        return False
