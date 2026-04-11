"""HTTP client for the ds-skills.com API. Zero external dependencies (stdlib only)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_DEFAULT_BASE_URL = "https://ds-skills.com"
_TIMEOUT = 30


class ApiError(Exception):
    """Raised when the API returns a non-2xx status."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"HTTP {status}: {message}")


class Client:
    """Thin wrapper around the ds-skills REST API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")

    # -- low-level ----------------------------------------------------------

    def _get_json(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/api{path}"
        if params:
            params = {k: v for k, v in params.items() if v is not None}
            if params:
                url += "?" + urlencode(params)
        req = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read())
        except HTTPError as exc:
            body = exc.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except (json.JSONDecodeError, AttributeError):
                detail = body
            raise ApiError(exc.code, detail) from None
        except URLError as exc:
            raise ApiError(0, f"Connection failed: {exc.reason}") from None

    def _post_json(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}/api{path}"
        data = json.dumps(body).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read())
        except HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            try:
                detail = json.loads(raw).get("detail", raw)
            except (json.JSONDecodeError, AttributeError):
                detail = raw
            raise ApiError(exc.code, detail) from None
        except URLError as exc:
            raise ApiError(0, f"Connection failed: {exc.reason}") from None

    def _get_bytes(self, path: str) -> bytes:
        url = f"{self.base_url}/api{path}"
        req = Request(url, headers={"Accept": "application/zip"})
        try:
            with urlopen(req, timeout=60) as resp:
                return resp.read()
        except HTTPError as exc:
            body = exc.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except (json.JSONDecodeError, AttributeError):
                detail = body
            raise ApiError(exc.code, detail) from None
        except URLError as exc:
            raise ApiError(0, f"Connection failed: {exc.reason}") from None

    # -- public API ---------------------------------------------------------

    def list_skills(
        self,
        domain: str | None = None,
        query: str | None = None,
        page: int = 1,
        limit: int = 200,
    ) -> dict:
        """GET /api/skills — list with optional domain/query filter."""
        return self._get_json(
            "/skills", {"domain": domain, "q": query, "page": page, "limit": limit}
        )

    def search(
        self,
        query: str,
        domain: str | None = None,
        page: int = 1,
        limit: int = 50,
    ) -> dict:
        """GET /api/search — full-text search with facets."""
        return self._get_json(
            "/search", {"q": query, "domain": domain, "page": page, "limit": limit}
        )

    def show_skill(self, domain: str, slug: str) -> dict:
        """GET /api/skills/{domain}/{slug} — full detail with markdown."""
        return self._get_json(f"/skills/{domain}/{slug}")

    def stats(self) -> dict:
        """GET /api/stats — aggregate statistics."""
        return self._get_json("/stats")

    def download_skill(self, domain: str, slug: str) -> bytes:
        """GET /api/download/{domain}/{slug} — single skill ZIP."""
        return self._get_bytes(f"/download/{domain}/{slug}")

    def download_domain(self, domain: str) -> bytes:
        """GET /api/download/{domain} — all skills in a domain ZIP."""
        return self._get_bytes(f"/download/{domain}")

    def post_feedback(self, payload: dict) -> dict:
        """POST /api/feedback — submit outcome/traces for a skill."""
        return self._post_json("/feedback", payload)

    # -- convenience --------------------------------------------------------

    def pull_skill(self, domain: str, slug: str, dest: Path) -> list[str]:
        """Download and extract a single skill. Returns list of written files."""
        data = self.download_skill(domain, slug)
        return self._extract_zip(data, dest)

    def pull_domain(self, domain: str, dest: Path) -> list[str]:
        """Download and extract all skills in a domain. Returns list of written files."""
        data = self.download_domain(domain)
        return self._extract_zip(data, dest)

    @staticmethod
    def _extract_zip(data: bytes, dest: Path) -> list[str]:
        dest.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                target = dest / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info))
                written.append(str(target))
        return written
