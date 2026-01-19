from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class BlandAPIError(RuntimeError):
    """Raised when the Bland API returns an error or an unexpected payload."""


@dataclass
class BlandAPIClient:
    """Minimal Bland API client for conversational pathways.

    Endpoints implemented (per docs.bland.ai):
      - POST /v1/pathway/create
      - POST /v1/pathway/{pathway_id}

    Auth: set BLAND_API_KEY in your environment.
    """

    api_key: Optional[str] = None
    base_url: str = "https://api.bland.ai"
    timeout_s: int = 60

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("BLAND_API_KEY")

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise BlandAPIError(
                "BLAND_API_KEY is not set. Set it in your environment before calling Bland APIs."
            )
        return {
            "Content-Type": "application/json",
            "authorization": self.api_key,
        }

    def _request(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=self._headers(),
                json=json_body,
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            raise BlandAPIError(f"Network error calling Bland API: {e}") from e

        # Try parse JSON even on non-200
        try:
            data = resp.json() if resp.content else {}
        except Exception:
            data = {"raw_text": resp.text}

        if resp.status_code >= 400:
            raise BlandAPIError(f"Bland API error {resp.status_code}: {data}")

        # Many Bland endpoints return {status: success|error, ...}
        if isinstance(data, dict) and data.get("status") == "error":
            raise BlandAPIError(f"Bland API returned error: {data}")

        if not isinstance(data, dict):
            raise BlandAPIError(f"Unexpected Bland API response (not a JSON object): {data}")

        return data

    def create_pathway(self, name: str, description: str = "") -> str:
        """Create a pathway and return the new pathway_id."""
        payload = {"name": name, "description": description}
        data = self._request("POST", "/v1/pathway/create", payload)
        pid = (data.get("pathway_id") or (data.get("data") or {}).get("pathway_id"))
        if not pid:
            raise BlandAPIError(f"Create pathway did not return pathway_id: {data}")
        return str(pid)

    def update_pathway(
        self,
        pathway_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        nodes: Optional[Any] = None,
        edges: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Update a pathway's fields including nodes and edges."""
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if nodes is not None:
            payload["nodes"] = nodes
        if edges is not None:
            payload["edges"] = edges
        if not payload:
            raise ValueError("update_pathway called with no fields to update")
        return self._request("POST", f"/v1/pathway/{pathway_id}", payload)

    def get_pathway(self, pathway_id: str) -> Dict[str, Any]:
        """Fetch single pathway information."""
        return self._request("GET", f"/v1/pathway/{pathway_id}")
