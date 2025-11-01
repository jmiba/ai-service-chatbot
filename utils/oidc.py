"""OIDC helpers for Streamlit admin authentication."""

from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse, urlunparse

import httpx
import streamlit as st


DEFAULT_SCOPES = ["openid", "profile", "email"]


def _get_oidc_secret_block() -> Optional[Dict[str, Any]]:
    try:
        block = st.secrets["oidc"]  # type: ignore[index]
    except KeyError:
        return None
    if not isinstance(block, dict):
        block = dict(block)
    return dict(block)


def _sanitize_next_target(value: Optional[str]) -> str:
    if not value:
        return "/"
    parsed = urlparse(str(value))
    if parsed.scheme or parsed.netloc:
        return "/"
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    query = parsed.query
    return urlunparse(("", "", path, "", query, ""))


@lru_cache(maxsize=1)
def _load_discovery_document() -> Dict[str, Any]:
    config = _get_oidc_secret_block() or {}
    discovery_url = config.get("discovery_url")
    if not discovery_url:
        issuer = config.get("issuer")
        if issuer:
            discovery_url = str(issuer).rstrip("/") + "/.well-known/openid-configuration"
    if not discovery_url:
        return {}
    timeout = float(config.get("timeout", 10.0))
    try:
        response = httpx.get(str(discovery_url), timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - depends on network
        raise RuntimeError(f"Failed to fetch OIDC discovery document: {exc}") from exc
    return response.json()


def _resolve_endpoint(name: str) -> str:
    config = _get_oidc_secret_block() or {}
    if value := config.get(name):
        return str(value)
    discovery = _load_discovery_document()
    value = discovery.get(name)
    if not value:
        raise RuntimeError(
            f"OIDC configuration missing '{name}'. Provide it in secrets or via discovery document."
        )
    return str(value)


def is_oidc_configured() -> bool:
    config = _get_oidc_secret_block()
    if not config:
        return False
    if not config.get("client_id"):
        return False
    if not config.get("redirect_uri"):
        return False
    try:
        _resolve_endpoint("authorization_endpoint")
        _resolve_endpoint("token_endpoint")
        _resolve_endpoint("userinfo_endpoint")
    except Exception:
        return False
    return True


def get_redirect_uri() -> str:
    config = _get_oidc_secret_block() or {}
    redirect_uri = config.get("redirect_uri")
    if not redirect_uri:
        raise RuntimeError("OIDC redirect_uri is not configured.")
    return str(redirect_uri)


def get_oidc_allowlist() -> set[str]:
    config = _get_oidc_secret_block() or {}
    allow = config.get("allowed_admin_emails") or config.get("allowlist") or []
    if isinstance(allow, str):
        allow = [allow]
    return {str(item).strip().lower() for item in allow if str(item).strip()}


def allow_password_fallback() -> bool:
    config = _get_oidc_secret_block() or {}
    return bool(config.get("allow_password_fallback", False))


def get_default_next_path() -> str:
    config = _get_oidc_secret_block() or {}
    return _sanitize_next_target(config.get("default_next"))


def generate_pkce_pair() -> tuple[str, str]:
    verifier_bytes = os.urandom(64)
    verifier = base64.urlsafe_b64encode(verifier_bytes).decode("utf-8").rstrip("=")
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return verifier, challenge


def build_oidc_login_url(state: str, code_challenge: str, *, extra_params: Optional[Dict[str, Any]] = None) -> str:
    config = _get_oidc_secret_block() or {}
    authorize_endpoint = _resolve_endpoint("authorization_endpoint")
    scopes = config.get("scopes", DEFAULT_SCOPES)
    if isinstance(scopes, str):
        scopes = [part.strip() for part in scopes.replace(",", " ").split() if part.strip()]
    scope_str = " ".join(scopes or DEFAULT_SCOPES)

    params: Dict[str, Any] = {
        "response_type": "code",
        "client_id": config.get("client_id"),
        "redirect_uri": get_redirect_uri(),
        "scope": scope_str,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    if audience := config.get("audience"):
        params["audience"] = audience

    additional = config.get("authorize_params")
    if isinstance(additional, dict):
        params.update({str(k): v for k, v in additional.items() if v is not None})
    if extra_params:
        params.update(extra_params)

    return f"{authorize_endpoint}?{urlencode(params, doseq=True)}"


def _token_auth_method() -> str:
    config = _get_oidc_secret_block() or {}
    method = str(config.get("token_auth_method", "client_secret_post")).lower()
    if method not in {"client_secret_post", "client_secret_basic"}:
        method = "client_secret_post"
    return method


def exchange_code_for_tokens(code: str, code_verifier: str, *, redirect_uri: Optional[str] = None) -> Dict[str, Any]:
    config = _get_oidc_secret_block() or {}
    token_endpoint = _resolve_endpoint("token_endpoint")
    redirect = redirect_uri or get_redirect_uri()
    timeout = float(config.get("timeout", 10.0))

    data: Dict[str, Any] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect,
        "code_verifier": code_verifier,
        "client_id": config.get("client_id"),
    }

    client_secret = config.get("client_secret")
    auth = None
    if client_secret:
        if _token_auth_method() == "client_secret_basic":
            auth = (str(config.get("client_id")), str(client_secret))
        else:
            data["client_secret"] = client_secret

    try:
        response = httpx.post(
            token_endpoint,
            data=data,
            auth=auth,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - depends on network
        raise RuntimeError(f"OIDC token request failed: {exc}") from exc

    return response.json()


def fetch_userinfo(access_token: str) -> Dict[str, Any]:
    userinfo_endpoint = _resolve_endpoint("userinfo_endpoint")
    config = _get_oidc_secret_block() or {}
    timeout = float(config.get("timeout", 10.0))
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = httpx.get(userinfo_endpoint, headers=headers, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - depends on network
        raise RuntimeError(f"OIDC userinfo request failed: {exc}") from exc
    return response.json()

