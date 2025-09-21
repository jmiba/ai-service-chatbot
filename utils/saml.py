"""SAML helpers for Streamlit admin authentication.

This module wires a minimal OneLogin python3-saml integration into the Streamlit
runtime. It exposes helper routines to build login URLs, register the SAML
FastAPI routes, and exchange short-lived tokens the UI can consume after the
Identity Provider posts back to the Assertion Consumer Service (ACS).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

import streamlit as st

try:  # Optional dependency until installed by the user
    from onelogin.saml2.auth import OneLogin_Saml2_Auth  # type: ignore
    from onelogin.saml2.settings import OneLogin_Saml2_Settings  # type: ignore
except ImportError:  # pragma: no cover - python3-saml not installed yet
    OneLogin_Saml2_Auth = None  # type: ignore
    OneLogin_Saml2_Settings = None  # type: ignore

try:  # FastAPI/Starlette types bundled with Streamlit
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, RedirectResponse, Response
except ImportError:  # pragma: no cover - Streamlit runtime not loaded
    Request = Any  # type: ignore
    Response = Any  # type: ignore
    RedirectResponse = Any  # type: ignore
    HTMLResponse = Any  # type: ignore

try:
    from streamlit.web.server.routes import get_router
except Exception:  # pragma: no cover - legacy Streamlit
    get_router = None  # type: ignore


_TOKEN_TTL_SECONDS = 300  # Hand-off token lifetime (5 minutes)
_LOGIN_STATE: Dict[str, Dict[str, Any]] = {}
_TOKEN_STORE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()
_ROUTES_REGISTERED = False


def _now() -> float:
    return time.time()


def _get_saml_secret_block() -> Optional[Dict[str, Any]]:
    try:
        block = st.secrets["saml"]
    except KeyError:
        return None
    if hasattr(block, "to_dict"):
        return block.to_dict()
    return dict(block)


def is_saml_configured() -> bool:
    """Return True when python3-saml is available and SAML secrets are set."""
    return OneLogin_Saml2_Auth is not None and _get_saml_secret_block() is not None


def _sanitize_next_target(value: Optional[str]) -> str:
    if not value:
        return "/"
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        # Forbid absolute URLs (open redirect)
        return "/"
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    query = parsed.query
    return urlunparse(("", "", path, "", query, ""))


def build_saml_login_url(state: str, *, next_path: Optional[str] = None) -> str:
    """Construct the relative login URL that kicks off the SAML handshake."""
    params = {"state": state}
    if next_path:
        params["next"] = _sanitize_next_target(next_path)
    return "/saml/login?" + urlencode(params)


def _append_query_param(url: str, **params: str) -> str:
    parsed = urlparse(url)
    existing = dict(parse_qsl(parsed.query, keep_blank_values=True))
    for key, value in params.items():
        if value is None:
            continue
        existing[key] = value
    new_query = urlencode(existing, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _cleanup_expired_locked() -> None:
    now = _now()
    expired_states = [key for key, value in _LOGIN_STATE.items() if now - value.get("created_at", 0) > _TOKEN_TTL_SECONDS]
    for key in expired_states:
        _LOGIN_STATE.pop(key, None)
    expired_tokens = [key for key, value in _TOKEN_STORE.items() if now > value.get("expires", 0)]
    for key in expired_tokens:
        _TOKEN_STORE.pop(key, None)


def _store_login_state(state: str, *, next_path: str) -> None:
    with _LOCK:
        _cleanup_expired_locked()
        _LOGIN_STATE[state] = {"created_at": _now(), "next": _sanitize_next_target(next_path)}


def _pop_login_state(state: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        _cleanup_expired_locked()
        return _LOGIN_STATE.pop(state, None)


def _store_token(payload: Dict[str, Any]) -> str:
    token = uuid.uuid4().hex
    with _LOCK:
        _cleanup_expired_locked()
        _TOKEN_STORE[token] = {"payload": payload, "expires": _now() + _TOKEN_TTL_SECONDS}
    return token


def pop_saml_token(token: str) -> Optional[Dict[str, Any]]:
    """Return and invalidate a short-lived token created by the ACS handler."""
    with _LOCK:
        _cleanup_expired_locked()
        entry = _TOKEN_STORE.pop(token, None)
    if not entry:
        return None
    return entry.get("payload")


@lru_cache(maxsize=1)
def _load_saml_settings() -> OneLogin_Saml2_Settings:
    if OneLogin_Saml2_Settings is None:
        raise RuntimeError("python3-saml is not installed. Add it to requirements.txt to enable SAML.")

    config = _get_saml_secret_block()
    if not config:
        raise RuntimeError("SAML configuration missing in Streamlit secrets.")

    try:
        strict = bool(config.get("strict", True))
        debug = bool(config.get("debug", False))
        sp_entity_id = config["sp_entity_id"]
        acs_url = config["sp_acs"].rstrip("/")
        idp_entity_id = config["idp_entity_id"]
        idp_sso_url = config["idp_sso_url"]
        idp_x509cert = config["idp_x509_cert"]
    except KeyError as exc:
        raise RuntimeError(f"Missing required SAML secret: {exc}") from exc

    slo_url = config.get("sp_sls")
    idp_slo_url = config.get("idp_slo_url")
    sp_x509 = config.get("sp_x509_cert", "")
    sp_private_key = config.get("sp_private_key", "")

    settings_dict = {
        "strict": strict,
        "debug": debug,
        "sp": {
            "entityId": sp_entity_id,
            "assertionConsumerService": {
                "url": acs_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
            },
            "x509cert": sp_x509,
            "privateKey": sp_private_key,
        },
        "idp": {
            "entityId": idp_entity_id,
            "singleSignOnService": {
                "url": idp_sso_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "x509cert": idp_x509cert,
        },
    }

    if slo_url:
        settings_dict["sp"]["singleLogoutService"] = {
            "url": slo_url,
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
        }
    if idp_slo_url:
        settings_dict["idp"]["singleLogoutService"] = {
            "url": idp_slo_url,
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
        }

    security_overrides = config.get("security") or {}
    if security_overrides:
        settings_dict["security"] = security_overrides

    return OneLogin_Saml2_Settings(settings=settings_dict, custom_base_path=None)


def _prepare_request_data(request: Request, post_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = request.url
    scheme = url.scheme
    host = request.headers.get("host", "")
    port = url.port or (443 if scheme == "https" else 80)
    get_params = dict(request.query_params)
    data = {
        "https": "on" if scheme == "https" else "off",
        "http_host": host,
        "server_port": str(port),
        "script_name": request.scope.get("root_path", ""),
        "path_info": request.url.path,
        "get_data": get_params,
        "post_data": post_data or {},
    }
    return data


def _build_auth(request: Request, post_data: Optional[Dict[str, Any]] = None) -> OneLogin_Saml2_Auth:
    settings = _load_saml_settings()
    request_data = _prepare_request_data(request, post_data)
    return OneLogin_Saml2_Auth(request_data, old_settings=settings)


def _build_metadata_response() -> Response:
    settings = _load_saml_settings()
    metadata = settings.get_sp_metadata()
    errors = settings.validate_metadata(metadata)
    if errors:
        detail = "\n".join(errors)
        return HTMLResponse(content=f"<h1>Invalid metadata</h1><pre>{detail}</pre>", status_code=500)
    return Response(content=metadata, media_type="application/xml")


def _extract_user_payload(auth: OneLogin_Saml2_Auth, relay_state: str, next_target: str) -> Dict[str, Any]:
    attributes = auth.get_attributes() or {}
    name_id = auth.get_nameid()
    session_index = auth.get_session_index()

    config = _get_saml_secret_block() or {}
    email_attr = config.get("email_attribute") or "mail"
    fallback_attrs = ["mail", "email", "userPrincipalName"]

    email = None
    for key in [email_attr] + fallback_attrs:
        value = attributes.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, str) and value.strip():
            email = value.strip()
            break
    if not email and isinstance(name_id, str):
        email = name_id

    name_attr = config.get("name_attribute") or "displayName"
    display_name = None
    for key in [name_attr, "cn", "displayName", "givenName"]:
        value = attributes.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, str) and value.strip():
            display_name = value.strip()
            break

    return {
        "email": email,
        "name": display_name,
        "attributes": attributes,
        "name_id": name_id,
        "session_index": session_index,
        "relay_state": relay_state,
        "next": next_target,
    }


def _saml_login_route_factory():
    async def saml_login(request: Request) -> Response:
        if not is_saml_configured():
            return Response("SAML not configured", status_code=404)

        state = request.query_params.get("state")
        if not state:
            return Response("Missing state parameter", status_code=400)
        next_target = _sanitize_next_target(request.query_params.get("next"))
        _store_login_state(state, next_path=next_target)

        auth = _build_auth(request)
        redirect_url = auth.login(return_to=state, stay=True)
        # stay=True returns the URL without triggering redirect, so we do it here
        return RedirectResponse(url=redirect_url, status_code=302)

    return saml_login


def _saml_acs_route_factory():
    async def saml_acs(request: Request) -> Response:
        if not is_saml_configured():
            return Response("SAML not configured", status_code=404)

        form = await request.form()
        form_dict = {k: v for k, v in form.multi_items()} if hasattr(form, "multi_items") else dict(form)
        auth = _build_auth(request, form_dict)
        auth.process_response()
        errors = auth.get_errors()
        if errors:
            detail = json.dumps(errors)
            return HTMLResponse(content=f"<h1>SAML error</h1><pre>{detail}</pre>", status_code=400)

        if not auth.is_authenticated():
            return HTMLResponse(content="<h1>Authentication failed</h1>", status_code=401)

        relay_state = form_dict.get("RelayState") or request.query_params.get("RelayState")
        if not relay_state:
            return HTMLResponse(content="<h1>Missing RelayState</h1>", status_code=400)

        state_info = _pop_login_state(relay_state)
        if not state_info:
            return HTMLResponse(content="<h1>Login session expired. Please try again.</h1>", status_code=440)

        payload = _extract_user_payload(auth, relay_state, state_info.get("next", "/"))
        token = _store_token(payload)

        target = state_info.get("next", "/")
        redirect_url = _append_query_param(target, saml_token=token)
        return RedirectResponse(url=redirect_url, status_code=302)

    return saml_acs


def _saml_metadata_route_factory():
    async def saml_metadata(_: Request) -> Response:
        if not is_saml_configured():
            return Response("SAML not configured", status_code=404)
        return _build_metadata_response()

    return saml_metadata


def ensure_saml_routes_registered() -> None:
    """Register FastAPI routes (idempotent)."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    if not is_saml_configured():
        return
    if get_router is None:
        raise RuntimeError("Streamlit version does not expose get_router(); upgrade to 1.25+ to use SAML.")

    router = get_router()
    router.add_api_route("/saml/login", _saml_login_route_factory(), methods=["GET"], name="saml_login")
    router.add_api_route("/saml/acs", _saml_acs_route_factory(), methods=["POST"], name="saml_acs")
    router.add_api_route("/saml/metadata", _saml_metadata_route_factory(), methods=["GET"], name="saml_metadata")
    _ROUTES_REGISTERED = True


def get_saml_allowlist() -> set[str]:
    config = _get_saml_secret_block() or {}
    allow = config.get("allowed_admin_emails") or config.get("allowlist") or []
    if isinstance(allow, str):
        allow = [allow]
    return {str(item).strip().lower() for item in allow if str(item).strip()}


def allow_password_fallback() -> bool:
    config = _get_saml_secret_block() or {}
    return bool(config.get("allow_password_fallback", False))


def get_default_next_path() -> str:
    config = _get_saml_secret_block() or {}
    default_target = config.get("default_next")
    return _sanitize_next_target(default_target)
