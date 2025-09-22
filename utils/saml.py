"""SAML helpers for Streamlit admin authentication.

This module wires a minimal OneLogin python3-saml integration into the Streamlit
runtime. It exposes helper routines to build login URLs, register the SAML
FastAPI routes, and exchange short-lived tokens the UI can consume after the
Identity Provider posts back to the Assertion Consumer Service (ACS).
"""
from __future__ import annotations

import gc
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

import streamlit as st
from streamlit import config

try:  # Optional dependency until installed by the user
    from onelogin.saml2.auth import OneLogin_Saml2_Auth  # type: ignore
    from onelogin.saml2.settings import OneLogin_Saml2_Settings  # type: ignore
except ImportError:  # pragma: no cover - python3-saml not installed yet
    OneLogin_Saml2_Auth = None  # type: ignore
    OneLogin_Saml2_Settings = None  # type: ignore

try:  # FastAPI/Starlette types bundled with Streamlit
    from starlette.requests import Request
    from starlette.responses import Response
except ImportError:  # pragma: no cover - Streamlit runtime not loaded
    Request = Any  # type: ignore
    Response = Any  # type: ignore

try:
    from streamlit.web.server.routes import get_router
except Exception:  # pragma: no cover - legacy Streamlit
    get_router = None  # type: ignore


_TOKEN_TTL_SECONDS = 300  # Hand-off token lifetime (5 minutes)
_LOGIN_STATE: Dict[str, Dict[str, Any]] = {}
_TOKEN_STORE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()
_ROUTES_REGISTERED = False


@dataclass
class SamlRequestContext:
    scheme: str
    host_header: str
    port: int
    path: str
    root_path: str
    query_params: Dict[str, Any]
    headers: Dict[str, str]
    form_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamlResponse:
    status_code: int
    body: bytes = b""
    headers: list[Tuple[str, str]] = field(default_factory=list)

    def content_type(self) -> Optional[str]:
        for key, value in self.headers:
            if key.lower() == "content-type":
                return value
        return None


def _encode_body(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8")


def _build_response(
    status_code: int,
    body: Any = b"",
    *,
    headers: Optional[list[Tuple[str, str]]] = None,
    content_type: Optional[str] = None,
) -> SamlResponse:
    header_list: list[Tuple[str, str]] = []
    if headers:
        header_list.extend((str(key), str(value)) for key, value in headers)
    if content_type:
        header_list.append(("Content-Type", content_type))
    return SamlResponse(status_code=status_code, body=_encode_body(body), headers=header_list)


def _normalize_form_items(items: Any) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    if items is None:
        return normalized
    iterator = items
    if hasattr(items, "multi_items"):
        iterator = items.multi_items()
    elif hasattr(items, "items"):
        iterator = items.items()

    for raw_key, raw_value in iterator:
        key = str(raw_key)
        value = raw_value
        if isinstance(value, list):
            value = value[0] if value else None
        if hasattr(value, "read"):
            # Skip uploaded files; python3-saml only needs textual fields
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif value is not None and not isinstance(value, str):
            value = str(value)
        normalized[key] = value
    return normalized


async def _context_from_starlette_request(
    request: "Request",
    *,
    include_form: bool,
) -> SamlRequestContext:
    scheme = request.url.scheme
    host_header = request.headers.get("host", "")
    port = request.url.port or (443 if scheme == "https" else 80)
    root_path = request.scope.get("root_path", "")
    query_params = dict(request.query_params)
    headers = dict(request.headers)
    form_data: Dict[str, Any] = {}
    if include_form:
        form = await request.form()
        form_data = _normalize_form_items(form)
    return SamlRequestContext(
        scheme=scheme,
        host_header=host_header,
        port=port,
        path=request.url.path,
        root_path=root_path,
        query_params=query_params,
        headers=headers,
        form_data=form_data,
    )


def _context_from_tornado_handler(handler: "tornado.web.RequestHandler") -> SamlRequestContext:
    request = handler.request
    scheme = request.protocol or "http"
    host_header = request.headers.get("Host", request.host or "")
    port = 443 if scheme == "https" else 80
    if host_header and ":" in host_header:
        maybe_port = host_header.rsplit(":", 1)[-1]
        try:
            port = int(maybe_port)
        except ValueError:
            port = 443 if scheme == "https" else 80
    query_params: Dict[str, Any] = {}
    for key, values in request.query_arguments.items():
        if not values:
            continue
        value = values[-1]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        query_params[str(key)] = value

    form_data: Dict[str, Any] = {}
    for key, values in request.body_arguments.items():
        if not values:
            continue
        value = values[-1]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        form_data[str(key)] = value

    root_path = config.get_option("server.baseUrlPath") or ""
    headers = {str(k): str(v) for k, v in request.headers.items()}

    return SamlRequestContext(
        scheme=scheme,
        host_header=host_header,
        port=port,
        path=request.path,
        root_path=root_path,
        query_params=query_params,
        headers=headers,
        form_data=form_data,
    )


def _to_starlette_response(payload: SamlResponse) -> "Response":
    content_type = payload.content_type()
    response = Response(content=payload.body, status_code=payload.status_code, media_type=content_type)
    if content_type is None:
        try:
            del response.headers["content-type"]
        except KeyError:
            pass
    for key, value in payload.headers:
        if key.lower() == "content-type":
            continue
        response.headers[key] = value
    return response


def _write_tornado_response(handler: "tornado.web.RequestHandler", payload: SamlResponse) -> None:
    handler.set_status(payload.status_code)
    for key, value in payload.headers:
        handler.set_header(key, value)
    if payload.body:
        handler.write(payload.body)
    handler.finish()


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


def _prepare_request_data(context: SamlRequestContext) -> Dict[str, Any]:
    scheme = context.scheme or "http"
    host = context.host_header
    port = context.port or (443 if scheme == "https" else 80)
    return {
        "https": "on" if scheme == "https" else "off",
        "http_host": host,
        "server_port": str(port),
        "script_name": context.root_path,
        "path_info": context.path,
        "get_data": context.query_params,
        "post_data": context.form_data or {},
    }


def _build_auth(context: SamlRequestContext) -> OneLogin_Saml2_Auth:
    settings = _load_saml_settings()
    request_data = _prepare_request_data(context)
    return OneLogin_Saml2_Auth(request_data, old_settings=settings)


def _build_metadata_response() -> SamlResponse:
    settings = _load_saml_settings()
    metadata = settings.get_sp_metadata()
    errors = settings.validate_metadata(metadata)
    if errors:
        detail = "\n".join(errors)
        return _build_response(500, f"<h1>Invalid metadata</h1><pre>{detail}</pre>", content_type="text/html")
    return _build_response(200, metadata, content_type="application/xml")


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


def _handle_login(context: SamlRequestContext) -> SamlResponse:
    if not is_saml_configured():
        return _build_response(404, "SAML not configured", content_type="text/plain")

    state = context.query_params.get("state")
    if not state:
        return _build_response(400, "Missing state parameter", content_type="text/plain")

    next_target = _sanitize_next_target(context.query_params.get("next"))
    _store_login_state(str(state), next_path=next_target)

    auth = _build_auth(context)
    redirect_url = auth.login(return_to=str(state), stay=True)
    return _build_response(302, b"", headers=[("Location", redirect_url)])


def _handle_acs(context: SamlRequestContext) -> SamlResponse:
    if not is_saml_configured():
        return _build_response(404, "SAML not configured", content_type="text/plain")

    auth = _build_auth(context)
    auth.process_response()
    errors = auth.get_errors()
    if errors:
        detail = json.dumps(errors)
        return _build_response(400, f"<h1>SAML error</h1><pre>{detail}</pre>", content_type="text/html")

    if not auth.is_authenticated():
        return _build_response(401, "<h1>Authentication failed</h1>", content_type="text/html")

    relay_state = context.form_data.get("RelayState") or context.query_params.get("RelayState")
    if not relay_state:
        return _build_response(400, "<h1>Missing RelayState</h1>", content_type="text/html")

    state_info = _pop_login_state(str(relay_state))
    if not state_info:
        return _build_response(440, "<h1>Login session expired. Please try again.</h1>", content_type="text/html")

    payload = _extract_user_payload(auth, str(relay_state), state_info.get("next", "/"))
    token = _store_token(payload)

    target = state_info.get("next", "/")
    redirect_url = _append_query_param(target, saml_token=token)
    return _build_response(302, b"", headers=[("Location", redirect_url)])


def _handle_metadata() -> SamlResponse:
    if not is_saml_configured():
        return _build_response(404, "SAML not configured", content_type="text/plain")
    return _build_metadata_response()


def _saml_login_route_factory():
    async def saml_login(request: Request) -> Response:
        context = await _context_from_starlette_request(request, include_form=False)
        payload = _handle_login(context)
        return _to_starlette_response(payload)

    return saml_login


def _saml_acs_route_factory():
    async def saml_acs(request: Request) -> Response:
        context = await _context_from_starlette_request(request, include_form=True)
        payload = _handle_acs(context)
        return _to_starlette_response(payload)

    return saml_acs


def _saml_metadata_route_factory():
    async def saml_metadata(_: Request) -> Response:
        payload = _handle_metadata()
        return _to_starlette_response(payload)

    return saml_metadata


def _find_tornado_app() -> Optional["tornado.web.Application"]:
    try:
        import tornado.web  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - tornado ships with Streamlit
        raise RuntimeError("tornado is required for SAML route registration") from exc

    for obj in gc.get_objects():
        try:
            if isinstance(obj, tornado.web.Application):
                return obj
        except ReferenceError:
            continue
    return None


def _register_tornado_routes() -> None:
    try:
        import tornado.web  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tornado is required for SAML route registration") from exc

    from streamlit.web.server.server import make_url_path_regex  # type: ignore

    app = _find_tornado_app()
    if app is None:
        raise RuntimeError("Unable to locate Streamlit Tornado application for SAML route registration.")

    base = config.get_option("server.baseUrlPath") or ""

    class SamlHandler(tornado.web.RequestHandler):
        def initialize(self, handler_func):
            self._handler_func = handler_func

        async def get(self):
            await self._dispatch()

        async def post(self):
            await self._dispatch()

        async def _dispatch(self):
            context = _context_from_tornado_handler(self)
            payload = self._handler_func(context)
            _write_tornado_response(self, payload)

    app.add_handlers(
        r".*$",
        [
            (make_url_path_regex(base, "/saml/login"), SamlHandler, {"handler_func": _handle_login}),
            (make_url_path_regex(base, "/saml/acs"), SamlHandler, {"handler_func": _handle_acs}),
            (make_url_path_regex(base, "/saml/metadata"), SamlHandler, {"handler_func": lambda _ctx: _handle_metadata()}),
        ],
    )


def ensure_saml_routes_registered() -> None:
    """Register SAML HTTP routes with the active Streamlit server."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    if not is_saml_configured():
        return
    if get_router is None:
        _register_tornado_routes()
    else:
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
