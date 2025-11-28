"""OIDC helpers for Streamlit admin authentication.

This module provides helper functions for the native Streamlit OIDC integration
(st.login, st.logout, st.user). Uses the [auth] block in secrets.toml.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st


def _get_auth_secret_block() -> Optional[dict[str, Any]]:
    """Get the [auth] configuration block from secrets."""
    try:
        block = st.secrets["auth"]  # type: ignore[index]
    except KeyError:
        return None
    if not isinstance(block, dict):
        block = dict(block)
    return dict(block)


def is_auth_configured() -> bool:
    """Check if Streamlit native auth is configured in secrets."""
    config = _get_auth_secret_block()
    if not config:
        return False
    # Minimum required: redirect_uri and cookie_secret
    if not config.get("redirect_uri"):
        return False
    if not config.get("cookie_secret"):
        return False
    # Need either direct config or a named provider
    has_direct = bool(config.get("client_id") and config.get("server_metadata_url"))
    # Check for named providers (auth.google, auth.microsoft, etc.)
    has_named = any(
        isinstance(v, dict) and v.get("client_id")
        for k, v in config.items()
        if k not in ("redirect_uri", "cookie_secret", "client_id", "client_secret", 
                     "server_metadata_url", "client_kwargs", "allowed_admin_emails", 
                     "allowlist", "allow_password_fallback")
    )
    return has_direct or has_named


def get_auth_allowlist() -> set[str]:
    """Get the list of allowed admin email addresses from [auth] config."""
    config = _get_auth_secret_block() or {}
    allow = config.get("allowed_admin_emails") or config.get("allowlist") or []
    if isinstance(allow, str):
        allow = [allow]
    return {str(item).strip().lower() for item in allow if str(item).strip()}


def allow_password_fallback() -> bool:
    """Check if password fallback is enabled for emergency access."""
    config = _get_auth_secret_block() or {}
    return bool(config.get("allow_password_fallback", False))


def get_provider_name() -> Optional[str]:
    """Get the name of the configured OIDC provider, if using a named provider.
    
    Returns None if using a default (unnamed) provider configuration.
    """
    config = _get_auth_secret_block() or {}
    # Check for named providers (auth.google, auth.microsoft, etc.)
    for key, value in config.items():
        if key not in ("redirect_uri", "cookie_secret", "client_id", "client_secret", 
                       "server_metadata_url", "client_kwargs", "allowed_admin_emails", 
                       "allowlist", "allow_password_fallback"):
            if isinstance(value, dict) and value.get("client_id"):
                return key
    return None

