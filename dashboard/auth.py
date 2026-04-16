import hashlib
import hmac
import os
from typing import Optional, Tuple

import streamlit as st


def _slug(name: str) -> str:
    return name.upper().replace("-", "_")


def _secret_get(path: str, default=None):
    current = st.secrets
    try:
        for part in path.split("."):
            current = current[part]
        return current
    except Exception:
        return default


def _get_credential_source(hospital_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    slug = _slug(hospital_id)

    username = (
        os.getenv(f"{slug}_USERNAME")
        or _secret_get(f"auth.{hospital_id}.username")
    )
    password = (
        os.getenv(f"{slug}_PASSWORD")
        or _secret_get(f"auth.{hospital_id}.password")
    )
    password_hash = (
        os.getenv(f"{slug}_PASSWORD_HASH")
        or _secret_get(f"auth.{hospital_id}.password_hash")
    )
    salt = (
        os.getenv(f"{slug}_PASSWORD_SALT")
        or _secret_get(f"auth.{hospital_id}.password_salt")
    )
    return username, password, f"{salt}:{password_hash}" if salt and password_hash else None


def _pbkdf2_hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()


def verify_credentials(hospital_id: str, username: str, password: str) -> Tuple[bool, str]:
    expected_username, plain_password, salted_hash = _get_credential_source(hospital_id)

    if not expected_username:
        return False, (
            f"Login is not configured for {hospital_id}. "
            f"Set `{hospital_id.upper()}_USERNAME` and either `{hospital_id.upper()}_PASSWORD` "
            f"or `{hospital_id.upper()}_PASSWORD_SALT` + `{hospital_id.upper()}_PASSWORD_HASH`."
        )

    if not hmac.compare_digest(username, expected_username):
        return False, "Invalid username or password."

    if plain_password is not None:
        return hmac.compare_digest(password, plain_password), "Invalid username or password."

    if salted_hash:
        salt, expected_hash = salted_hash.split(":", 1)
        actual_hash = _pbkdf2_hash(password, salt)
        return hmac.compare_digest(actual_hash, expected_hash), "Invalid username or password."

    return False, "No password configured for this hospital."


def render_login(hospital_id: str, title: str, accent: str) -> bool:
    auth_key = f"auth_{hospital_id}"
    if st.session_state.get(auth_key, False):
        return True

    st.markdown(
        f"""
<style>
    .login-card {{
        max-width: 440px;
        margin: 5rem auto 1rem auto;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        border-top: 4px solid {accent};
        background: rgba(255,255,255,0.03);
    }}
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="login-card">', unsafe_allow_html=True)
    st.markdown(f"### {title} Login")
    st.caption("Authorized staff only. Credentials are checked against environment variables or Streamlit secrets.")
    username = st.text_input("Username", key=f"{hospital_id}_login_user")
    password = st.text_input("Password", type="password", key=f"{hospital_id}_login_password")
    if st.button("Sign in", type="primary", key=f"{hospital_id}_login_button"):
        ok, message = verify_credentials(hospital_id, username, password)
        if ok:
            st.session_state[auth_key] = True
            st.session_state[f"auth_user_{hospital_id}"] = username
            st.rerun()
        st.error(message)
    st.markdown("</div>", unsafe_allow_html=True)
    return False


def logout(hospital_id: str) -> None:
    st.session_state[f"auth_{hospital_id}"] = False
    st.session_state.pop(f"auth_user_{hospital_id}", None)
