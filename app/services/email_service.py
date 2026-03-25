from __future__ import annotations

import asyncio
import os
import smtplib
from email.message import EmailMessage
from urllib.parse import urlencode


class EmailConfigError(RuntimeError):
    pass


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_invite_base_url() -> str:
    explicit = (
        os.getenv("INVITE_BASE_URL", "").strip()
        or os.getenv("APP_BASE_URL", "").strip()
    )
    if explicit:
        return explicit.rstrip("/")

    origins = [
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
        if origin.strip() and origin.strip() != "*"
    ]
    if origins:
        return origins[0].rstrip("/")

    return "http://127.0.0.1:3000"


def build_agent_invite_url(token: str, email: str) -> str:
    query = urlencode({"invite_token": token, "email": email})
    return f"{get_invite_base_url()}/register?{query}"


def build_password_reset_url(token: str, email: str) -> str:
    query = urlencode({"token": token, "email": email})
    return f"{get_invite_base_url()}/reset-password?{query}"


def _send_email_sync(*, to_email: str, subject: str, text_body: str) -> None:
    host = os.getenv("SMTP_HOST", "").strip()
    from_email = os.getenv("MAIL_FROM", "").strip() or os.getenv("SMTP_FROM_EMAIL", "").strip()
    if not host or not from_email:
        raise EmailConfigError("SMTP_HOST and MAIL_FROM must be configured to send emails.")

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    use_ssl = _flag("SMTP_USE_SSL", False)
    use_tls = _flag("SMTP_USE_TLS", not use_ssl)

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email
    message.set_content(text_body)

    smtp_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
    with smtp_cls(host, port, timeout=20) as server:
        if not use_ssl and use_tls:
            server.starttls()
        if username:
            server.login(username, password)
        server.send_message(message)


async def send_agent_invitation_email(
    *,
    to_email: str,
    agent_name: str,
    inviter_email: str,
    invite_url: str,
) -> None:
    subject = f"You're invited to access {agent_name}"
    text_body = (
        f"{inviter_email} invited you to access the shared agent \"{agent_name}\".\n\n"
        "Create your chatbot account using the link below, and the agent will be shared with you automatically:\n\n"
        f"{invite_url}\n\n"
        "If you were not expecting this invitation, you can ignore this email."
    )
    await asyncio.to_thread(
        _send_email_sync,
        to_email=to_email,
        subject=subject,
        text_body=text_body,
    )


async def send_password_reset_email(*, to_email: str, reset_url: str) -> None:
    subject = "Reset your Chatbot SaaS password"
    text_body = (
        "We received a request to reset your password for Chatbot SaaS.\n\n"
        "Use the link below to choose a new password:\n\n"
        f"{reset_url}\n\n"
        "This link will expire in 1 hour.\n"
        "If you did not request a password reset, you can ignore this email."
    )
    await asyncio.to_thread(
        _send_email_sync,
        to_email=to_email,
        subject=subject,
        text_body=text_body,
    )