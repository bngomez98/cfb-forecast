"""Email notifications for completed forecasts and validations."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def send(recipient, subject, body):
    msg = EmailMessage()
    msg["From"] = os.environ.get("CFB_MAIL_FROM", "cfb-forecast@localhost")
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "25"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    if host:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            if os.environ.get("SMTP_STARTTLS", "").lower() in {"1", "true", "yes"}:
                try:
                    server.starttls()
                    server.ehlo()
                except smtplib.SMTPNotSupportedError:
                    pass
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return "smtp"
    import shutil
    import subprocess
    sendmail = shutil.which("sendmail") or "/usr/sbin/sendmail"
    with subprocess.Popen([sendmail, "-t"], stdin=subprocess.PIPE) as p:
        p.stdin.write(msg.as_bytes())
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"sendmail exited {rc}")
    return "sendmail"


def queue_dir():
    d = Path(os.environ.get("CFB_MAIL_QUEUE", ".cfb_mail_outbox"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_queue_file(recipient, subject, body):
    from datetime import UTC, datetime
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = queue_dir() / f"{stamp}-{abs(hash(subject)) % 10**8}.eml"
    path.write_text(f"To: {recipient}\nSubject: {subject}\n\n{body}\n")
    return str(path)
