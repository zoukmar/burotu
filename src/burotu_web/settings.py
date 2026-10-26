"""Configuration du site, lue dans les variables d'environnement."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip().lower() for v in value.split(",") if v.strip())


@dataclass(frozen=True)
class Settings:
    # Domaines : le site public répond sur PUBLIC_HOSTS, le suivi sur ADMIN_HOST.
    public_hosts: tuple[str, ...] = field(
        default_factory=lambda: _csv(os.environ.get("BUROTU_PUBLIC_HOSTS", "burotu.com,localhost,127.0.0.1,testserver"))
    )
    admin_host: str = field(default_factory=lambda: os.environ.get("BUROTU_ADMIN_HOST", "admin.localhost").lower())

    admin_user: str = field(default_factory=lambda: os.environ.get("BUROTU_ADMIN_USER", "admin"))
    admin_password: str = field(default_factory=lambda: os.environ.get("BUROTU_ADMIN_PASSWORD", ""))
    secret_key: str = field(default_factory=lambda: os.environ.get("BUROTU_SECRET_KEY") or secrets.token_hex(32))
    secure_cookies: bool = field(default_factory=lambda: os.environ.get("BUROTU_SECURE_COOKIES", "0") == "1")

    db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("BUROTU_DB_PATH", Path(__file__).resolve().parents[2] / "data" / "site.db")
        )
    )


def get_settings() -> Settings:
    return Settings()
