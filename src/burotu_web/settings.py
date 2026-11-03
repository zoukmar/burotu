"""Site settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip().lower() for v in value.split(",") if v.strip())


@dataclass(frozen=True)
class Settings:
    # Domains: the site answers on PUBLIC_HOSTS; the retired tracking host redirects to burotu.com.
    public_hosts: tuple[str, ...] = field(
        default_factory=lambda: _csv(os.environ.get("BUROTU_PUBLIC_HOSTS", "burotu.com,localhost,127.0.0.1,testserver"))
    )
    admin_host: str = field(default_factory=lambda: os.environ.get("BUROTU_ADMIN_HOST", "admin.localhost").lower())


def get_settings() -> Settings:
    return Settings()
