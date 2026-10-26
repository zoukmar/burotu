# syntax=docker/dockerfile:1.7
# Burotu web image (burotu.com + admin.burotu.com). The ML pipeline is not included.

FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[web]"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BUROTU_DB_PATH=/data/site.db \
    BUROTU_SECURE_COOKIES=1

RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels burotu[web] \
 && useradd --system --uid 10001 --home /nonexistent burotu \
 && mkdir -p /data && chown burotu /data

USER burotu
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

# --forwarded-allow-ips: Caddy runs on the host and forwards the visitor's real IP.
CMD ["uvicorn", "burotu_web.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--workers", "1"]
