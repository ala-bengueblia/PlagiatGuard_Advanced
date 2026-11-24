
# src/plagiatguard/research_security.py

import ipaddress
import os
import socket
import time

from collections import defaultdict, deque
from urllib.parse import urlparse, urljoin

import jwt
import requests

from fastapi import Depends, HTTPException, Request


ROLE_LEVEL = {
    "viewer": 1,
    "analyst": 2,
    "admin": 3,
}


def current_identity(request: Request):
    authorization = request.headers.get("Authorization", "")

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Jeton Bearer requis.",
        )

    secret = os.getenv("JWT_SECRET", "")

    if len(secret) < 32:
        raise HTTPException(
            status_code=500,
            detail="JWT_SECRET doit contenir au moins 32 caractères.",
        )

    try:
        payload = jwt.decode(
            authorization[7:],
            secret,
            algorithms=["HS256"],
            options={"require": ["sub", "role", "exp"]},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Jeton invalide ou expiré.",
        )

    role = payload.get("role")

    if role not in ROLE_LEVEL:
        raise HTTPException(
            status_code=403,
            detail="Rôle inconnu.",
        )

    return {
        "username": payload["sub"],
        "role": role,
    }


def require_role(minimum_role):
    if minimum_role not in ROLE_LEVEL:
        raise ValueError("Rôle inconnu.")

    def dependency(identity=Depends(current_identity)):
        if ROLE_LEVEL[identity["role"]] < ROLE_LEVEL[minimum_role]:
            raise HTTPException(
                status_code=403,
                detail="Permissions insuffisantes.",
            )

        return identity

    return dependency


class InMemoryRateLimiter:
    """Pour le développement mono-processus uniquement."""

    def __init__(self, max_requests=60, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.events = defaultdict(deque)

    async def __call__(self, request: Request):
        client_ip = (
            request.client.host if request.client else "unknown"
        )
        now = time.monotonic()
        queue = self.events[client_ip]

        while queue and now - queue[0] >= self.window_seconds:
            queue.popleft()

        if len(queue) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail="Trop de requêtes. Réessayez plus tard.",
            )

        queue.append(now)


def validate_public_http_url(url):
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL HTTP/HTTPS absolue obligatoire.")

    if parsed.username or parsed.password:
        raise ValueError("Identifiants dans l'URL interdits.")

    host = parsed.hostname.rstrip(".").lower()

    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("Hôte local interdit.")

    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    host,
                    parsed.port or (
                        443 if parsed.scheme == "https" else 80
                    ),
                )
            }
        except socket.gaierror as exc:
            raise ValueError("Nom d'hôte impossible à résoudre.") from exc

    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Adresse IP non publique interdite.")

    return url


def safe_fetch(url, max_bytes=2_000_000):
    """Exemple défensif; une protection réseau complémentaire reste nécessaire."""
    for _ in range(3):
        validate_public_http_url(url)

        with requests.get(
            url,
            timeout=(4, 12),
            allow_redirects=False,
            stream=True,
            headers={"User-Agent": "PlagiatGuardResearch/1.0"},
        ) as response:
            if 300 <= response.status_code < 400:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Redirection sans destination.")
                url = urljoin(url, location)
                continue

            response.raise_for_status()
            chunks = []
            total = 0

            for chunk in response.iter_content(65536):
                total += len(chunk)

                if total > max_bytes:
                    raise ValueError("Réponse distante trop volumineuse.")

                chunks.append(chunk)

            return b"".join(chunks)

    raise ValueError("Trop de redirections.")
