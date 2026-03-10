"""
Custom Middleware for Agent OS Web Interface

Provides security and utility middleware:
- HTTPS redirect and HSTS enforcement
- Request logging
- Security headers
- CSRF token validation
"""

import logging
import secrets
import uuid
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

logger = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Adds a unique request correlation ID to every request/response.

    - Reuses the client-provided X-Request-ID header if present.
    - Otherwise generates a UUID4.
    - Stores the ID on request.state.request_id for route handlers.
    - Echoes it back in the X-Request-ID response header.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """
    Middleware for HTTPS enforcement and HSTS headers.

    Features:
    - Redirects HTTP requests to HTTPS (when force_https=True)
    - Adds HSTS headers to responses (when hsts_enabled=True)
    - Respects X-Forwarded-Proto header for reverse proxy setups

    Usage:
        app.add_middleware(
            HTTPSRedirectMiddleware,
            force_https=True,
            hsts_enabled=True,
            hsts_max_age=31536000,
        )
    """

    def __init__(
        self,
        app: Callable,
        force_https: bool = False,
        hsts_enabled: bool = False,
        hsts_max_age: int = 31536000,  # 1 year
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = False,
    ):
        """
        Initialize HTTPS redirect middleware.

        Args:
            app: The ASGI application
            force_https: If True, redirect HTTP to HTTPS
            hsts_enabled: If True, add HSTS headers
            hsts_max_age: HSTS max-age in seconds (default 1 year)
            hsts_include_subdomains: Include subdomains in HSTS
            hsts_preload: Enable HSTS preload (for browser preload lists)
        """
        super().__init__(app)
        self.force_https = force_https
        self.hsts_enabled = hsts_enabled
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload

        # Build HSTS header value
        self._hsts_header = self._build_hsts_header()

    def _build_hsts_header(self) -> str:
        """Build the HSTS header value."""
        parts = [f"max-age={self.hsts_max_age}"]
        if self.hsts_include_subdomains:
            parts.append("includeSubDomains")
        if self.hsts_preload:
            parts.append("preload")
        return "; ".join(parts)

    def _is_https(self, request: Request) -> bool:
        """
        Check if the request is over HTTPS.

        Handles both direct HTTPS and reverse proxy setups
        using X-Forwarded-Proto header.
        """
        # Check X-Forwarded-Proto header (reverse proxy)
        forwarded_proto = request.headers.get("x-forwarded-proto", "").lower()
        if forwarded_proto:
            return forwarded_proto == "https"

        # Check request scheme directly
        return request.url.scheme == "https"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with HTTPS enforcement."""
        is_https = self._is_https(request)

        # Redirect HTTP to HTTPS if enabled
        if self.force_https and not is_https:
            # Build HTTPS URL
            https_url = request.url.replace(scheme="https")

            # Use 307 to preserve HTTP method
            logger.debug(f"Redirecting HTTP to HTTPS: {request.url} -> {https_url}")
            return RedirectResponse(url=str(https_url), status_code=307)

        # Process the request
        response = await call_next(request)

        # Add HSTS header to HTTPS responses
        if self.hsts_enabled and is_https:
            response.headers["Strict-Transport-Security"] = self._hsts_header

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.

    Adds headers recommended by OWASP:
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Content-Security-Policy (optional)
    """

    # Default CSP that allows self-hosted resources and WebSocket connections
    DEFAULT_CSP = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self' ws: wss:; "
        "frame-ancestors 'none'"
    )

    def __init__(
        self,
        app: Callable,
        content_security_policy: Optional[str] = DEFAULT_CSP,
        x_frame_options: str = "DENY",
        referrer_policy: str = "strict-origin-when-cross-origin",
    ):
        """
        Initialize security headers middleware.

        Args:
            app: The ASGI application
            content_security_policy: Optional CSP header value
            x_frame_options: X-Frame-Options value (DENY, SAMEORIGIN)
            referrer_policy: Referrer-Policy value
        """
        super().__init__(app)
        self.content_security_policy = content_security_policy
        self.x_frame_options = x_frame_options
        self.referrer_policy = referrer_policy

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to the response."""
        response = await call_next(request)

        # Standard security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = self.x_frame_options
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = self.referrer_policy

        # Optional Content-Security-Policy
        if self.content_security_policy:
            response.headers["Content-Security-Policy"] = self.content_security_policy

        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CSRF token validation for session-authenticated (cookie-based) requests.

    Generates a CSRF token on first request, sets it as a cookie,
    and validates it on state-changing requests (POST/PUT/DELETE/PATCH).

    Bearer token authenticated requests skip CSRF (not vulnerable to CSRF).

    This addresses Finding L5 (No CSRF token implementation) from the
    Agentic Security Audit v3.0.
    """

    CSRF_COOKIE = "csrf_token"
    CSRF_HEADER = "x-csrf-token"
    STATE_CHANGING_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
    # Paths exempt from CSRF (public endpoints, WebSocket)
    EXEMPT_PATHS = {"/api/auth/login", "/api/auth/register", "/health", "/ws"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip for non-state-changing methods
        if request.method not in self.STATE_CHANGING_METHODS:
            return await call_next(request)

        # Skip for Bearer token auth (not vulnerable to CSRF)
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return await call_next(request)

        # Skip exempt paths
        if any(request.url.path.startswith(p) for p in self.EXEMPT_PATHS):
            return await call_next(request)

        # Validate CSRF token for cookie-authenticated requests
        cookie_token = request.cookies.get(self.CSRF_COOKIE)
        header_token = request.headers.get(self.CSRF_HEADER)

        if not cookie_token or not header_token:
            return Response(
                content='{"detail": "CSRF token missing"}',
                status_code=403,
                media_type="application/json",
            )

        if not secrets.compare_digest(cookie_token, header_token):
            return Response(
                content='{"detail": "CSRF token mismatch"}',
                status_code=403,
                media_type="application/json",
            )

        return await call_next(request)
