"""Optional API authentication, including protected media in the browser."""

import hmac
import secrets
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

COOKIE = "marg_session"
SESSION_SECONDS = 3600


def install_auth(app: FastAPI, token: str | None, secure_cookie: bool = False) -> None:
    sessions: dict[str, float] = {}

    def valid_bearer(request: Request) -> bool:
        authorization = request.headers.get("authorization", "")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        return bool(token and supplied and hmac.compare_digest(supplied, token))

    def authenticated(request: Request) -> bool:
        if not token or valid_bearer(request):
            return True
        session = request.cookies.get(COOKIE, "")
        return bool(session and sessions.get(session, 0) > time.time())

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        path = request.url.path
        public = path in {"/api/health", "/api/capabilities", "/api/session"}
        if path.startswith("/api/") and not public and not authenticated(request):
            return JSONResponse(
                status_code=401,
                content={"detail": "Sign in to access this workspace"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        if path.startswith("/api/") and request.method not in {
            "GET",
            "HEAD",
            "OPTIONS",
        }:
            origin = request.headers.get("origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.netloc != request.headers.get(
                    "host"
                ) or parsed.scheme not in {"http", "https"}:
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Cross-origin writes are not allowed"},
                    )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/session")
    def session_status(request: Request) -> dict[str, bool]:
        return {
            "authenticated": authenticated(request),
            "authentication_required": bool(token),
        }

    @app.post("/api/session")
    def create_session(request: Request, response: Response) -> dict[str, bool]:
        if token and not valid_bearer(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid workspace access token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if token:
            for key, expiry in list(sessions.items()):
                if expiry <= time.time():
                    sessions.pop(key, None)
            if len(sessions) >= 1000:
                raise HTTPException(
                    status_code=429, detail="Too many active sessions; retry later"
                )
            session = secrets.token_urlsafe(32)
            sessions[session] = time.time() + SESSION_SECONDS
            response.set_cookie(
                COOKIE,
                session,
                httponly=True,
                secure=secure_cookie,
                samesite="strict",
                max_age=SESSION_SECONDS,
                path="/api",
            )
        return {"authenticated": True}

    @app.delete("/api/session")
    def delete_session(request: Request, response: Response) -> dict[str, bool]:
        sessions.pop(request.cookies.get(COOKIE, ""), None)
        response.delete_cookie(COOKIE, path="/api")
        return {"authenticated": False}
