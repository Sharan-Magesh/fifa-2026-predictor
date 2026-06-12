"""
FIFA 2026 World Cup Predictor — FastAPI Backend
Entry point. Mounts all routers and configures CORS for the React frontend.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import homepage, match, team, player, simulation

app = FastAPI(
    title="FIFA 2026 World Cup Predictor",
    description="ML-powered predictions for WC 2026: match outcomes, player impact, bracket paths.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — React dev server runs on localhost:3000 by default.
#
# In production, set the ALLOWED_ORIGINS env var to a comma-separated list
# of allowed frontend origins, e.g.:
#   ALLOWED_ORIGINS=https://fifa-2026-predictor.vercel.app,https://my-domain.com
#
# Note: "*" cannot be combined with allow_credentials=True (browsers reject
# wildcard origins on credentialed requests, and FastAPI/Starlette will
# echo "*" back literally, which is a footgun). This API is read-only and
# does not use cookies/auth, so we disable credentials and keep an explicit
# origin allowlist.
# ---------------------------------------------------------------------------
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers — each file owns one page's worth of endpoints
# ---------------------------------------------------------------------------
app.include_router(homepage.router, prefix="/api/homepage", tags=["Homepage"])
app.include_router(match.router,    prefix="/api/match",    tags=["Match"])
app.include_router(team.router,     prefix="/api/team",     tags=["Team"])
app.include_router(player.router,   prefix="/api/player",   tags=["Player"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])


@app.get("/")
def root():
    return {"status": "ok", "message": "FIFA 2026 Predictor API is running."}
