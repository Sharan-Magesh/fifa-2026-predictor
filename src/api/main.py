"""
FIFA 2026 World Cup Predictor — FastAPI Backend
Entry point. Mounts all routers and configures CORS for the React frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import homepage, match, team, player

app = FastAPI(
    title="FIFA 2026 World Cup Predictor",
    description="ML-powered predictions for WC 2026: match outcomes, player impact, bracket paths.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — React dev server runs on localhost:3000 by default.
# In production replace "*" with your deployed frontend URL.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
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


@app.get("/")
def root():
    return {"status": "ok", "message": "FIFA 2026 Predictor API is running."}