import { useState, useEffect } from 'react'
import { API, flag, pct } from '../utils'

const OUTCOME_BARS = (res, teamA, teamB) => {
  const op = res.outcome_probabilities
  return [
    { label: teamA + ' Win', val: op.team_a_win, color: 'var(--gold)' },
    op.draw != null
      ? { label: 'Draw', val: op.draw, color: 'var(--cyan)' }
      : { label: 'Extra Time / Pens', val: op.extra_time_or_pens, color: 'var(--cyan)' },
    { label: teamB + ' Win', val: op.team_b_win, color: 'var(--gold)' },
  ].filter(x => x.val != null)
}

export default function MatchPage() {
  const [teams, setTeams] = useState([])
  const [teamA, setTeamA] = useState('')
  const [teamB, setTeamB] = useState('')
  const [knockout, setKnockout] = useState(false)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API}/api/match/teams`)
      .then(r => r.json())
      .then(d => setTeams(d.teams ?? []))
  }, [])

  const predict = async () => {
    setLoading(true); setError(null); setResult(null)
    try {
      const url = `${API}/api/match/predict?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}&knockout=${knockout}`
      const r = await fetch(url)
      const d = await r.json()
      if (!r.ok) throw new Error(d.detail ?? r.statusText)
      setResult(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const sc  = result?.most_likely_scoreline
  const xg  = result?.expected_goals
  const bars = result ? OUTCOME_BARS(result, teamA, teamB) : []

  return (
    <div className="page">
      <h1 className="pg-title">Match <span className="accent">Predictor</span></h1>
      <p className="pg-sub">XGBoost · Bivariate Poisson · 15 engineered features</p>

      <div style={{ marginTop: 32 }}>
        <div className="match-grid">
          {/* Team A */}
          <div className="match-panel">
            <div className="mp-label">Team A</div>
            <select
              className="fifa-select"
              value={teamA}
              onChange={e => { setTeamA(e.target.value); setResult(null) }}
            >
              <option value="">— Select team —</option>
              {teams.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            {teamA && (
              <>
                <span className="mp-flag">{flag(teamA)}</span>
                <div className="mp-name">{teamA}</div>
              </>
            )}
          </div>

          <div className="mp-vs">VS</div>

          {/* Team B */}
          <div className="match-panel">
            <div className="mp-label">Team B</div>
            <select
              className="fifa-select"
              value={teamB}
              onChange={e => { setTeamB(e.target.value); setResult(null) }}
            >
              <option value="">— Select team —</option>
              {teams.filter(t => t !== teamA).map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            {teamB && (
              <>
                <span className="mp-flag">{flag(teamB)}</span>
                <div className="mp-name">{teamB}</div>
              </>
            )}
          </div>
        </div>

        {/* Knockout toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20, justifyContent: 'center' }}>
          <label style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 2, color: 'var(--muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}>
            <input
              type="checkbox"
              checked={knockout}
              onChange={e => { setKnockout(e.target.checked); setResult(null) }}
              style={{ width: 14, height: 14, accentColor: 'var(--gold)' }}
            />
            Knockout stage (no draw)
          </label>
        </div>

        <div className="match-cta">
          <button
            className="btn-gold"
            onClick={predict}
            disabled={!teamA || !teamB || loading}
          >
            {loading ? 'Predicting…' : 'Predict Match'}
          </button>
        </div>

        {error && <div className="err-box">{error}</div>}

        {result && (
          <div className="result-card">
            {/* Scoreline */}
            <div className="res-score">
              <div className="res-digits">{sc.team_a_goals} – {sc.team_b_goals}</div>
              <div className="res-teams">
                {teamA} vs {teamB} · Most likely scoreline · {pct(sc.probability)} probability
              </div>
            </div>

            {/* xG */}
            <div className="res-xg">
              <div className="xg-block">
                <div className="xg-val">{xg.team_a_xg}</div>
                <div className="xg-lbl">{teamA} xG</div>
              </div>
              <div style={{ width: 1, background: 'var(--border)' }} />
              <div className="xg-block">
                <div className="xg-val">{xg.team_b_xg}</div>
                <div className="xg-lbl">{teamB} xG</div>
              </div>
            </div>

            {/* Outcome bars */}
            <p className="section-label">Outcome probabilities</p>
            <div className="res-bars">
              {bars.map(bar => (
                <div key={bar.label} className="res-prob-row">
                  <span className="rplab">{bar.label}</span>
                  <div className="rptrack">
                    <div className="rpfill" style={{ width: `${bar.val * 100}%`, background: bar.color }} />
                  </div>
                  <span className="rppct" style={{ color: bar.color }}>{pct(bar.val, 0)}</span>
                </div>
              ))}
            </div>

            {result.penalty_probability != null && (
              <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border)', fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1.5, color: 'var(--muted)' }}>
                Penalty shootout probability: <span style={{ color: 'var(--gold)' }}>{pct(result.penalty_probability)}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
