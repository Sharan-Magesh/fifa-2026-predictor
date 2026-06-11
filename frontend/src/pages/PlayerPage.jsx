import { useState, useRef } from 'react'
import {
  RadarChart, PolarGrid, PolarAngleAxis,
  Radar, ResponsiveContainer, Legend,
} from 'recharts'
import { API, pct } from '../utils'

const RADAR_KEYS = {
  international_xg: 'Intl xG',
  club_form:        'Club Form',
  experience:       'Experience',
  composite_score:  'Overall',
  shot_quality:     'Shot Quality',
  involvement:      'Involvement',
}

const BREAKDOWN_ITEMS = [
  { key: 'intl_xg_contribution',    label: 'International xG', max: 0.45 },
  { key: 'club_form_contribution',  label: 'Club Form',        max: 0.35 },
  { key: 'experience_contribution', label: 'Experience',       max: 0.20 },
]

function PlayerSearch({ label, onSelect }) {
  const [q,       setQ]       = useState('')
  const [results, setResults] = useState([])
  const [busy,    setBusy]    = useState(false)
  const [chosen,  setChosen]  = useState('')

  const search = async () => {
    const q2 = q.trim(); if (!q2) return
    setBusy(true)
    try {
      const r = await fetch(`${API}/api/player/search?q=${encodeURIComponent(q2)}`)
      const d = await r.json()
      setResults(d.results ?? [])
    } catch { setResults([]) }
    finally  { setBusy(false) }
  }

  const pick = (name) => {
    setChosen(name); setQ(name); setResults([]); onSelect(name)
  }

  return (
    <div className="search-area">
      <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 3, color: 'var(--muted)', marginBottom: 8 }}>
        {label}
      </div>
      <div className="search-row">
        <input
          className="fifa-input"
          value={q}
          placeholder="Type name…"
          onChange={e => { setQ(e.target.value); if (!e.target.value) { setChosen(''); onSelect('') } }}
          onKeyDown={e => e.key === 'Enter' && search()}
        />
        <button
          className="btn-gold"
          style={{ padding: '11px 18px', clipPath: 'none', fontSize: 13 }}
          onClick={search}
          disabled={busy}
        >
          {busy ? '…' : 'Search'}
        </button>
      </div>
      {results.length > 0 && (
        <div className="search-results">
          {results.map(r => (
            <div key={r.player} className="search-result-item" onClick={() => pick(r.player)}>
              {r.player}
              <span className="sri-team">{r.team} · {r.position}</span>
            </div>
          ))}
        </div>
      )}
      {chosen && results.length === 0 && (
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--gold)', letterSpacing: 2, textTransform: 'uppercase', marginTop: 6 }}>
          Selected: {chosen}
        </div>
      )}
    </div>
  )
}

function PlayerCard({ profile, color }) {
  const bd = profile.breakdown ?? {}
  return (
    <div className="p-profile">
      <div className="p-name">{profile.player}</div>
      <div className="p-team">{profile.team} · {profile.position}</div>
      <div className="p-score" style={{ color }}>{(profile.composite_score * 100).toFixed(1)}</div>
      <div className="p-score-lbl">Composite score</div>
      <div className="p-breakdown">
        {BREAKDOWN_ITEMS.map(({ key, label, max }) => {
          const val = bd[key] ?? 0
          return (
            <div key={key} className="pbar">
              <div className="pbar-head">
                <span>{label}</span>
                <span style={{ color }}>{(val * 100).toFixed(1)}</span>
              </div>
              <div className="pbar-track">
                <div className="pbar-fill" style={{ width: `${(val / max) * 100}%`, background: color }} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function buildRadarData(ra, rb) {
  return Object.entries(RADAR_KEYS).map(([k, label]) => ({
    subject: label,
    A: parseFloat((( ra[k] ?? 0) * 100).toFixed(1)),
    B: parseFloat((( rb[k] ?? 0) * 100).toFixed(1)),
  }))
}

export default function PlayerPage() {
  const [playerA, setPlayerA] = useState('')
  const [playerB, setPlayerB] = useState('')
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  const compare = async () => {
    if (!playerA || !playerB) return
    setLoading(true); setError(null); setResult(null)
    try {
      const url = `${API}/api/player/compare?player_a=${encodeURIComponent(playerA)}&player_b=${encodeURIComponent(playerB)}`
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

  const radarData = result
    ? buildRadarData(result.player_a.radar, result.player_b.radar)
    : []

  return (
    <div className="page">
      <h1 className="pg-title">Player <span className="accent">Comparison</span></h1>
      <p className="pg-sub">Head-to-head · Radar chart · Composite score breakdown</p>

      <div style={{ marginTop: 32 }}>
        <div className="player-grid">
          {/* Panel A */}
          <div className="player-panel">
            <PlayerSearch label="Player A" onSelect={setPlayerA} />
            {result && <PlayerCard profile={result.player_a} color="var(--gold)" />}
          </div>

          {/* Middle column */}
          <div className="p-vs-col">
            <div className="p-vs-text">VS</div>
            <button
              className="btn-gold"
              style={{ padding: '12px 16px', clipPath: 'none', fontSize: 13, marginTop: 8 }}
              onClick={compare}
              disabled={!playerA || !playerB || loading}
            >
              {loading ? '…' : 'Compare'}
            </button>

            {result && (
              <div style={{ marginTop: 24, textAlign: 'center' }}>
                <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 3, color: 'var(--muted)', marginBottom: 8 }}>
                  Advantage
                </div>
                <div className={`advantage-badge ${result.advantage === 'player_a' ? 'a' : 'b'}`}>
                  {result.advantage === 'player_a'
                    ? result.player_a.player
                    : result.player_b.player}
                  <br />
                  <span style={{ fontSize: 11, opacity: 0.8 }}>
                    +{(result.score_delta * 100).toFixed(1)} pts
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Panel B */}
          <div className="player-panel">
            <PlayerSearch label="Player B" onSelect={setPlayerB} />
            {result && <PlayerCard profile={result.player_b} color="var(--cyan)" />}
          </div>
        </div>

        {error && <div className="err-box">{error}</div>}

        {/* Radar chart */}
        {result && radarData.length > 0 && (
          <div className="radar-card">
            <p className="section-label">Attribute radar</p>
            <ResponsiveContainer width="100%" height={340}>
              <RadarChart data={radarData} margin={{ top: 16, right: 48, bottom: 16, left: 48 }}>
                <PolarGrid stroke="rgba(255,255,255,0.10)" />
                <PolarAngleAxis
                  dataKey="subject"
                  tick={{
                    fill: 'rgba(255,255,255,0.55)',
                    fontSize: 12,
                    fontWeight: 700,
                    fontFamily: 'Barlow Condensed, sans-serif',
                    textTransform: 'uppercase',
                  }}
                />
                <Radar
                  name={result.player_a.player}
                  dataKey="A"
                  stroke="#f5c518"
                  fill="#f5c518"
                  fillOpacity={0.18}
                  strokeWidth={2}
                />
                <Radar
                  name={result.player_b.player}
                  dataKey="B"
                  stroke="#17c1e8"
                  fill="#17c1e8"
                  fillOpacity={0.18}
                  strokeWidth={2}
                />
                <Legend
                  wrapperStyle={{
                    fontFamily: 'Barlow Condensed, sans-serif',
                    fontSize: 13,
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    letterSpacing: 1.5,
                    color: 'rgba(255,255,255,0.7)',
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}
