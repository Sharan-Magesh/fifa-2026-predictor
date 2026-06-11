import { useState, useEffect } from 'react'
import { API, flag, pct } from '../utils'

const PROB_META = [
  { key: 'p_win_tournament', label: 'Win Tournament' },
  { key: 'p_final',          label: 'Reach Final'    },
  { key: 'p_advance_sf',     label: 'Reach Semi-final'},
  { key: 'p_advance_qf',     label: 'Reach Quarter-final' },
]

export default function TeamPage() {
  const [teams,    setTeams]    = useState([])
  const [selected, setSelected] = useState('')
  const [data,     setData]     = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)

  useEffect(() => {
    fetch(`${API}/api/match/teams`)
      .then(r => r.json())
      .then(d => setTeams(d.teams ?? []))
  }, [])

  const load = async (name) => {
    if (!name) { setData(null); return }
    setLoading(true); setError(null)
    try {
      const r = await fetch(`${API}/api/team/${encodeURIComponent(name)}`)
      const d = await r.json()
      if (!r.ok) throw new Error(d.detail ?? r.statusText)
      setData(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const probs = data?.tournament_probabilities ?? {}
  const elo   = data?.elo_trajectory?.[0]?.elo

  return (
    <div className="page">
      <h1 className="pg-title">Team <span className="accent">Profile</span></h1>
      <p className="pg-sub">Bracket path · Squad depth · Key players · Elo rating</p>

      <div style={{ marginTop: 32 }}>
        <div className="team-select-row">
          <select
            className="fifa-select"
            value={selected}
            onChange={e => { setSelected(e.target.value); load(e.target.value) }}
            style={{ maxWidth: 360 }}
          >
            <option value="">— Select a team —</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>

        {loading && <div className="spinner" />}
        {error   && <div className="err-box">{error}</div>}

        {data && !loading && (
          <>
            {/* Hero */}
            <div className="team-hero">
              <span className="th-flag">{flag(data.team)}</span>
              <div className="th-info">
                <div className="th-name">{data.team}</div>
                <div className="th-meta">
                  Group {data.group}
                  {data.group_opponents?.length > 0 && (
                    <> · vs {data.group_opponents.join(', ')}</>
                  )}
                </div>
              </div>
              {elo != null && (
                <div className="th-elo">
                  <div className="th-elo-val">{Math.round(elo)}</div>
                  <div className="th-elo-lbl">Elo Rating</div>
                </div>
              )}
            </div>

            {/* 2-column: probs + key players */}
            <div className="team-cols">
              <div className="t-card">
                <div className="t-card-title">Tournament probabilities</div>
                <div className="t-probs">
                  {PROB_META.map(({ key, label }) =>
                    probs[key] != null ? (
                      <div key={key} className="pbar">
                        <div className="pbar-head">
                          <span>{label}</span>
                          <span style={{ color: 'var(--gold)' }}>{pct(probs[key])}</span>
                        </div>
                        <div className="pbar-track">
                          <div className="pbar-fill" style={{ width: `${probs[key] * 100}%` }} />
                        </div>
                      </div>
                    ) : null
                  )}
                </div>
              </div>

              <div className="t-card">
                <div className="t-card-title">Key players</div>
                {data.key_players?.length > 0 ? (
                  data.key_players.map((p, i) => (
                    <div key={i} className="kp-row">
                      <span className="kp-pos">{(p.position ?? '??').slice(0, 3)}</span>
                      <span className="kp-name">{p.player ?? p.player_name ?? '—'}</span>
                      {p.composite_score != null && (
                        <span className="kp-score">{(p.composite_score * 100).toFixed(0)}</span>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="empty-state" style={{ padding: '20px 0' }}>No data yet</div>
                )}
              </div>
            </div>

            {/* Squad table */}
            {data.squad?.length > 0 && (
              <div className="squad-wrap">
                <div className="t-card-title" style={{ marginBottom: 16 }}>
                  Full squad — {data.squad.length} players
                </div>
                <table className="squad-table">
                  <thead>
                    <tr>
                      <th>Pos</th>
                      <th>Player</th>
                      <th>Club</th>
                      <th>Age</th>
                      <th>Caps</th>
                      <th>Goals</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.squad.map((p, i) => (
                      <tr key={i}>
                        <td style={{ color: 'var(--gold)', fontSize: 12 }}>{p.position}</td>
                        <td style={{ fontWeight: 700 }}>{p.player_name}</td>
                        <td style={{ color: 'var(--muted)', fontSize: 13 }}>{p.club}</td>
                        <td style={{ color: 'var(--muted)' }}>{p.age}</td>
                        <td style={{ color: 'var(--cyan)' }}>{p.caps}</td>
                        <td style={{ color: 'var(--muted)' }}>{p.goals}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
