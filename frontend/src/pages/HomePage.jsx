import { useState, useEffect } from 'react'
import { API, flag, pct } from '../utils'

const MEDAL = ['1ST', '2ND', '3RD']
const MEDAL_CLS = ['gold', 'silver', 'bronze']

function PodiumCard({ team, rank }) {
  return (
    <div className={`podium-card ${MEDAL_CLS[rank]}`}>
      <div className="pod-pos">{MEDAL[rank]} Place</div>
      <span className="pod-flag">{flag(team.team)}</span>
      <div className="pod-name">{team.team}</div>
      <div className="pod-grp">Group {team.group}</div>
      <div className="pod-pct">{pct(team.p_win_tournament)}</div>
      <div className="pod-lbl">Win probability</div>
    </div>
  )
}

export default function HomePage() {
  const [teams, setTeams] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API}/api/homepage/teams`)
      .then(r => r.ok ? r.json() : r.json().then(d => Promise.reject(d.detail)))
      .then(d => setTeams(d.teams ?? []))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="page"><div className="spinner" /></div>

  const maxWin = teams[0]?.p_win_tournament || 0.01

  return (
    <div className="page">
      <div className="home-head">
        <h1 className="pg-title">FIFA World Cup <span className="accent">2026</span></h1>
        <p className="pg-sub">AI-powered predictor · 100,000 Monte Carlo simulations · XGBoost model</p>
      </div>

      {error && <div className="err-box">{error}</div>}

      {teams.length > 0 && (
        <>
          <p className="section-label">Top contenders</p>
          <div className="podium">
            {teams.slice(0, 3).map((t, i) => (
              <PodiumCard key={t.team} team={t} rank={i} />
            ))}
          </div>

          <p className="section-label">All 48 teams — ranked by win probability</p>
          <div className="team-list">
            {teams.map((t, i) => (
              <div key={t.team} className={`team-row${i < 3 ? ' top3' : ''}`}>
                <span className="tr-rank">#{i + 1}</span>
                <span className="tr-flag">{flag(t.team)}</span>
                <span className="tr-name">{t.team}</span>
                <span className="tr-grp">Grp {t.group}</span>
                <div className="tr-bar">
                  <div className="tr-fill" style={{ width: `${(t.p_win_tournament / maxWin) * 100}%` }} />
                </div>
                <span className="tr-pct">{pct(t.p_win_tournament)}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
