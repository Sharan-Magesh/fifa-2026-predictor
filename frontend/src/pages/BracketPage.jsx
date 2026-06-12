import { useState, useEffect, useRef } from 'react'
import { API, flag } from '../utils'

// Order in which the bracket reveals — drives both the progress strip
// and which sections are mounted (each mount triggers its CSS reveal animation).
const STAGE_KEYS = ['groups', 'round_of_32', 'round_of_16', 'quarterfinal', 'semifinal', 'final', 'champion']
const STAGE_LABELS = {
  groups:        'Group Stage',
  round_of_32:   'Round of 32',
  round_of_16:   'Round of 16',
  quarterfinal:  'Quarterfinals',
  semifinal:     'Semifinals',
  final:         'Final',
  champion:      'Champion',
}

const REVEAL_DELAY_MS = 1100

function GroupCard({ name, standings, bestThirds, index }) {
  return (
    <div className="bgroup" style={{ animationDelay: `${index * 0.05}s` }}>
      <div className="bgroup-title">Group {name}</div>
      {standings.map((row, i) => {
        const qualifies = i < 2 || bestThirds.includes(row.team)
        return (
          <div key={row.team} className={`bgroup-row${qualifies ? ' q' : ''}`}>
            <span className="bgr-pos">{row.position}</span>
            <span className="bgr-flag">{flag(row.team)}</span>
            <span>{row.team}</span>
            <span className="bgr-pts">{row.pts} pts</span>
          </div>
        )
      })}
    </div>
  )
}

function MatchCard({ teamA, teamB, winner, index }) {
  return (
    <div className="bracket-match" style={{ animationDelay: `${index * 0.12}s` }}>
      <div className={`bm-team${winner === teamA ? ' winner' : ' loser'}`}>
        <span className="bm-flag">{flag(teamA)}</span>
        <span>{teamA}</span>
      </div>
      <div className={`bm-team${winner === teamB ? ' winner' : ' loser'}`}>
        <span className="bm-flag">{flag(teamB)}</span>
        <span>{teamB}</span>
      </div>
    </div>
  )
}

export default function BracketPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [revealStage, setRevealStage] = useState(-1)
  const timerRef = useRef(null)

  const runSimulation = async () => {
    setLoading(true); setError(null); setData(null); setRevealStage(-1)
    if (timerRef.current) clearInterval(timerRef.current)

    try {
      const r = await fetch(`${API}/api/simulation/run`)
      const d = await r.json()
      if (!r.ok) throw new Error(d.detail ?? r.statusText)
      setData(d)
    } catch (e) {
      setError(e.message)
      setLoading(false)
      return
    }

    setLoading(false)
    setRevealStage(0)
  }

  useEffect(() => {
    runSimulation()
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Step through the reveal sequence once data has arrived.
  useEffect(() => {
    if (!data || revealStage < 0) return
    if (revealStage >= STAGE_KEYS.length - 1) return

    timerRef.current = setTimeout(() => {
      setRevealStage(s => s + 1)
    }, REVEAL_DELAY_MS)

    return () => clearTimeout(timerRef.current)
  }, [data, revealStage])

  return (
    <div className="page">
      <div className="bracket-head">
        <div>
          <h1 className="pg-title">Live Bracket <span className="accent">Simulation</span></h1>
          <p className="pg-sub">One full WC 2026 tournament · official FIFA knockout bracket · round by round</p>
        </div>
        <button className="btn-claude" onClick={runSimulation} disabled={loading}>
          {loading ? 'Simulating…' : 'Run New Simulation'}
        </button>
      </div>

      {error && <div className="err-box">{error}</div>}

      {loading && (
        <div className="pixel-loader">
          <div className="pixel-loader-ball" />
          <div className="pixel-loader-text">Kicking off simulation…</div>
        </div>
      )}

      {data && (
        <>
          {/* Progress strip */}
          <div className="sim-progress">
            {STAGE_KEYS.map((key, i) => (
              <div
                key={key}
                className={`sim-step${i === revealStage ? ' active' : ''}${i < revealStage ? ' done' : ''}`}
              >
                {STAGE_LABELS[key]}
              </div>
            ))}
          </div>

          {/* Group stage */}
          {revealStage >= 0 && (
            <>
              <p className="section-label">Group Stage — final standings</p>
              <div className="bracket-groups">
                {Object.entries(data.groups).map(([name, g], i) => (
                  <GroupCard
                    key={name}
                    name={name}
                    standings={g.standings}
                    bestThirds={data.best_thirds}
                    index={i}
                  />
                ))}
              </div>
            </>
          )}

          {/* Knockout rounds */}
          {revealStage >= 1 && (
            <>
              <p className="section-label">Knockout Bracket</p>
              <div className="bracket-rounds">
                {data.rounds.map((round) => {
                  const stageIndex = STAGE_KEYS.indexOf(round.key)
                  if (stageIndex > revealStage) return null
                  return (
                    <div className="bracket-col" key={round.key}>
                      <div className="bracket-col-title">{round.name}</div>
                      {round.matches.map((m, mi) => (
                        <MatchCard
                          key={`${m.team_a}-${m.team_b}`}
                          teamA={m.team_a}
                          teamB={m.team_b}
                          winner={m.winner}
                          index={mi}
                        />
                      ))}
                    </div>
                  )
                })}
              </div>
            </>
          )}

          {/* Champion */}
          {revealStage >= STAGE_KEYS.length - 1 && (
            <div className="champion-card" style={{ position: 'relative' }}>
              <div className="confetti" aria-hidden="true">
                {Array.from({ length: 28 }, (_, i) => (
                  <i
                    key={i}
                    style={{
                      left: `${(i * 37) % 100}%`,
                      background: ['#ffd23f', '#00e5ff', '#ff2ee6', '#39ff88'][i % 4],
                      '--cf-dur': `${2.1 + (i % 5) * 0.35}s`,
                      '--cf-delay': `${(i % 7) * 0.3}s`,
                    }}
                  />
                ))}
              </div>
              <div className="champion-trophy">🏆</div>
              <div className="champion-label">2026 World Cup Champion</div>
              <div className="champion-flag">{flag(data.champion)}</div>
              <div className="champion-name">{data.champion}</div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
