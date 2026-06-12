import { useEffect, lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import Navbar from './components/Navbar'
import HomePage from './pages/HomePage'
import MatchPage from './pages/MatchPage'
import TeamPage from './pages/TeamPage'
import BracketPage from './pages/BracketPage'

// PlayerPage pulls in recharts (~400kB min) — lazy-load it so the other
// pages ship a much smaller initial bundle.
const PlayerPage = lazy(() => import('./pages/PlayerPage'))
import { reducedMotion } from './motion'

// N64-style "motion physics" — every card-like panel tilts toward the
// cursor like a 3D console menu tile, then springs back into place when
// the mouse leaves. Implemented globally with a MutationObserver so it
// works on every page without touching each component.
// Perf: pointer math runs inside requestAnimationFrame (one style write
// per frame max) and the whole system disables itself for touch devices
// and prefers-reduced-motion users.
const TILT_SELECTOR = [
  '.card', '.podium-card', '.team-row', '.t-card', '.player-panel',
  '.match-panel', '.result-card', '.radar-card', '.bgroup',
  '.squad-wrap', '.bracket-match', '.team-hero', '.champion-card',
].join(', ')

const MAX_TILT = 8 // degrees

function useTiltPhysics() {
  useEffect(() => {
    if (reducedMotion() || window.matchMedia('(hover: none)').matches) return

    const tracked = new WeakSet()
    let raf = null
    let pending = null // { el, x, y }

    const flush = () => {
      raf = null
      if (!pending) return
      const { el, x, y } = pending
      const r = el.getBoundingClientRect()
      const px = (x - r.left) / r.width
      const py = (y - r.top) / r.height
      const rotY = (px - 0.5) * MAX_TILT * 2
      const rotX = (0.5 - py) * MAX_TILT * 2
      el.style.transition = 'transform 60ms linear, box-shadow .2s ease, border-color .25s ease'
      el.style.transform = `perspective(900px) rotateX(${rotX.toFixed(2)}deg) rotateY(${rotY.toFixed(2)}deg) translateZ(8px)`
    }

    const handleMove = (e) => {
      pending = { el: e.currentTarget, x: e.clientX, y: e.clientY }
      if (!raf) raf = requestAnimationFrame(flush)
    }

    const handleLeave = (e) => {
      const el = e.currentTarget
      if (pending?.el === el) pending = null
      el.style.transition = 'transform var(--bounce, .55s cubic-bezier(.34,1.56,.64,1)), box-shadow .2s ease, border-color .25s ease'
      el.style.transform = 'perspective(900px) rotateX(0deg) rotateY(0deg) translateZ(0px)'
    }

    const attach = () => {
      document.querySelectorAll(TILT_SELECTOR).forEach((el) => {
        if (tracked.has(el)) return
        tracked.add(el)
        el.classList.add('tilt-ready')
        el.addEventListener('mousemove', handleMove)
        el.addEventListener('mouseleave', handleLeave)
      })
    }

    attach()
    const observer = new MutationObserver(attach)
    observer.observe(document.body, { childList: true, subtree: true })

    return () => {
      observer.disconnect()
      if (raf) cancelAnimationFrame(raf)
    }
  }, [])
}

// Re-mounts the route subtree on navigation so each page plays its
// entrance animation (CSS .route-enter on the wrapper).
function AnimatedRoutes() {
  const location = useLocation()
  return (
    <div key={location.pathname} className="route-enter">
      <Routes location={location}>
        <Route path="/" element={<HomePage />} />
        <Route path="/match" element={<MatchPage />} />
        <Route path="/team" element={<TeamPage />} />
        <Route path="/player" element={
          <Suspense fallback={<div className="page"><div className="spinner" /></div>}>
            <PlayerPage />
          </Suspense>
        } />
        <Route path="/bracket" element={<BracketPage />} />
      </Routes>
    </div>
  )
}

export default function App() {
  useTiltPhysics()

  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Navbar />
      <AnimatedRoutes />
    </BrowserRouter>
  )
}
