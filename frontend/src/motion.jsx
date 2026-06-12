// Shared motion primitives — tiny, dependency-free, GPU-friendly.
// Everything respects prefers-reduced-motion.
import { useEffect, useRef, useState } from 'react'

export const reducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/**
 * Count-up hook: animates a number from 0 to `target` with ease-out cubic.
 * Returns the current animated value. Re-runs when `target` changes.
 */
export function useCountUp(target, { duration = 900, decimals = 1 } = {}) {
  const [value, setValue] = useState(0)
  const rafRef = useRef(null)

  useEffect(() => {
    if (target == null || Number.isNaN(target)) { setValue(0); return }
    if (reducedMotion()) { setValue(target); return }

    const start = performance.now()
    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      setValue(target * eased)
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [target, duration])

  return Number(value.toFixed(decimals))
}

/** Animated percentage text, e.g. <CountPct value={0.234} /> -> "23.4%" */
export function CountPct({ value, decimals = 1 }) {
  const v = useCountUp((value ?? 0) * 100, { decimals })
  return <>{value != null ? `${v.toFixed(decimals)}%` : '—'}</>
}

/** Animated plain number, e.g. Elo rating. */
export function CountNum({ value, decimals = 0, duration = 900 }) {
  const v = useCountUp(value ?? 0, { decimals, duration })
  return <>{v.toFixed(decimals)}</>
}

/**
 * Bar that animates its fill width from 0 on mount / when `frac` changes.
 * frac: 0..1. Color via `color` (css value).
 */
export function Bar({ frac, color = 'var(--claude)', height = 8, glow = true }) {
  const [w, setW] = useState(0)
  useEffect(() => {
    if (reducedMotion()) { setW(frac); return }
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setW(frac)))
    return () => cancelAnimationFrame(id)
  }, [frac])

  return (
    <div className="m-track" style={{ height }}>
      <div
        className="m-fill"
        style={{
          width: `${Math.min(Math.max(w ?? 0, 0), 1) * 100}%`,
          background: color,
          boxShadow: glow ? `0 0 8px ${color}` : 'none',
        }}
      />
    </div>
  )
}

/**
 * Scroll-reveal wrapper: children fade-slide in when entering the viewport.
 * `delay` (ms) lets callers stagger siblings.
 */
export function Reveal({ children, delay = 0, as: Tag = 'div', className = '', ...rest }) {
  const ref = useRef(null)
  const [shown, setShown] = useState(false)

  useEffect(() => {
    if (reducedMotion()) { setShown(true); return }
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) { setShown(true); io.disconnect() }
      },
      { threshold: 0.08, rootMargin: '0px 0px -32px 0px' }
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  return (
    <Tag
      ref={ref}
      className={`reveal${shown ? ' reveal-in' : ''} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
      {...rest}
    >
      {children}
    </Tag>
  )
}
