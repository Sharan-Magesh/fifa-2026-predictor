import { NavLink } from 'react-router-dom'

const TABS = [
  { to: '/',        label: 'Home'    },
  { to: '/match',   label: 'Match'   },
  { to: '/team',    label: 'Team'    },
  { to: '/player',  label: 'Player'  },
  { to: '/bracket', label: 'Bracket', live: true },
]

export default function Navbar() {
  return (
    <nav className="nav">
      <NavLink to="/" className="nav-logo">
        <span className="nav-logo-main">FIFA</span>
        <span className="nav-logo-sub">2026</span>
      </NavLink>

      <div className="nav-links">
        {TABS.map(({ to, label, live }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}${live ? ' nav-link-live' : ''}`}
          >
            {label}
            {live && <span className="live-dot" />}
          </NavLink>
        ))}
      </div>

      <span className="nav-badge">Predictor</span>
    </nav>
  )
}
