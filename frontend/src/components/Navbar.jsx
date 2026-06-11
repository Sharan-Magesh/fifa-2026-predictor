import { NavLink } from 'react-router-dom'

const TABS = [
  { to: '/',       label: 'Home'   },
  { to: '/match',  label: 'Match'  },
  { to: '/team',   label: 'Team'   },
  { to: '/player', label: 'Player' },
]

export default function Navbar() {
  return (
    <nav className="nav">
      <NavLink to="/" className="nav-logo">
        <span className="nav-logo-main">FIFA</span>
        <span className="nav-logo-sub">2026</span>
      </NavLink>

      <div className="nav-links">
        {TABS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            {label}
          </NavLink>
        ))}
      </div>

      <span className="nav-badge">Predictor</span>
    </nav>
  )
}
