export const API = 'http://localhost:8000'

export const FLAGS = {
  Algeria: '🇩🇿', Argentina: '🇦🇷', Australia: '🇦🇺', Austria: '🇦🇹',
  Belgium: '🇧🇪', 'Bosnia and Herzegovina': '🇧🇦', Brazil: '🇧🇷',
  Canada: '🇨🇦', 'Cape Verde': '🇨🇻', Colombia: '🇨🇴', Croatia: '🇭🇷',
  'Curaçao': '🇨🇼', 'Curacao': '🇨🇼', 'Czech Republic': '🇨🇿',
  'DR Congo': '🇨🇩', Ecuador: '🇪🇨', Egypt: '🇪🇬', England: '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  France: '🇫🇷', Germany: '🇩🇪', Ghana: '🇬🇭', Haiti: '🇭🇹',
  Iran: '🇮🇷', Iraq: '🇮🇶', 'Ivory Coast': '🇨🇮', Japan: '🇯🇵',
  Jordan: '🇯🇴', Mexico: '🇲🇽', Morocco: '🇲🇦', Netherlands: '🇳🇱',
  'New Zealand': '🇳🇿', Norway: '🇳🇴', Panama: '🇵🇦', Paraguay: '🇵🇾',
  Portugal: '🇵🇹', Qatar: '🇶🇦', 'Saudi Arabia': '🇸🇦', Scotland: '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
  Senegal: '🇸🇳', 'South Africa': '🇿🇦', 'South Korea': '🇰🇷',
  Spain: '🇪🇸', Sweden: '🇸🇪', Switzerland: '🇨🇭', Tunisia: '🇹🇳',
  Turkey: '🇹🇷', 'United States': '🇺🇸', Uruguay: '🇺🇾', Uzbekistan: '🇺🇿',
}

export const flag = (team) => FLAGS[team] ?? '🌍'

export const pct = (v, decimals = 1) =>
  v != null ? `${(v * 100).toFixed(decimals)}%` : '—'
