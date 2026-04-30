import { useState } from 'react'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const { login } = useAuth()
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(email.trim(), password)
    } catch (err) {
      const msg = err?.response?.data?.detail
      setError(typeof msg === 'string' ? msg : 'Invalid email or password.')
    } finally {
      setLoading(false)
    }
  }

  function fillDemo() {
    setEmail('admin@gmail.com')
    setPassword('admin123')
    setError('')
  }

  return (
    <div style={styles.root}>
      {/* Left panel */}
      <div style={styles.left}>
        <div style={styles.brand}>
          <div style={styles.brandIcon}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <span style={styles.brandName}>FraudLens</span>
        </div>

        <div style={styles.leftContent}>
          <h1 style={styles.headline}>AI-Powered<br/>Claim Verification</h1>
          <p style={styles.sub}>Multi-agent fraud detection and consensus-based processing for insurance claims.</p>

          <div style={styles.features}>
            {['Real-time fraud detection', 'Blockchain audit trail', 'IPFS document storage', 'Multi-agent consensus'].map(f => (
              <div key={f} style={styles.feature}>
                <div style={styles.featureDot} />
                <span>{f}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={styles.demoSection}>
          <p style={styles.demoLabel}>Demo account</p>
          <div style={styles.demoRow}>
            <button style={styles.demoBtn} onClick={fillDemo}>
              <span style={styles.demoBadge}>Insurer</span>
              admin@gmail.com
            </button>
          </div>
        </div>
      </div>

      {/* Right panel */}
      <div style={styles.right}>
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>Sign in</h2>
          <p style={styles.cardSub}>Enter your credentials to access the dashboard</p>

          <form onSubmit={handleSubmit} style={styles.form}>
            <div style={styles.field}>
              <label style={styles.label}>Email</label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@claimguard.com"
                required
                style={styles.input}
                autoComplete="email"
              />
            </div>

            <div style={styles.field}>
              <label style={styles.label}>Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                style={styles.input}
                autoComplete="current-password"
              />
            </div>

            {error && <div style={styles.error}>{error}</div>}

            <button type="submit" disabled={loading} style={loading ? styles.btnDisabled : styles.btn}>
              {loading ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          <div style={styles.hint}>
            <span style={styles.hintDot} />
            <span style={{color: 'var(--text-muted)', fontSize: '12px'}}>
              Use the demo buttons on the left to auto-fill credentials
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  root: {
    display: 'flex',
    minHeight: '100vh',
  },
  left: {
    width: '420px',
    flexShrink: 0,
    background: 'var(--accent)',
    color: '#fff',
    display: 'flex',
    flexDirection: 'column',
    padding: '32px',
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '48px',
  },
  brandIcon: {
    width: '40px',
    height: '40px',
    background: 'rgba(255,255,255,0.12)',
    borderRadius: '10px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandName: {
    fontSize: '20px',
    fontWeight: '700',
    letterSpacing: '-0.3px',
  },
  leftContent: {
    flex: 1,
  },
  headline: {
    fontSize: '32px',
    fontWeight: '700',
    lineHeight: 1.2,
    marginBottom: '16px',
    letterSpacing: '-0.5px',
  },
  sub: {
    fontSize: '15px',
    color: 'rgba(255,255,255,0.65)',
    lineHeight: 1.6,
    marginBottom: '32px',
  },
  features: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  feature: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '14px',
    color: 'rgba(255,255,255,0.8)',
  },
  featureDot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: '#63b3ed',
    flexShrink: 0,
  },
  demoSection: {
    borderTop: '1px solid rgba(255,255,255,0.12)',
    paddingTop: '24px',
  },
  demoLabel: {
    fontSize: '11px',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: 'rgba(255,255,255,0.4)',
    marginBottom: '12px',
  },
  demoRow: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  demoBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    background: 'rgba(255,255,255,0.06)',
    border: '1px solid rgba(255,255,255,0.12)',
    borderRadius: '8px',
    padding: '10px 14px',
    color: 'rgba(255,255,255,0.8)',
    fontSize: '13px',
    cursor: 'pointer',
    textAlign: 'left',
    transition: 'background 150ms',
  },
  demoBadge: {
    background: 'rgba(255,255,255,0.15)',
    color: '#fff',
    fontSize: '10px',
    fontWeight: '600',
    padding: '2px 7px',
    borderRadius: '4px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  right: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--bg-base)',
    padding: '32px',
  },
  card: {
    background: 'var(--bg-surface)',
    borderRadius: 'var(--radius-xl)',
    border: '1px solid var(--border)',
    padding: '40px',
    width: '100%',
    maxWidth: '400px',
    boxShadow: 'var(--shadow-lg)',
  },
  cardTitle: {
    fontSize: '24px',
    fontWeight: '700',
    color: 'var(--text-primary)',
    marginBottom: '6px',
    letterSpacing: '-0.3px',
  },
  cardSub: {
    fontSize: '14px',
    color: 'var(--text-secondary)',
    marginBottom: '32px',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  field: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  label: {
    fontSize: '13px',
    fontWeight: '500',
    color: 'var(--text-primary)',
  },
  input: {
    padding: '10px 14px',
    border: '1px solid var(--border-strong)',
    borderRadius: 'var(--radius-md)',
    fontSize: '14px',
    color: 'var(--text-primary)',
    background: 'var(--bg-elevated)',
    outline: 'none',
    transition: 'border-color var(--transition)',
  },
  error: {
    background: 'var(--danger-bg)',
    border: '1px solid var(--danger-border)',
    borderRadius: 'var(--radius-md)',
    color: 'var(--danger)',
    fontSize: '13px',
    padding: '10px 14px',
  },
  btn: {
    background: 'var(--accent)',
    color: '#fff',
    border: 'none',
    borderRadius: 'var(--radius-md)',
    padding: '12px',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    marginTop: '4px',
    transition: 'opacity 150ms',
  },
  btnDisabled: {
    background: 'var(--border-strong)',
    color: 'var(--text-muted)',
    border: 'none',
    borderRadius: 'var(--radius-md)',
    padding: '12px',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'not-allowed',
    marginTop: '4px',
  },
  hint: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    marginTop: '20px',
    paddingTop: '20px',
    borderTop: '1px solid var(--border)',
  },
  hintDot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: 'var(--success)',
    flexShrink: 0,
  },
}
