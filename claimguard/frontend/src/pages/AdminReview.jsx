import { useState } from 'react'
import { Icons } from '../components'

const FILTERS = ['all', 'pending', 'approved', 'rejected']

function StatusBadge({ decision }) {
  const color = decision === 'APPROVED' ? 'var(--success)' : decision === 'REJECTED' ? 'var(--danger)' : 'var(--text-muted)'
  const bg    = decision === 'APPROVED' ? 'var(--success-bg)' : decision === 'REJECTED' ? 'var(--danger-bg)' : 'var(--bg-elevated)'
  return (
    <span style={{ background: bg, color, border: `1px solid ${color}`, borderRadius: '5px', padding: '2px 8px', fontSize: '11px', fontWeight: 700, letterSpacing: '0.04em' }}>
      {decision ?? 'PENDING'}
    </span>
  )
}

function ScoreBar({ score }) {
  const pct = score == null ? 0 : score <= 1 ? Math.round(score * 100) : Math.min(Math.round(score), 100)
  const color = pct >= 70 ? 'var(--danger)' : pct >= 40 ? '#f59e0b' : 'var(--success)'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div style={{ flex: 1, height: '6px', background: 'var(--bg-elevated)', borderRadius: '3px', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: '3px', transition: 'width 0.3s' }} />
      </div>
      <span style={{ fontSize: '12px', fontWeight: 600, color, minWidth: '32px', textAlign: 'right' }}>{pct}%</span>
    </div>
  )
}

export default function AdminReview({ claims, claimsLoading, claimsError, fetchClaims, shortHex, safeText }) {
  const [filter, setFilter]     = useState('all')
  const [selected, setSelected] = useState(null)
  const [actioning, setActioning] = useState(false)
  const [actionError, setActionError] = useState('')
  const [actionSuccess, setActionSuccess] = useState('')

  const visible = filter === 'approved'
    ? claims.filter(c => c.decision === 'APPROVED')
    : filter === 'rejected'
    ? claims.filter(c => c.decision === 'REJECTED')
    : claims

  async function submitReview(claimId, decision) {
    setActioning(true); setActionError(''); setActionSuccess('')
    try {
      const res = await fetch(`/api/claim/${claimId}/review`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          ...(localStorage.getItem('cg_token') ? { Authorization: `Bearer ${localStorage.getItem('cg_token')}` } : {}),
        },
        body: JSON.stringify({ decision }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err?.detail || `HTTP ${res.status}`)
      }
      setActionSuccess(`Claim ${decision === 'APPROVED' ? 'approved' : 'rejected'} successfully.`)
      await fetchClaims()
      setSelected(prev => prev ? { ...prev, decision } : null)
    } catch (e) {
      setActionError(e.message || 'Action failed.')
    } finally {
      setActioning(false)
    }
  }

  return (
    <div style={{ display: 'flex', gap: '16px', height: 'calc(100vh - 100px)', minHeight: 0 }}>

      {/* ── Left: claim list ─────────────────────────────── */}
      <div style={{ width: '380px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div className="cg-page-header" style={{ marginBottom: 0 }}>
          <div>
            <div className="cg-page-title">Human Review</div>
            <div className="cg-page-sub">Review and adjudicate flagged claims</div>
          </div>
          <button className="cg-btn cg-btn-ghost cg-btn-sm" onClick={() => fetchClaims()}>
            <Icons.RefreshCw /> Refresh
          </button>
        </div>

        <div className="cg-filter-group" style={{ flexWrap: 'wrap' }}>
          {FILTERS.map(f => (
            <button key={f} type="button" onClick={() => setFilter(f)}
              className={`cg-filter-pill${filter === f ? ` active-${f === 'pending' ? 'all' : f}` : ''}`}>
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>

        <div className="cg-card" style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div className="cg-card-header">
            <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.Inbox /></div>Claims</div>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{visible.length} record{visible.length !== 1 ? 's' : ''}</span>
          </div>

          {claimsLoading && <div className="cg-empty"><span className="cg-spinner" style={{ width: 18, height: 18, color: '#6366f1' }} /></div>}
          {claimsError  && <div className="cg-alert error"><Icons.AlertTriangle />{safeText(claimsError)}</div>}

          {!claimsLoading && !claimsError && (
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {visible.length === 0 && (
                <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                  No claims match this filter.
                </div>
              )}
              {visible.map(claim => (
                <div
                  key={claim.claim_id}
                  onClick={() => { setSelected(claim); setActionError(''); setActionSuccess('') }}
                  style={{
                    padding: '12px 16px',
                    borderBottom: '1px solid var(--border)',
                    cursor: 'pointer',
                    background: selected?.claim_id === claim.claim_id ? 'var(--bg-elevated)' : 'transparent',
                    transition: 'background 120ms',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'monospace', color: 'var(--text-primary)' }}>
                      {shortHex(claim.claim_id)}
                    </span>
                    <StatusBadge decision={claim.decision} />
                  </div>
                  <ScoreBar score={claim.score} />
                  <div style={{ marginTop: '4px', fontSize: '11px', color: 'var(--text-muted)' }}>
                    {claim.agent_results?.length ?? 0} agents · {claim.contradictions?.length ?? 0} contradictions
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Right: detail / review panel ─────────────────── */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto' }}>
        {!selected ? (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '14px', flexDirection: 'column', gap: '12px' }}>
            <Icons.Inbox style={{ width: 40, height: 40, opacity: 0.3 }} />
            <span>Select a claim to review</span>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="cg-card" style={{ padding: '16px 20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '2px' }}>Claim ID</div>
                  <div style={{ fontFamily: 'monospace', fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>{selected.claim_id}</div>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                  <StatusBadge decision={selected.decision} />
                  <button
                    onClick={() => submitReview(selected.claim_id, 'APPROVED')}
                    disabled={actioning || selected.decision === 'APPROVED'}
                    className="cg-btn cg-btn-sm"
                    style={{ background: 'var(--success)', color: '#fff', border: 'none', opacity: selected.decision === 'APPROVED' ? 0.5 : 1 }}
                  >
                    <Icons.CheckCircle style={{ width: 14, height: 14 }} /> Approve
                  </button>
                  <button
                    onClick={() => submitReview(selected.claim_id, 'REJECTED')}
                    disabled={actioning || selected.decision === 'REJECTED'}
                    className="cg-btn cg-btn-sm"
                    style={{ background: 'var(--danger)', color: '#fff', border: 'none', opacity: selected.decision === 'REJECTED' ? 0.5 : 1 }}
                  >
                    <Icons.XCircle style={{ width: 14, height: 14 }} /> Reject
                  </button>
                </div>
              </div>
              {actionSuccess && <div className="cg-alert success" style={{ marginTop: '10px' }}><Icons.CheckCircle />{actionSuccess}</div>}
              {actionError   && <div className="cg-alert error"   style={{ marginTop: '10px' }}><Icons.AlertTriangle />{actionError}</div>}
            </div>

            {/* Score + trust breakdown */}
            <div className="cg-card">
              <div className="cg-card-header">
                <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.BarChart2 /></div>Fraud Score</div>
              </div>
              <div style={{ padding: '12px 16px 16px' }}>
                <ScoreBar score={selected.score} />
                {selected.Ts != null && (
                  <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                    Trust score (Ts): <strong>{typeof selected.Ts === 'number' ? selected.Ts.toFixed(3) : selected.Ts}</strong>
                  </div>
                )}
                {selected.mahic_breakdown && Object.keys(selected.mahic_breakdown).length > 0 && (
                  <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px,1fr))', gap: '8px' }}>
                    {Object.entries(selected.mahic_breakdown).map(([k, v]) => (
                      <div key={k} style={{ background: 'var(--bg-elevated)', borderRadius: '6px', padding: '8px 10px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{k}</div>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                          {typeof v === 'number' ? v.toFixed(2) : String(v)}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Agent results */}
            {selected.agent_results?.length > 0 && (
              <div className="cg-card">
                <div className="cg-card-header">
                  <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.Users /></div>Agent Results</div>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{selected.agent_results.length} agents</span>
                </div>
                <div style={{ padding: '0 0 8px' }}>
                  {selected.agent_results.map((ag, i) => (
                    <div key={i} style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>{ag.agent_name}</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontSize: '11px', color: ag.decision ? 'var(--danger)' : 'var(--success)', fontWeight: 600 }}>
                            {ag.decision ? 'FRAUD' : 'LEGIT'}
                          </span>
                          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>
                            {typeof ag.score === 'number' ? ag.score.toFixed(1) : ag.score}
                          </span>
                        </div>
                      </div>
                      <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{ag.reasoning}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Contradictions */}
            {selected.contradictions?.length > 0 && (
              <div className="cg-card">
                <div className="cg-card-header">
                  <div className="cg-card-title"><div className="cg-card-title-icon" style={{ background: 'var(--danger-bg)', color: 'var(--danger)' }}><Icons.AlertTriangle /></div>Contradictions</div>
                  <span style={{ fontSize: '11px', color: 'var(--danger)' }}>{selected.contradictions.length} found</span>
                </div>
                <div style={{ padding: '0 0 8px' }}>
                  {selected.contradictions.map((ct, i) => (
                    <div key={i} style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                      {typeof ct === 'string' ? ct : JSON.stringify(ct)}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
