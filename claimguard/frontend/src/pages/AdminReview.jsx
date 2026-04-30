import { useEffect, useRef, useState } from 'react'
import { Icons } from '../components'

const FILTERS = ['human_review', 'all', 'approved', 'rejected']

/* ── IPFS URL helper ────────────────────────────────────────── */
function resolveIpfsUrl(raw) {
  if (!raw) return null
  const s = String(raw).trim()
  if (s.startsWith('https://') || s.startsWith('http://')) return s
  const bare = s.replace(/^ipfs:\/\//, '')
  return bare ? `https://gateway.pinata.cloud/ipfs/${bare}` : null
}

/* ── Contradiction formatter ────────────────────────────────── */
const TECH_REPLACEMENTS = [
  [/\bIdentityAgent\b/gi, "identity verification"],
  [/\bAnomalyAgent\b/gi, "unusual activity patterns"],
  [/\bDocumentAgent\b/gi, "document verification"],
  [/\bPolicyAgent\b/gi, "policy compliance"],
  [/\bFraudAgent\b/gi, "fraud risk assessment"],
  [/\bMedianAgent\b/gi, "claim consistency check"],
  [/\bAgent\b/gi, "check"],
  [/\banomaly\b/gi, "unusual activity"],
  [/\bH_penalty\b/gi, "risk level"],
  [/\bpenalty\b/gi, "risk factor"],
  [/\bblackboard\b/gi, "claim data"],
  [/\bts_score\b/gi, "confidence level"],
  [/\bscore\b/gi, "assessment level"],
]

function cleanTechText(text) {
  let result = String(text || '')
  for (const [pattern, replacement] of TECH_REPLACEMENTS) {
    result = result.replace(pattern, replacement)
  }
  return result
}

function formatSingleContradiction(ct) {
  if (!ct) return null
  if (typeof ct === 'string') return cleanTechText(ct)

  const reason = cleanTechText(ct?.reason || '').trim()
  if (reason) {
    return `There is an inconsistency in this claim:\n${reason}\nThis may require further verification.`
  }
  return 'An inconsistency was detected in this claim that requires further verification.'
}

function formatContradictions(contradictions) {
  if (!contradictions?.length) return null
  if (contradictions.length === 1) return formatSingleContradiction(contradictions[0])
  const items = contradictions.map((ct, i) => `${i + 1}. ${formatSingleContradiction(ct)}`)
  return `Several inconsistencies were detected:\n${items.join('\n')}`
}

function StatusBadge({ decision }) {
  const color = decision === 'APPROVED' ? 'var(--success)' : decision === 'REJECTED' ? 'var(--danger)' : 'var(--text-muted)'
  const bg    = decision === 'APPROVED' ? 'var(--success-bg)' : decision === 'REJECTED' ? 'var(--danger-bg)' : 'var(--bg-elevated)'
  return (
    <span style={{ background: bg, color, border: `1px solid ${color}`, borderRadius: '5px', padding: '2px 8px', fontSize: '11px', fontWeight: 700, letterSpacing: '0.04em' }}>
      {decision ?? 'HUMAN_REVIEW'}
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
  const [filter, setFilter]         = useState('human_review')
  const [selected, setSelected]     = useState(null)
  const [actioning, setActioning]   = useState(false)
  const [actionError, setActionError]   = useState('')
  const [actionSuccess, setActionSuccess] = useState('')
  const [reviewContext, setReviewContext] = useState(null)
  const [contextLoading, setContextLoading] = useState(false)
  const [pdfViewed, setPdfViewed]   = useState(false)
  const [notes, setNotes]           = useState('')
  const [docBlobUrl, setDocBlobUrl] = useState(null)
  const [docLoading, setDocLoading] = useState(false)
  const blobUrlRef = useRef(null)

  const token = localStorage.getItem('cg_token') || ''
  const authHeader = token ? { Authorization: `Bearer ${token}` } : {}

  const [docError, setDocError] = useState('')

  // Fetch document blob when context changes so the iframe can load it
  // (iframes cannot send Authorization headers; blob URLs bypass this)
  useEffect(() => {
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current)
      blobUrlRef.current = null
    }
    setDocBlobUrl(null)
    setDocError('')

    const rawUrl = reviewContext?.document_url
    console.log('[AdminReview] document_url from context:', rawUrl)
    if (!rawUrl) return

    // External URLs (IPFS, etc.) work directly — no auth needed
    if (rawUrl.startsWith('http://') || rawUrl.startsWith('https://')) {
      setDocBlobUrl(rawUrl)
      return
    }

    // Local API paths need the JWT — fetch as blob then create object URL
    let cancelled = false
    setDocLoading(true)
    fetch(rawUrl, { headers: authHeader })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`)
        return r.blob()
      })
      .then(blob => {
        if (cancelled) return
        console.log('[AdminReview] Document blob loaded, size:', blob.size, 'type:', blob.type)
        const blobType = blob.type || 'application/pdf'
        const typedBlob = blob.type ? blob : new Blob([blob], { type: blobType })
        const url = URL.createObjectURL(typedBlob)
        blobUrlRef.current = url
        setDocBlobUrl(url)
      })
      .catch(err => {
        console.error('[AdminReview] Document fetch failed:', err)
        if (!cancelled) {
          setDocBlobUrl(null)
          setDocError(`Unable to load document: ${err.message}`)
        }
      })
      .finally(() => { if (!cancelled) setDocLoading(false) })

    return () => {
      cancelled = true
      setDocLoading(false)
    }
  }, [reviewContext?.document_url])

  // Revoke blob URL on unmount
  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current)
    }
  }, [])

  const visible = filter === 'approved'
    ? claims.filter(c => c.decision === 'APPROVED')
    : filter === 'rejected'
    ? claims.filter(c => c.decision === 'REJECTED')
    : filter === 'human_review'
    ? claims.filter(c => c.decision === 'HUMAN_REVIEW')
    : claims

  // Load human-review-context (PDF URL) when a HUMAN_REVIEW claim is selected
  useEffect(() => {
    if (!selected || selected.decision !== 'HUMAN_REVIEW') {
      setReviewContext(null)
      setDocBlobUrl(null)
      setDocError('')
      setPdfViewed(false)
      return
    }
    let cancelled = false
    setContextLoading(true)
    setReviewContext(null)
    setDocError('')
    setPdfViewed(false)
    console.log('[AdminReview] Fetching review context for claim:', selected.claim_id)
    fetch(`/api/v2/claim/${selected.claim_id}/human-review-context`, { headers: authHeader })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(data => {
        if (cancelled) return
        console.log('[AdminReview] Review context received:', { document_url: data?.document_url, claim_id: data?.claim_id })
        setReviewContext(data)
      })
      .catch(err => {
        console.error('[AdminReview] Review context fetch failed:', err)
        if (!cancelled) setDocError(`Review context unavailable: ${err.message}`)
      })
      .finally(() => { if (!cancelled) setContextLoading(false) })
    return () => { cancelled = true }
  }, [selected?.claim_id])

  async function submitReview(claimId, decision) {
    setActioning(true); setActionError(''); setActionSuccess('')
    try {
      const res = await fetch('/api/v2/claim/human-decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeader },
        body: JSON.stringify({ claim_id: claimId, decision, reviewer_id: 'admin', notes }),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(body?.detail || `HTTP ${res.status}`)

      if (decision === 'APPROVED') {
        const tx  = body.tx_hash  ? `tx: ${body.tx_hash.slice(0, 16)}…`  : ''
        const cid = body.ipfs_cid ? `IPFS: ${body.ipfs_cid.slice(0, 16)}…` : ''
        setActionSuccess(`Claim approved and anchored. ${[tx, cid].filter(Boolean).join(' · ')}`)
      } else {
        setActionSuccess('Claim rejected. No blockchain/IPFS anchoring performed.')
      }
      await fetchClaims()
      setSelected(prev => prev ? { ...prev, decision } : null)
      setPdfViewed(false)
    } catch (e) {
      setActionError(e.message || 'Action failed.')
    } finally {
      setActioning(false)
    }
  }

  const isHumanReview = selected?.decision === 'HUMAN_REVIEW'
  const canDecide = !isHumanReview || pdfViewed

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
              className={`cg-filter-pill${filter === f ? ` active-${f === 'human_review' ? 'all' : f}` : ''}`}>
              {f === 'human_review' ? 'Human Review' : (f.charAt(0).toUpperCase() + f.slice(1))}
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
                  onClick={() => { setSelected(claim); setActionError(''); setActionSuccess(''); setNotes('') }}
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
                  {(claim.contradictions?.length ?? 0) > 0 && (
                    <div style={{ marginTop: '4px', fontSize: '11px', color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: 4 }}>
                      <Icons.AlertTriangle style={{ width: 11, height: 11 }} />
                      {claim.contradictions.length} inconsistenc{claim.contradictions.length === 1 ? 'y' : 'ies'} detected
                    </div>
                  )}
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
            {/* Header with decision buttons */}
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
                    disabled={actioning || selected.decision === 'APPROVED' || !canDecide}
                    title={!canDecide ? 'Please review the document before deciding' : ''}
                    className="cg-btn cg-btn-sm"
                    style={{ background: 'var(--success)', color: '#fff', border: 'none', opacity: (selected.decision === 'APPROVED' || !canDecide) ? 0.45 : 1 }}
                  >
                    <Icons.CheckCircle style={{ width: 14, height: 14 }} /> Approve
                  </button>
                  <button
                    onClick={() => submitReview(selected.claim_id, 'REJECTED')}
                    disabled={actioning || selected.decision === 'REJECTED' || !canDecide}
                    title={!canDecide ? 'Please review the document before deciding' : ''}
                    className="cg-btn cg-btn-sm"
                    style={{ background: 'var(--danger)', color: '#fff', border: 'none', opacity: (selected.decision === 'REJECTED' || !canDecide) ? 0.45 : 1 }}
                  >
                    <Icons.XCircle style={{ width: 14, height: 14 }} /> Reject
                  </button>
                </div>
              </div>

              {/* Warn if doc not yet viewed */}
              {isHumanReview && !pdfViewed && (
                <div className="cg-alert" style={{ marginTop: '10px', background: '#fffbeb', border: '1px solid #fde68a', color: '#92400e', borderRadius: 8, padding: '8px 12px', fontSize: 12, display: 'flex', gap: 8, alignItems: 'center' }}>
                  <Icons.AlertTriangle style={{ width: 14, height: 14, flexShrink: 0 }} />
                  Review and confirm the document below before you can approve or reject.
                </div>
              )}
              {actionSuccess && <div className="cg-alert success" style={{ marginTop: '10px' }}><Icons.CheckCircle />{actionSuccess}</div>}
              {actionError   && <div className="cg-alert error"   style={{ marginTop: '10px' }}><Icons.AlertTriangle />{actionError}</div>}
            </div>

            {/* PDF viewer for HUMAN_REVIEW claims */}
            {isHumanReview && (
              <div className="cg-card">
                <div className="cg-card-header">
                  <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.FileText /></div>Supporting document</div>
                  {pdfViewed && (
                    <span style={{ fontSize: '11px', color: 'var(--success)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <Icons.CheckCircle style={{ width: 13, height: 13 }} /> Reviewed
                    </span>
                  )}
                </div>
                <div style={{ padding: '12px 16px' }}>
                  {(contextLoading || docLoading) && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 13 }}>
                      <span className="cg-spinner" style={{ width: 16, height: 16 }} />
                      Loading document...
                    </div>
                  )}
                  {!contextLoading && !docLoading && (() => {
                    const iframeUrl = docBlobUrl || resolveIpfsUrl(selected?.ipfs_hash)
                    if (docError && !iframeUrl) {
                      return (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, padding: '24px 0', color: 'var(--danger)' }}>
                          <Icons.AlertTriangle style={{ width: 32, height: 32, opacity: 0.6 }} />
                          <span style={{ fontSize: 13, textAlign: 'center', maxWidth: 400 }}>{docError}</span>
                          <button className="cg-btn cg-btn-ghost cg-btn-sm" onClick={() => setPdfViewed(true)}>
                            Continue without document
                          </button>
                        </div>
                      )
                    }
                    if (!iframeUrl) {
                      return (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, padding: '24px 0', color: 'var(--text-muted)' }}>
                          <Icons.FileText style={{ width: 32, height: 32, opacity: 0.35 }} />
                          <span style={{ fontSize: 13 }}>Document unavailable</span>
                          <button className="cg-btn cg-btn-ghost cg-btn-sm" onClick={() => setPdfViewed(true)}>
                            Continue without document
                          </button>
                        </div>
                      )
                    }
                    return (
                      <>
                        <iframe
                          key={iframeUrl}
                          src={iframeUrl}
                          title="Supporting document"
                          style={{ width: '100%', height: '600px', border: '1px solid var(--border)', borderRadius: '8px', display: 'block' }}
                        />
                        {!pdfViewed && (
                          <button
                            className="cg-btn cg-btn-primary"
                            style={{ marginTop: 12, width: '100%' }}
                            onClick={() => setPdfViewed(true)}
                          >
                            <Icons.CheckCircle style={{ width: 15, height: 15 }} /> I have reviewed and verified this document
                          </button>
                        )}
                        {pdfViewed && (
                          <div style={{ marginTop: 10, fontSize: 12, color: 'var(--success)', fontWeight: 600 }}>
                            Document verified — you can now approve or reject the claim.
                          </div>
                        )}
                      </>
                    )
                  })()}
                </div>
              </div>
            )}

            {/* Reviewer notes */}
            {isHumanReview && (
              <div className="cg-card">
                <div className="cg-card-header">
                  <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.FileText /></div>Reviewer notes</div>
                </div>
                <div style={{ padding: '12px 16px' }}>
                  <textarea
                    className="w-full rounded border border-[var(--border)] p-2"
                    style={{ width: '100%', minHeight: 80, padding: '8px', borderRadius: 6, border: '1px solid var(--border)', fontSize: 13, resize: 'vertical' }}
                    placeholder="Optional notes (rejection reason, observations...)"
                    value={notes}
                    onChange={e => setNotes(e.target.value)}
                  />
                </div>
              </div>
            )}

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

            {/* Detected inconsistencies */}
            {selected.contradictions?.length > 0 && (() => {
              const formatted = formatContradictions(selected.contradictions)
              if (!formatted) return null
              const lines = formatted.split('\n').filter(Boolean)
              return (
                <div className="cg-card">
                  <div className="cg-card-header">
                    <div className="cg-card-title">
                      <div className="cg-card-title-icon" style={{ background: 'var(--danger-bg)', color: 'var(--danger)' }}>
                        <Icons.AlertTriangle />
                      </div>
                      Detected inconsistencies
                    </div>
                  </div>
                  <div style={{ padding: '14px 16px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {lines.map((line, i) => {
                      const isNumbered = /^\d+\./.test(line)
                      const isHeader = i === 0 && lines.length > 2
                      return (
                        <div key={i} style={{
                          fontSize: isHeader ? 13 : 13,
                          fontWeight: isHeader ? 600 : 400,
                          color: isHeader ? 'var(--text-primary)' : 'var(--text-secondary)',
                          lineHeight: 1.6,
                          paddingLeft: isNumbered ? 4 : 0,
                        }}>
                          {line}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })()}
          </>
        )}
      </div>
    </div>
  )
}
