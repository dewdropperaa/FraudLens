import { useEffect, useMemo, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

function toRiskLabel(ts) {
  const n = Number(ts || 0)
  if (n >= 75) return 'HIGH'
  if (n >= 60) return 'MEDIUM'
  return 'LOW'
}

export default function InvestigationClaimPage({ claimId, user, token, onBackToDashboard }) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [context, setContext] = useState(null)
  const [notes, setNotes] = useState('')
  const [actionState, setActionState] = useState({ loading: false, error: '', success: '' })
  const [numPages, setNumPages] = useState(0)
  const [selectedEvidence, setSelectedEvidence] = useState(null)
  const [hoveredEvidence, setHoveredEvidence] = useState(null)
  const [debouncedHeatmap, setDebouncedHeatmap] = useState([])
  const explanationPanelRef = useRef(null)
  const role = String(user?.role || '').toLowerCase()
  const isAdminDemoUser = String(user?.email || '').toLowerCase() === 'admin@gmail.com'
  const canInvestigate = role === 'admin' || role === 'investigator' || isAdminDemoUser
  const [retryTick, setRetryTick] = useState(0)
  const [is403, setIs403] = useState(false)

  function sanitizeText(value) {
    return String(value || '')
      .replace(/[<>"'`]/g, '')
      .split('')
      .filter((ch) => {
        const code = ch.charCodeAt(0)
        return code >= 32 && code !== 127
      })
      .join('')
      .trim()
  }

  useEffect(() => {
    if (!canInvestigate) {
      setLoading(false)
      setError('Investigator/Admin access is required to open human review context.')
      return () => {}
    }
    let cancelled = false
    async function loadContext() {
      setLoading(true)
      setError('')
      try {
        const res = await fetch(`/api/v2/claim/${claimId}/human-review-context`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        })
        if (!res.ok) {
          if (res.status === 403) {
            setIs403(true)
            throw new Error('Accès refusé — vérifiez vos permissions')
          }
          throw new Error(`Failed to load context (${res.status})`)
        }
        setIs403(false)
        const body = await res.json()
        if (!cancelled) setContext(body)
      } catch (e) {
        if (!cancelled) setError(e?.message || 'Unable to load investigation context')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    loadContext()
    return () => { cancelled = true }
  }, [claimId, token, canInvestigate, retryTick])

  const ts = Number(context?.ts || 0)
  const risk = useMemo(() => toRiskLabel(ts), [ts])
  const fallbackEvidence = useMemo(() => {
    if (!context) return []
    return Array.isArray(context.heatmap_fallback) ? context.heatmap_fallback.slice(0, 30) : []
  }, [context])

  useEffect(() => {
    const timer = setTimeout(() => {
      const rows = Array.isArray(context?.heatmap) ? context.heatmap : []
      const pageCounter = new Map()
      const limited = rows.filter((item) => {
        const page = Number(item?.page || 1)
        const count = pageCounter.get(page) || 0
        if (count >= 20) return false
        pageCounter.set(page, count + 1)
        return true
      })
      setDebouncedHeatmap(limited)
    }, 120)
    return () => clearTimeout(timer)
  }, [context?.heatmap])

  async function submitDecision(decision) {
    setActionState({ loading: true, error: '', success: '' })
    try {
      const reviewerId = user?.email || user?.full_name || 'investigator'
      const res = await fetch('/api/v2/claim/human-decision', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          claim_id: claimId,
          decision,
          reviewer_id: reviewerId,
          notes,
        }),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(body?.detail || `Decision update failed (${res.status})`)
      setActionState({
        loading: false,
        error: '',
        success: decision === 'APPROVED'
          ? `Finalized. tx=${body?.tx_hash || 'n/a'} cid=${body?.ipfs_cid || 'n/a'}`
          : 'Claim rejected and stored.',
      })
    } catch (e) {
      setActionState({ loading: false, error: e?.message || 'Failed to submit decision', success: '' })
    }
  }

  if (loading) return <div className="cg-card">Chargement du dossier investigateur...</div>
  if (is403) {
    return (
      <div className="cg-card">
        <div style={{ color: 'var(--danger)', marginBottom: 10 }}>Accès refusé — vérifiez vos permissions</div>
        <button className="cg-btn cg-btn-ghost" onClick={() => setRetryTick((v) => v + 1)}>Réessayer</button>
      </div>
    )
  }
  if (error) return <div className="cg-card" style={{ color: 'var(--danger)' }}>{error}</div>
  if (!context) return <div className="cg-card">No human review context found.</div>

  return (
    <div className="space-y-4">
      <div className="cg-page-header">
        <div>
          <div className="cg-page-sub">Dashboard → Claims → Investigateur {claimId}</div>
          <div className="cg-page-title">Claim Investigation #{claimId}</div>
          <div className="cg-page-sub">Human-in-the-loop review and trust-layer finalization</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className="cg-btn cg-btn-ghost"
            onClick={() => {
              if (typeof window !== 'undefined') window.location.href = `/proof/${claimId}`
            }}
          >
            Proof Mode
          </button>
          <button className="cg-btn cg-btn-ghost" onClick={onBackToDashboard}>Back</button>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="cg-card">
          <div className="cg-card-title">Document Viewer</div>
          {context.document_url ? (
            <div className="h-[70vh] overflow-auto rounded border border-[var(--border)] p-2">
              <Document
                file={context.document_url}
                onLoadSuccess={({ numPages: pages }) => setNumPages(pages)}
                loading="Loading PDF..."
                error="Unable to load PDF document."
              >
                {Array.from({ length: numPages }, (_, index) => {
                  const pageNumber = index + 1
                  const pageHeatmap = debouncedHeatmap.filter((row) => Number(row.page) === pageNumber)
                  return (
                    <div key={pageNumber} style={{ position: 'relative', marginBottom: 16 }}>
                      <Page pageNumber={pageNumber} width={760} renderAnnotationLayer={false} renderTextLayer={false} />
                      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
                        {pageHeatmap.map((item, idx) => {
                          const bbox = Array.isArray(item?.bbox) ? item.bbox : [0, 0, 0, 0]
                          const [x0, y0, x1, y1] = bbox.map((v) => Number(v || 0))
                          const pageWidth = Number(item?.page_width || 1)
                          const pageHeight = Number(item?.page_height || 1)
                          const left = (x0 / pageWidth) * 100
                          const top = (y0 / pageHeight) * 100
                          const width = ((x1 - x0) / pageWidth) * 100
                          const height = ((y1 - y0) / pageHeight) * 100
                          const severity = String(item?.severity || 'LOW').toUpperCase()
                          const colors = severity === 'HIGH'
                            ? { border: '#dc2626', bg: 'rgba(255, 0, 0, 0.30)' }
                            : severity === 'MEDIUM'
                              ? { border: '#f97316', bg: 'rgba(249, 115, 22, 0.30)' }
                              : { border: '#eab308', bg: 'rgba(234, 179, 8, 0.30)' }
                          return (
                            <button
                              key={`${pageNumber}-${idx}`}
                              type="button"
                              style={{
                                position: 'absolute',
                                left: `${Math.max(0, left)}%`,
                                top: `${Math.max(0, top)}%`,
                                width: `${Math.max(0.8, width)}%`,
                                height: `${Math.max(1, height)}%`,
                                border: `2px solid ${colors.border}`,
                                background: colors.bg,
                                pointerEvents: 'auto',
                                cursor: 'pointer',
                              }}
                              onMouseEnter={() => setHoveredEvidence({
                                page: pageNumber,
                                reason: sanitizeText(item?.reason),
                                severity,
                                agent: sanitizeText(item?.agent),
                              })}
                              onMouseLeave={() => setHoveredEvidence(null)}
                              onClick={() => {
                                setSelectedEvidence(item)
                                explanationPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                              }}
                              title={`${sanitizeText(item?.reason)} detected by ${sanitizeText(item?.agent)}`}
                            />
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </Document>
              {hoveredEvidence && (
                <div className="mt-2 rounded border border-[var(--border)] bg-white p-2 text-xs shadow">
                  {hoveredEvidence.reason} detected by {hoveredEvidence.agent} ({hoveredEvidence.severity})
                </div>
              )}
              {debouncedHeatmap.length === 0 && fallbackEvidence.length > 0 && (
                <div className="mt-3 rounded border border-[var(--border)] p-2">
                  <div className="cg-card-title" style={{ fontSize: 14 }}>Fallback Suspicious Evidence</div>
                  <div className="space-y-2 text-sm">
                    {fallbackEvidence.map((item, idx) => (
                      <div key={`fallback-${idx}`} className="rounded border border-[var(--border)] p-2">
                        <b>{sanitizeText(item?.text) || 'Suspicious text'}</b>
                        <div>{sanitizeText(item?.reason)} - {sanitizeText(item?.agent)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="cg-page-sub">No temporary document URL available.</div>
          )}
        </div>
        <div className="space-y-3">
          <div className="cg-card">
            <div className="cg-card-title">AI Suggested Decision</div>
            <div className="cg-page-sub">{context.ai_suggested_decision || 'HUMAN_REVIEW'}</div>
            <div style={{ marginTop: 8 }}><b>Reason:</b> {context.reason || 'Uncertain / requires human validation'}</div>
            <div><b>Ts:</b> {context.ts}</div>
            <div><b>Risk:</b> {risk}</div>
          </div>
          <div className="cg-card">
            <div className="cg-card-title">Extracted Fields</div>
            <div><b>CIN:</b> {context?.extracted_data?.cin || 'N/A'}</div>
            <div><b>Amount:</b> {context?.extracted_data?.amount || 'N/A'}</div>
            <div><b>Provider:</b> {context?.extracted_data?.provider || 'N/A'}</div>
          </div>
          <div className="cg-card" ref={explanationPanelRef}>
            <div className="cg-card-title">Agent Breakdown</div>
            {selectedEvidence && (
              <div className="mb-2 rounded border border-[var(--border)] bg-gray-50 p-2 text-sm">
                <b>Selected Highlight:</b> {sanitizeText(selectedEvidence?.reason)} detected by {sanitizeText(selectedEvidence?.agent)}
              </div>
            )}
            <div className="space-y-2">
              {(context.agent_breakdown || []).map((row) => (
                <div key={row.agent} className="rounded border border-[var(--border)] p-2 text-sm">
                  <b>{row.agent}</b> score={row.score} confidence={row.confidence}
                  <div className="mt-1">{sanitizeText(row.explanation)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="cg-card">
        <div className="cg-card-title">Final Human Decision</div>
        <textarea
          className="w-full rounded border border-[var(--border)] p-2"
          placeholder="Optional reviewer notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
        <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
          <button className="cg-btn cg-btn-primary" disabled={actionState.loading} onClick={() => submitDecision('APPROVED')}>APPROVE</button>
          <button className="cg-btn" style={{ background: '#dc2626', color: '#fff' }} disabled={actionState.loading} onClick={() => submitDecision('REJECTED')}>REJECT</button>
        </div>
        {actionState.error && <div style={{ color: 'var(--danger)', marginTop: 8 }}>{actionState.error}</div>}
        {actionState.success && <div style={{ color: 'var(--success)', marginTop: 8 }}>{actionState.success}</div>}
      </div>
    </div>
  )
}
