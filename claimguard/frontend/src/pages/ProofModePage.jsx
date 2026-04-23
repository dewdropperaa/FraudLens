import { useEffect, useMemo, useState } from 'react'

const STAGE_LABELS = {
  PRE_VALIDATION: 'PRE_VALIDATION',
  OCR_EXTRACTION: 'OCR',
  FIELD_VERIFICATION: 'FIELD_VERIFICATION',
  AGENTS: 'AGENTS',
  CONSENSUS: 'CONSENSUS',
  FINAL_DECISION: 'FINAL_DECISION',
  TRUST_LAYER: 'TRUST_LAYER',
  CRITICAL_STOP: 'CRITICAL_STOP',
}

function statusColor(status) {
  const s = String(status || '').toUpperCase()
  if (s === 'PASS') return '#16a34a'
  if (s === 'FAIL') return '#dc2626'
  return '#6b7280'
}

function maskCin(value) {
  const raw = String(value || '').trim()
  if (!raw) return raw
  if (raw.length <= 4) return `${raw[0] || ''}***`
  return `${raw.slice(0, 2)}***${raw.slice(-2)}`
}

function sanitizeSensitive(value) {
  if (Array.isArray(value)) return value.map(sanitizeSensitive)
  if (value && typeof value === 'object') {
    const out = {}
    for (const [k, v] of Object.entries(value)) {
      const key = String(k).toLowerCase()
      if (key.includes('cin')) out[k] = maskCin(v)
      else out[k] = sanitizeSensitive(v)
    }
    return out
  }
  return value
}

export default function ProofModePage({ claimId, token, onBack }) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [trace, setTrace] = useState(null)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [replayMode, setReplayMode] = useState(false)
  const [replayIndex, setReplayIndex] = useState(0)

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      setError('')
      try {
        const res = await fetch(`/api/v2/claim/${claimId}/proof`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        })
        const body = await res.json().catch(() => ({}))
        if (!res.ok) throw new Error(body?.detail || `Failed to load proof trace (${res.status})`)
        if (!cancelled) {
          setTrace(body)
          setSelectedIndex(0)
          setReplayIndex(0)
        }
      } catch (e) {
        if (!cancelled) setError(e?.message || 'Unable to load proof trace')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [claimId, token])

  useEffect(() => {
    if (!replayMode || !trace?.stages?.length) return undefined
    const timer = window.setInterval(() => {
      setReplayIndex((n) => (n + 1) % trace.stages.length)
    }, 900)
    return () => window.clearInterval(timer)
  }, [replayMode, trace])

  const stages = useMemo(() => Array.isArray(trace?.stages) ? trace.stages : [], [trace])
  const visibleIndex = replayMode ? replayIndex : selectedIndex
  const selected = stages[visibleIndex] || null
  const finalDecisionStage = [...stages].reverse().find((s) => s?.stage === 'FINAL_DECISION')
  const ts = Number(finalDecisionStage?.inputs?.Ts || 0)
  const reason = finalDecisionStage?.reason || 'No decision reason available'
  const triggerPoint = (stages.find((s) => String(s?.status).toUpperCase() === 'FAIL') || finalDecisionStage)?.stage || 'UNKNOWN'
  const blockedConditions = useMemo(() => {
    const allFlags = stages.flatMap((s) => Array.isArray(s?.flags) ? s.flags : [])
    return Array.from(new Set(allFlags.filter(Boolean)))
  }, [stages])
  const hasCriticalStop = stages.some((s) => s?.stage === 'CRITICAL_STOP')
  const isHumanReview = String(finalDecisionStage?.outputs?.final_decision || '').toUpperCase() === 'HUMAN_REVIEW'
  const isApproved = String(finalDecisionStage?.outputs?.final_decision || '').toUpperCase() === 'APPROVED'

  if (loading) return <div className="cg-card">Loading proof mode trace...</div>
  if (error) return <div className="cg-card" style={{ color: 'var(--danger)' }}>{error}</div>
  if (!trace) return <div className="cg-card">No proof trace available.</div>

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <div className="cg-card" style={{ borderLeft: `4px solid ${isApproved ? '#16a34a' : isHumanReview ? '#f59e0b' : '#dc2626'}` }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center' }}>
          <div>
            <div className="cg-card-title">Proof Mode — Claim {claimId}</div>
            <div className="cg-page-sub">Final decision: <b>{finalDecisionStage?.outputs?.final_decision || 'UNKNOWN'}</b> | Ts: <b>{ts || 'N/A'}</b></div>
            <div className="cg-page-sub">{reason}</div>
            <div className="cg-page-sub">Trigger point: <b>{triggerPoint}</b></div>
            {hasCriticalStop && <div style={{ marginTop: 8, color: '#dc2626', fontWeight: 700 }}>PIPELINE TERMINATED HERE</div>}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="cg-btn cg-btn-ghost" onClick={() => setReplayMode((v) => !v)}>{replayMode ? 'Stop Replay' : 'Replay Mode'}</button>
            <button className="cg-btn cg-btn-ghost" onClick={onBack}>Back</button>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
        <div className="cg-card">
          <div className="cg-card-title">Forensic Timeline</div>
          <div style={{ display: 'grid', gap: 10, marginTop: 10 }}>
            {stages.map((stage, idx) => {
              const selectedNode = idx === visibleIndex
              const nodeColor = statusColor(stage?.status)
              const pulse = stage?.stage === 'CRITICAL_STOP'
              return (
                <button
                  key={`${stage?.stage}-${idx}`}
                  onClick={() => { setSelectedIndex(idx); setReplayMode(false) }}
                  style={{
                    textAlign: 'left',
                    border: selectedNode ? `2px solid ${nodeColor}` : '1px solid var(--border)',
                    borderRadius: 8,
                    background: 'var(--card)',
                    padding: '10px 12px',
                    color: 'var(--text-primary)',
                    cursor: 'pointer',
                    boxShadow: pulse ? '0 0 0 2px rgba(220,38,38,0.2)' : 'none',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <b>{STAGE_LABELS[stage?.stage] || stage?.stage}</b>
                    <span style={{ color: nodeColor, fontWeight: 700 }}>{stage?.status}</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{stage?.timestamp}</div>
                </button>
              )
            })}
          </div>
          {!!blockedConditions.length && (
            <div style={{ marginTop: 14 }}>
              <div className="cg-card-title" style={{ fontSize: 14 }}>Why not approved?</div>
              <ul style={{ margin: '8px 0 0 16px' }}>
                {blockedConditions.map((flag) => <li key={flag}>{flag}</li>)}
              </ul>
            </div>
          )}
        </div>

        <div className="cg-card">
          <div className="cg-card-title">Stage Detail</div>
          {!selected ? (
            <div className="cg-page-sub">Select a stage to inspect forensic details.</div>
          ) : (
            <div style={{ display: 'grid', gap: 10 }}>
              <div><b>Stage:</b> {selected.stage}</div>
              <div><b>Status:</b> <span style={{ color: statusColor(selected.status) }}>{selected.status}</span></div>
              <div><b>Reasoning:</b> {selected.reason || 'N/A'}</div>
              <div><b>Flags:</b> {(selected.flags || []).join(', ') || 'None'}</div>
              <div>
                <b>Inputs used:</b>
                <pre style={{ whiteSpace: 'pre-wrap', marginTop: 6 }}>{JSON.stringify(sanitizeSensitive(selected.inputs || {}), null, 2)}</pre>
              </div>
              <div>
                <b>Outputs generated:</b>
                <pre style={{ whiteSpace: 'pre-wrap', marginTop: 6 }}>{JSON.stringify(sanitizeSensitive(selected.outputs || {}), null, 2)}</pre>
              </div>
              {!!selected.decision_snapshot && <div><b>Decision snapshot:</b> {selected.decision_snapshot}</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
