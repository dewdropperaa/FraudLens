import { useEffect, useMemo, useState } from 'react'
import {
  Area, AreaChart, CartesianGrid, Pie, PieChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Icons } from '../components'
import { ClaimFlowGraph } from '../components/ClaimFlowGraph'

const RISK_COLORS = { low: '#16a34a', medium: '#d97706', high: '#dc2626' }
const DECISION_COLORS = {
  approved: 'var(--success)',
  rejected: 'var(--danger)',
  human_review: '#d97706',
}

function toNumber(v) {
  if (typeof v === 'number') return v
  if (typeof v === 'string') {
    const n = Number(v.replace(/[^\d.-]/g, ''))
    return Number.isNaN(n) ? null : n
  }
  return null
}

function scorePercent(score) {
  const n = toNumber(score)
  if (n == null) return 0
  if (n <= 1) return Math.round(n * 100)
  if (n <= 100) return Math.round(n)
  return 100
}

function getAgentBadge(agent) {
  // UI-FIX
  const status = String(agent?.status || '').toUpperCase()
  const score = Number(agent?.score || 0)
  if (status === 'ERROR' || status === 'TIMEOUT') return 'FAIL'
  if (status === 'DONE' && score >= 60) return 'OK'
  if (status === 'DONE' && score >= 40) return 'WARN'
  return 'FAIL'
}

function inferDecision(claim) {
  return String(claim?.decision ?? claim?.final_decision ?? claim?.status ?? 'HUMAN_REVIEW').toUpperCase()
}

function inferStatus(claim) {
  return String(claim?.status ?? claim?.decision ?? 'HUMAN_REVIEW').toUpperCase()
}

function inferRisk(claim) {
  const score = scorePercent(claim?.score)
  if (score >= 70) return 'HIGH'
  if (score >= 40) return 'MEDIUM'
  return 'LOW'
}

function displayTs(claim) {
  const val = claim?.Ts ?? claim?.ts_score ?? claim?.trust_score
  const n = toNumber(val)
  return n == null ? null : Number(n.toFixed(3))
}

function extractedField(claim, key) {
  const extracted = claim?.extracted_fields || claim?.ocr_fields || claim?.fields || {}
  return extracted?.[key] ?? claim?.[key] ?? null
}

function normalizeClaim(raw) {
  const fieldVerification = raw?.field_verification
    || raw?.blackboard?.field_verification
    || raw?.reasoning_trace?.field_verification
    || []
  const systemFlags = raw?.system_flags || raw?.decision_trace?.system_flags || raw?.blackboard?.system_flags || []
  const memoryStatus = String(raw?.blackboard?.memory_status || '').toUpperCase()
  const criticalFailures = raw?.decision_trace?.critical_failures
    || raw?.blackboard?.critical_failures
    || []
  const hasCriticalFailures = Array.isArray(criticalFailures) && criticalFailures.length > 0
  return {
    ...raw,
    id: raw?.claim_id || raw?.id || raw?._id || `claim-${Math.random().toString(16).slice(2)}`,
    cin: extractedField(raw, 'cin') || extractedField(raw, 'patient_id') || raw?.patient_id || 'N/A',
    provider: extractedField(raw, 'provider') || extractedField(raw, 'provider_id') || raw?.provider_id || 'N/A',
    amount: toNumber(extractedField(raw, 'amount') ?? raw?.amount) ?? 0,
    ts: displayTs(raw),
    decision: inferDecision(raw),
    status: inferStatus(raw),
    risk: inferRisk(raw),
    scorePct: scorePercent(raw?.score),
    date: raw?.timestamp ? new Date(raw.timestamp) : null,
    contradictions: raw?.contradictions || raw?.reasoning_trace?.contradictions || [],
    hallucinations: raw?.hallucination_flags || raw?.reasoning_trace?.hallucination_flags || [],
    agentResults: raw?.agent_results || [],
    docHash: raw?.ipfs_hashes?.[0] || raw?.ipfs_hash || null,
    docHashes: raw?.ipfs_hashes || (raw?.ipfs_hash ? [raw.ipfs_hash] : []),
    extractedFields: raw?.extracted_fields || raw?.ocr_fields || raw?.fields || {},
    fieldVerification: Array.isArray(fieldVerification) ? fieldVerification : [],
    inputTrust: toNumber(raw?.input_trust_score ?? raw?.input_ts),
    reasoningSummary: raw?.reasoning_summary || raw?.reasoning || raw?.summary || '',
    preValidation: raw?.pre_validation_result || raw?.blackboard?.pre_validation || null,
    systemFlags: Array.isArray(systemFlags) ? systemFlags : [],
    memoryStatus,
    criticalFailures: Array.isArray(criticalFailures) ? criticalFailures : [],
  }
}

function Stat({ label, value, sub, color = '#111827' }) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-white p-4">
      <div className="text-xs text-[var(--text-muted)]">{label}</div>
      <div className="mt-1 text-2xl font-bold" style={{ color }}>{value}</div>
      {sub && <div className="mt-1 text-xs text-[var(--text-secondary)]">{sub}</div>}
    </div>
  )
}

export default function InvestigatorDashboard({ claims = [], claimsLoading, fetchClaims, user }) {
  const [query, setQuery] = useState('')
  const [riskFilter, setRiskFilter] = useState('ALL')
  const [decisionFilter, setDecisionFilter] = useState('ALL')
  const [dateFilter, setDateFilter] = useState('ALL')
  const [sortBy, setSortBy] = useState('ts_desc')
  const [selectedId, setSelectedId] = useState(null)
  const [actionReason, setActionReason] = useState('policy_violation')
  const [actionComment, setActionComment] = useState('')
  const [actionState, setActionState] = useState({ loading: false, error: '', success: '' })
  const [selectionStartedAt, setSelectionStartedAt] = useState(() => Date.now())

  const normalized = useMemo(() => claims.map(normalizeClaim), [claims])
  const selectedClaim = useMemo(() => normalized.find((c) => c.id === selectedId) || normalized[0] || null, [normalized, selectedId])

  useEffect(() => {
    if (!selectedId && normalized.length) setSelectedId(normalized[0].id)
  }, [normalized, selectedId])

  useEffect(() => {
    setSelectionStartedAt(Date.now())
  }, [selectedId])

  const filtered = useMemo(() => {
    const lower = query.trim().toLowerCase()
    const now = Date.now()
    const day = 24 * 60 * 60 * 1000
    const list = normalized.filter((c) => {
      const searchable = `${c.cin} ${c.provider}`.toLowerCase()
      const searchOk = !lower || searchable.includes(lower)
      const riskOk = riskFilter === 'ALL' || c.risk === riskFilter
      const decisionOk = decisionFilter === 'ALL' || c.decision === decisionFilter
      let dateOk = true
      if (dateFilter !== 'ALL' && c.date) {
        const age = now - c.date.getTime()
        dateOk = dateFilter === '7D' ? age <= 7 * day : dateFilter === '30D' ? age <= 30 * day : age <= 90 * day
      }
      return searchOk && riskOk && decisionOk && dateOk
    })
    list.sort((a, b) => {
      if (sortBy === 'ts_desc') return b.scorePct - a.scorePct
      if (sortBy === 'ts_asc') return a.scorePct - b.scorePct
      if (sortBy === 'amount_desc') return b.amount - a.amount
      if (sortBy === 'amount_asc') return a.amount - b.amount
      return 0
    })
    return list
  }, [normalized, query, riskFilter, decisionFilter, dateFilter, sortBy])

  const overview = useMemo(() => {
    const total = normalized.length
    const approved = normalized.filter((c) => c.decision === 'APPROVED').length
    const review = normalized.filter((c) => c.decision === 'HUMAN_REVIEW' || c.status === 'HUMAN_REVIEW').length
    const rejected = normalized.filter((c) => c.decision === 'REJECTED').length
    const tsValues = normalized.map((c) => c.ts).filter((v) => v != null)
    const avgTs = tsValues.length ? (tsValues.reduce((a, b) => a + b, 0) / tsValues.length).toFixed(3) : '0.000'
    return {
      total,
      approvedRate: total ? Math.round((approved / total) * 100) : 0,
      reviewRate: total ? Math.round((review / total) * 100) : 0,
      rejectedRate: total ? Math.round((rejected / total) * 100) : 0,
      avgTs,
    }
  }, [normalized])

  const claimsOverTime = useMemo(() => {
    const map = new Map()
    for (const c of normalized) {
      if (!c.date) continue
      const d = c.date.toISOString().slice(0, 10)
      map.set(d, (map.get(d) || 0) + 1)
    }
    return Array.from(map.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .slice(-30)
      .map(([date, count]) => ({ date: date.slice(5), count }))
  }, [normalized])

  const riskDistribution = useMemo(() => {
    const counts = { LOW: 0, MEDIUM: 0, HIGH: 0 }
    normalized.forEach((c) => { counts[c.risk] += 1 })
    return [
      { name: 'Low', value: counts.LOW, color: RISK_COLORS.low },
      { name: 'Medium', value: counts.MEDIUM, color: RISK_COLORS.medium },
      { name: 'High', value: counts.HIGH, color: RISK_COLORS.high },
    ]
  }, [normalized])

  const alerts = useMemo(() => {
    const out = []
    const highRiskCount = normalized.filter((c) => c.risk === 'HIGH').length
    if (normalized.length > 0 && highRiskCount / normalized.length > 0.25) out.push('Anomaly spike: high-risk claims exceed 25%.')

    const cinCounts = new Map()
    normalized.forEach((c) => cinCounts.set(c.cin, (cinCounts.get(c.cin) || 0) + 1))
    const repeatedCin = Array.from(cinCounts.entries()).filter(([, n]) => n >= 3).slice(0, 3)
    repeatedCin.forEach(([cin, n]) => out.push(`Repeated CIN alert: ${cin} appears in ${n} claims.`))

    const approved = normalized.filter((c) => c.decision === 'APPROVED').length
    if (normalized.length > 10 && approved / normalized.length > 0.9) out.push('Abnormal approval spike: approval rate above 90%.')
    return out
  }, [normalized])

  const graph = useMemo(() => {
    const cinNodes = new Set()
    const providerNodes = new Set()
    const edges = []
    normalized.slice(0, 24).forEach((c) => {
      const cin = `CIN:${c.cin}`
      const provider = `P:${c.provider}`
      const claim = `CL:${c.id.slice(0, 8)}`
      cinNodes.add(cin)
      providerNodes.add(provider)
      edges.push([cin, provider, c.risk === 'HIGH'])
      edges.push([cin, claim, c.risk === 'HIGH'])
    })
    return {
      nodes: [...cinNodes, ...providerNodes],
      edges,
      suspiciousClusters: edges.filter((e) => e[2]).length,
    }
  }, [normalized])

  async function sendDecision(decision) {
    if (!selectedClaim) return
    setActionState({ loading: true, error: '', success: '' })
    const token = localStorage.getItem('cg_token')
    const base = import.meta.env.VITE_API_BASE_URL || '/api'
    const headers = {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    }
    const payload = {
      decision,
      reason: actionReason,
      investigator_notes: actionComment,
      comment: actionComment,
      notes: actionComment,
      review_time_seconds: Math.max(0, Math.round((Date.now() - selectionStartedAt) / 1000)),
    }
    try {
      const reviewRes = await fetch(`${base}/claim/${selectedClaim.id}/review`, {
        method: 'PATCH',
        headers,
        body: JSON.stringify(payload),
      })
      if (!reviewRes.ok) throw new Error(`Review update failed (${reviewRes.status})`)

      await fetch(`${base}/claim/${selectedClaim.id}/feedback`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          claim_id: selectedClaim.id,
          decision,
          investigator_notes: actionComment,
          memory_event: 'human_feedback_capture',
        }),
      }).catch(() => null)

      setActionState({ loading: false, error: '', success: `Decision "${decision}" saved and feedback captured.` })
      setActionComment('')
      await fetchClaims()
    } catch (error) {
      setActionState({ loading: false, error: error?.message || 'Failed to save action', success: '' })
    }
  }

  const highRiskSelected = selectedClaim?.risk === 'HIGH'
  const preValidationFlags = Array.isArray(selectedClaim?.preValidation?.flags) ? selectedClaim.preValidation.flags : []
  const blockedForNonClaim = preValidationFlags.includes('NON_CLAIM')
  const blockedForInjection = preValidationFlags.includes('PROMPT_INJECTION')
  const memoryDegradedSelected = Boolean(
    selectedClaim?.systemFlags?.includes('MEMORY_DEGRADED')
      || (selectedClaim?.memoryStatus && selectedClaim.memoryStatus !== 'OK')
  )

  return (
    <div className="space-y-4">
      <div className="cg-page-header">
        <div>
          <div className="cg-page-title">Investigator Dashboard</div>
          <div className="cg-page-sub">Analyze, validate, and act on claims with full traceability</div>
        </div>
        <div className="rounded-lg border border-[var(--border)] bg-white px-3 py-2 text-xs text-[var(--text-secondary)]">
          Role: <span className="font-semibold text-[var(--text-primary)]">{user?.role || 'unknown'}</span>
        </div>
      </div>

      {highRiskSelected && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700">
          ⚠️ High Risk Detected
        </div>
      )}
      {blockedForNonClaim && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700">
          ❌ Not a valid medical claim
        </div>
      )}
      {blockedForInjection && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-700">
          ⚠️ Suspicious content detected (prompt injection)
        </div>
      )}
      {memoryDegradedSelected && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-700">
          ⚠️ Reduced fraud detection (memory unavailable)
        </div>
      )}
      {selectedClaim?.criticalFailures?.length > 0 && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700">
          ❌ Claim rejected: data not found in supporting document
        </div>
      )}

      <section className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <Stat label="Total Claims Processed" value={claimsLoading ? '—' : overview.total} />
        <Stat label="Approval / Review / Rejection" value={`${overview.approvedRate}% / ${overview.reviewRate}% / ${overview.rejectedRate}%`} />
        <Stat label="Average Ts Score" value={overview.avgTs} sub="Computed from available trust scores" />
        <Stat label="System Alerts" value={alerts.length} color={alerts.length > 0 ? '#dc2626' : '#16a34a'} />
      </section>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="rounded-lg border border-[var(--border)] bg-white p-4 xl:col-span-2">
          <div className="mb-2 text-sm font-semibold text-[var(--text-primary)]">Claims Over Time</div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={claimsOverTime}>
              <defs>
                <linearGradient id="claimsFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#1d4ed8" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#1d4ed8" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="date" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Area type="monotone" dataKey="count" stroke="#1d4ed8" fill="url(#claimsFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="rounded-lg border border-[var(--border)] bg-white p-4">
          <div className="mb-2 text-sm font-semibold text-[var(--text-primary)]">Fraud Risk Distribution</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={riskDistribution} dataKey="value" nameKey="name" innerRadius={45} outerRadius={80} paddingAngle={3}>
                {riskDistribution.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-lg border border-[var(--border)] bg-white p-4">
        <div className="mb-3 flex flex-wrap gap-2">
          <input
            className="min-w-56 rounded-md border border-[var(--border)] px-3 py-2 text-sm outline-none focus:border-blue-400"
            placeholder="Search by CIN or provider"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select className="rounded-md border border-[var(--border)] px-2 py-2 text-sm" value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}>
            <option value="ALL">Risk: All</option>
            <option value="LOW">Low</option>
            <option value="MEDIUM">Medium</option>
            <option value="HIGH">High</option>
          </select>
          <select className="rounded-md border border-[var(--border)] px-2 py-2 text-sm" value={decisionFilter} onChange={(e) => setDecisionFilter(e.target.value)}>
            <option value="ALL">Decision: All</option>
            <option value="APPROVED">Approved</option>
            <option value="HUMAN_REVIEW">Human Review</option>
            <option value="REJECTED">Rejected</option>
          </select>
          <select className="rounded-md border border-[var(--border)] px-2 py-2 text-sm" value={dateFilter} onChange={(e) => setDateFilter(e.target.value)}>
            <option value="ALL">Date: All</option>
            <option value="7D">Last 7 days</option>
            <option value="30D">Last 30 days</option>
            <option value="90D">Last 90 days</option>
          </select>
          <select className="rounded-md border border-[var(--border)] px-2 py-2 text-sm" value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="ts_desc">Sort: Ts desc</option>
            <option value="ts_asc">Sort: Ts asc</option>
            <option value="amount_desc">Amount desc</option>
            <option value="amount_asc">Amount asc</option>
          </select>
        </div>
        <div className="max-h-80 overflow-auto">
          <table className="cg-table">
            <thead>
              <tr>
                <th>Claim ID</th><th>CIN</th><th>Provider</th><th>Amount</th><th>Ts Score</th><th>Decision</th><th>Risk Level</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((c) => (
                <tr key={c.id} onClick={() => setSelectedId(c.id)} style={{ cursor: 'pointer', background: selectedClaim?.id === c.id ? '#f8fafc' : undefined }}>
                  <td className="mono">{c.id.slice(0, 12)}</td>
                  <td>{c.cin}</td>
                  <td>{c.provider}</td>
                  <td>{c.amount.toLocaleString()}</td>
                  <td className="score">{c.ts != null ? c.ts : c.scorePct / 100}</td>
                  <td style={{ color: DECISION_COLORS[c.decision.toLowerCase()] }}>{c.decision}</td>
                  <td style={{ color: RISK_COLORS[c.risk.toLowerCase()] }}>{c.risk}</td>
                  <td>{c.status}</td>
                </tr>
              ))}
              {filtered.length === 0 && <tr><td colSpan={8} style={{ padding: 20, textAlign: 'center' }}>No claims match filters.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      {selectedClaim && (
        <section className="grid grid-cols-1 gap-4 2xl:grid-cols-3">
          <div className="space-y-4 2xl:col-span-2">
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="rounded-lg border border-[var(--border)] bg-white p-4">
                <div className="mb-3 flex items-center gap-2 text-sm font-semibold"><Icons.FileText />Document Viewer</div>
                {selectedClaim.docHash ? (
                  <>
                    <iframe
                      title="Claim document"
                      src={`https://gateway.pinata.cloud/ipfs/${selectedClaim.docHash}`}
                      className="h-64 w-full rounded-md border border-[var(--border)]"
                    />
                    <div className="mt-3 space-y-2 text-sm">
                      <div className="rounded bg-amber-50 px-2 py-1"><span className="font-semibold">amount:</span> {selectedClaim.extractedFields?.amount ?? selectedClaim.amount}</div>
                      <div className="rounded bg-blue-50 px-2 py-1"><span className="font-semibold">CIN:</span> {selectedClaim.extractedFields?.cin ?? selectedClaim.cin}</div>
                      <div className="rounded bg-green-50 px-2 py-1"><span className="font-semibold">date:</span> {selectedClaim.extractedFields?.date ?? (selectedClaim.date ? selectedClaim.date.toISOString().slice(0, 10) : 'N/A')}</div>
                    </div>
                  </>
                ) : (
                  <div className="text-sm text-[var(--text-muted)]">No uploaded document available.</div>
                )}
              </div>

              <div className="rounded-lg border border-[var(--border)] bg-white p-4">
                <div className="mb-3 text-sm font-semibold">Analysis Panel</div>
                <div className="space-y-2 text-sm">
                  <div><span className="font-semibold">Final Decision:</span> {selectedClaim.decision}</div>
                  <div><span className="font-semibold">Score de confiance:</span> {selectedClaim.scorePct}/100</div>
                  <div><span className="font-semibold">Input Trust Score:</span> {selectedClaim.inputTrust ?? 'N/A'}</div>
                  <div className="mt-2 h-2 rounded bg-gray-100">
                    <div className="h-2 rounded" style={{ width: `${selectedClaim.scorePct}%`, background: '#22c55e' }} />
                  </div>
                  <div className="text-xs text-[var(--text-muted)]">Seuil d'approbation: 65</div>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-3 text-sm font-semibold">Agent Breakdown</div>
              <div className="space-y-2">
                {(selectedClaim.agentResults || []).map((ag, idx) => (
                  <div key={`${ag.agent_name || idx}-${idx}`} className="rounded-md border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-semibold">{ag.agent_name || `Agent ${idx + 1}`}</span>
                      <span className="text-xs">{getAgentBadge(ag)}</span>
                    </div>
                    <div className="mt-1 text-xs text-[var(--text-secondary)]">
                      score: {(Number(ag.score || 0) > 0 ? Math.round(Number(ag.score || 0)) : '—')} | confidence: {ag.confidence ?? 'N/A'}
                    </div>
                    <div className="mt-1 text-xs">{ag.explanation || ag.reasoning || 'Analyse complétée sans explication détaillée'}</div>
                    {ag.evidence_used && <div className="mt-1 text-xs text-[var(--text-secondary)]">evidence: {String(ag.evidence_used)}</div>}
                  </div>
                ))}
                {selectedClaim.agentResults?.length === 0 && <div className="text-sm text-[var(--text-muted)]">No agent breakdown available.</div>}
              </div>
            </div>

            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-3 text-sm font-semibold">Decision Trace</div>
              <div className="grid grid-cols-1 gap-2 text-sm md:grid-cols-2">
                <div className="rounded border border-[var(--border)] p-2"><span className="font-semibold">OCR Snapshot:</span> {selectedClaim.extractedFields?.ocr_snapshot || 'Available in extracted fields'}</div>
                <div className="rounded border border-[var(--border)] p-2"><span className="font-semibold">Extracted Fields:</span> {Object.keys(selectedClaim.extractedFields || {}).length || 0} fields</div>
                <div className="rounded border border-[var(--border)] p-2"><span className="font-semibold">Contradictions:</span> {selectedClaim.contradictions?.length || 0}</div>
                <div className="rounded border border-[var(--border)] p-2"><span className="font-semibold">Hallucination Flags:</span> {selectedClaim.hallucinations?.length || 0}</div>
              </div>
              <div className="mt-2 rounded border border-[var(--border)] bg-[var(--bg-elevated)] p-2 text-sm">
                <span className="font-semibold">Reasoning Summary:</span> {selectedClaim.reasoningSummary || 'No summary available'}
              </div>
              <div className="mt-3">
                <div className="mb-2 text-sm font-semibold">Field Verification</div>
                <div className="space-y-1 text-sm">
                  {selectedClaim.fieldVerification.length > 0 ? selectedClaim.fieldVerification.map((row, idx) => {
                    const isVerified = Boolean(row?.verified)
                    const label = String(row?.field || `field_${idx}`)
                    const value = row?.value ?? 'N/A'
                    const confidence = row?.match_confidence ?? 0
                    return (
                      <div
                        key={`${label}-${idx}`}
                        className={`rounded px-2 py-1 ${isVerified ? 'bg-green-50 text-green-800 border border-green-200' : 'bg-red-50 text-red-800 border border-red-200'}`}
                      >
                        <span className="font-semibold">{label}:</span> {String(value)} {isVerified ? '✅' : `❌ (not found, confidence ${confidence})`}
                      </div>
                    )
                  }) : (
                    <div className="text-[var(--text-muted)]">No field verification data available.</div>
                  )}
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-3 text-sm font-semibold">Fraud Relationships Graph</div>
              <svg viewBox="0 0 720 220" className="h-56 w-full rounded-md border border-[var(--border)] bg-slate-50">
                {graph.nodes.slice(0, 18).map((n, i) => {
                  const x = 40 + (i % 6) * 120
                  const y = 35 + Math.floor(i / 6) * 70
                  const isCin = n.startsWith('CIN:')
                  return (
                    <g key={n}>
                      <circle cx={x} cy={y} r="18" fill={isCin ? '#dbeafe' : '#ede9fe'} stroke={isCin ? '#1d4ed8' : '#7c3aed'} />
                      <text x={x} y={y + 4} textAnchor="middle" fontSize="9" fill="#1f2937">{n.split(':')[1]?.slice(0, 6) || 'N'}</text>
                    </g>
                  )
                })}
                {graph.edges.slice(0, 30).map(([from, to, risky], idx) => {
                  const a = graph.nodes.indexOf(from)
                  const b = graph.nodes.indexOf(to)
                  if (a < 0 || b < 0 || a > 17 || b > 17) return null
                  const x1 = 40 + (a % 6) * 120
                  const y1 = 35 + Math.floor(a / 6) * 70
                  const x2 = 40 + (b % 6) * 120
                  const y2 = 35 + Math.floor(b / 6) * 70
                  return <line key={`${idx}-${from}-${to}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={risky ? '#dc2626' : '#94a3b8'} strokeWidth={risky ? 2.5 : 1.4} />
                })}
              </svg>
              <div className="mt-2 text-xs text-[var(--text-secondary)]">
                Repeated entities and suspicious clusters highlighted in red ({graph.suspiciousClusters} risky links).
              </div>
            </div>

            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-3 text-sm font-semibold">Real-Time Flow Tracker</div>
              <ClaimFlowGraph claimId={selectedClaim.id} agentOutputs={selectedClaim.agentResults || []} />
            </div>
          </div>

          <aside className="space-y-4">
            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-3 text-sm font-semibold">Action Panel</div>
              <div className="space-y-2">
                <select className="w-full rounded-md border border-[var(--border)] px-2 py-2 text-sm" value={actionReason} onChange={(e) => setActionReason(e.target.value)}>
                  <option value="policy_violation">Policy violation</option>
                  <option value="missing_docs">Missing documents</option>
                  <option value="identity_mismatch">Identity mismatch</option>
                  <option value="manual_escalation">Manual escalation</option>
                </select>
                <textarea
                  className="h-24 w-full rounded-md border border-[var(--border)] px-2 py-2 text-sm"
                  placeholder="Add investigator notes..."
                  value={actionComment}
                  onChange={(e) => setActionComment(e.target.value)}
                />
                <button className="cg-btn cg-btn-primary cg-btn-full" disabled={actionState.loading} onClick={() => sendDecision('APPROVED')}>Approve</button>
                <button className="cg-btn cg-btn-full" style={{ background: '#dc2626', color: '#fff' }} disabled={actionState.loading} onClick={() => sendDecision('REJECTED')}>Reject</button>
                <button className="cg-btn cg-btn-ghost cg-btn-full" disabled={actionState.loading} onClick={() => sendDecision('HUMAN_REVIEW')}>Send to Manual Review</button>
                {actionState.error && <div className="text-xs text-red-600">{actionState.error}</div>}
                {actionState.success && <div className="text-xs text-green-700">{actionState.success}</div>}
              </div>
            </div>

            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-2 text-sm font-semibold">Alerts Panel</div>
              {alerts.length === 0 ? (
                <div className="text-sm text-[var(--text-muted)]">No active alerts.</div>
              ) : (
                <div className="space-y-2">
                  {alerts.map((a, i) => (
                    <div key={`${a}-${i}`} className="rounded-md border border-amber-200 bg-amber-50 px-2 py-2 text-xs text-amber-800">
                      {a}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>
        </section>
      )}
    </div>
  )
}
