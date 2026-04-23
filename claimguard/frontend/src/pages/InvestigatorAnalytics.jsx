import { useEffect, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const RISK_ORDER = ['LOW', 'MEDIUM', 'HIGH']
const DECISIONS = ['APPROVED', 'HUMAN_REVIEW', 'REJECTED']

function pct(v) {
  return `${Math.round((Number(v) || 0) * 100)}%`
}

function clamp01(v) {
  const n = Number(v) || 0
  return Math.max(0, Math.min(1, n))
}

function heatColor(value) {
  const t = clamp01(value)
  const red = Math.round(255 - (t * 70))
  const green = Math.round(90 + (t * 130))
  const blue = Math.round(90 + (t * 60))
  return `rgb(${red}, ${green}, ${blue})`
}

export default function InvestigatorAnalytics({ user }) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [data, setData] = useState(null)
  const [selectedId, setSelectedId] = useState('')

  useEffect(() => {
    let mounted = true
    async function load() {
      setLoading(true)
      setError('')
      try {
        const token = localStorage.getItem('cg_token')
        const base = import.meta.env.VITE_API_BASE_URL || '/api'
        const res = await fetch(`${base}/v2/investigator-analytics`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        })
        if (!res.ok) throw new Error(`Failed to load analytics (${res.status})`)
        const payload = await res.json()
        if (!mounted) return
        setData(payload)
        const first = payload?.leaderboard?.[0]?.investigator_id || ''
        setSelectedId((prev) => prev || first)
      } catch (e) {
        if (!mounted) return
        setError(e?.message || 'Failed to load investigator analytics')
      } finally {
        if (mounted) setLoading(false)
      }
    }
    load()
    return () => { mounted = false }
  }, [])

  const leaderboard = data?.leaderboard || []
  const selected = useMemo(
    () => leaderboard.find((row) => row.investigator_id === selectedId) || leaderboard[0] || null,
    [leaderboard, selectedId],
  )

  const agreementChart = useMemo(() => (
    leaderboard.map((row) => ({
      investigator: row.investigator_id.slice(0, 10),
      agreement: Math.round((row.agreement_rate || 0) * 100),
    }))
  ), [leaderboard])

  const trendChart = useMemo(() => {
    if (!selected) return []
    const recent = selected.profile?.recent_decisions || []
    const sorted = [...recent].sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)))
    return sorted.map((r, idx) => ({
      idx: idx + 1,
      agreement: String(r.investigator_decision || '').toUpperCase() === String(r.system_decision || '').toUpperCase() ? 100 : 0,
      reviewTime: Number(r.review_time_seconds || 0),
      ts: Number(r.system_Ts || 0),
    }))
  }, [selected])

  const pieData = useMemo(() => {
    if (!selected) return []
    const dist = selected.profile?.decision_distribution || {}
    return Object.entries(dist).map(([name, value]) => ({ name, value }))
  }, [selected])

  const heatmapRows = useMemo(() => {
    if (!selected) return []
    const counts = {}
    const recents = selected.profile?.recent_decisions || []
    for (const risk of RISK_ORDER) {
      counts[risk] = {}
      for (const dec of DECISIONS) counts[risk][dec] = 0
    }
    recents.forEach((r) => {
      const risk = String(r.system_risk_level || 'LOW').toUpperCase()
      const dec = String(r.investigator_decision || '').toUpperCase()
      if (!counts[risk]) counts[risk] = {}
      counts[risk][dec] = (counts[risk][dec] || 0) + 1
    })
    return RISK_ORDER.map((risk) => {
      const row = { risk }
      DECISIONS.forEach((d) => { row[d] = counts[risk]?.[d] || 0 })
      return row
    })
  }, [selected])

  if (user?.role !== 'admin') {
    return (
      <div className="rounded-lg border border-[var(--border)] bg-white p-6">
        <div className="text-lg font-semibold">Restricted Access</div>
        <div className="mt-2 text-sm text-[var(--text-muted)]">
          Investigator analytics are available only for admin role.
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="cg-page-header">
        <div>
          <div className="cg-page-title">Investigator Performance Analytics</div>
          <div className="cg-page-sub">Quality, consistency, risk-alignment, and review behavior insights</div>
        </div>
      </div>

      {loading && <div className="rounded-lg border border-[var(--border)] bg-white p-4 text-sm">Loading analytics...</div>}
      {error && <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</div>}

      {!loading && !error && (
        <>
          <section className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="text-xs text-[var(--text-muted)]">Total Human Reviews</div>
              <div className="mt-1 text-2xl font-bold">{data?.total_reviews || 0}</div>
            </div>
            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="text-xs text-[var(--text-muted)]">Investigators</div>
              <div className="mt-1 text-2xl font-bold">{leaderboard.length}</div>
            </div>
            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="text-xs text-[var(--text-muted)]">Active Alerts</div>
              <div className="mt-1 text-2xl font-bold text-red-600">{(data?.alerts || []).length}</div>
            </div>
            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="text-xs text-[var(--text-muted)]">Fairness Rule</div>
              <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                Patterns highlighted; disagreement is not auto-penalized
              </div>
            </div>
          </section>

          <section className="rounded-lg border border-[var(--border)] bg-white p-4">
            <div className="mb-2 text-sm font-semibold">Leaderboard</div>
            <div className="max-h-56 overflow-auto">
              <table className="cg-table">
                <thead>
                  <tr>
                    <th>Investigator</th>
                    <th>Total Reviews</th>
                    <th>Agreement Rate</th>
                    <th>High-Risk Detection</th>
                    <th>Stability</th>
                    <th>Alerts</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((row) => (
                    <tr
                      key={row.investigator_id}
                      onClick={() => setSelectedId(row.investigator_id)}
                      style={{ cursor: 'pointer', background: selected?.investigator_id === row.investigator_id ? '#f8fafc' : undefined }}
                    >
                      <td>{row.investigator_id}</td>
                      <td>{row.total_reviews}</td>
                      <td>{pct(row.agreement_rate)}</td>
                      <td>{pct(row.high_risk_detection_accuracy)}</td>
                      <td>{pct(row.stability_score)}</td>
                      <td>{row.alerts_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-2 text-sm font-semibold">Agreement Rate by Investigator</div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={agreementChart}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="investigator" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="agreement" fill="#2563eb" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-2 text-sm font-semibold">Decision Distribution</div>
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={45} outerRadius={85} paddingAngle={3}>
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={entry.name === 'APPROVED' ? '#16a34a' : entry.name === 'REJECTED' ? '#dc2626' : '#d97706'} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-2 text-sm font-semibold">Performance Over Time</div>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={trendChart}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="idx" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="agreement" stroke="#2563eb" strokeWidth={2} dot={false} name="Agreement %" />
                  <Line type="monotone" dataKey="reviewTime" stroke="#16a34a" strokeWidth={2} dot={false} name="Review time (s)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-2 text-sm font-semibold">Risk vs Decision Heatmap</div>
              <div className="grid grid-cols-5 gap-2 text-xs">
                <div />
                {DECISIONS.map((d) => <div key={d} className="text-center font-semibold">{d}</div>)}
                {heatmapRows.map((row) => (
                  <FragmentRow key={row.risk} row={row} />
                ))}
              </div>
            </div>
          </section>

          {selected && (
            <section className="rounded-lg border border-[var(--border)] bg-white p-4">
              <div className="mb-2 text-sm font-semibold">Investigator Profile: {selected.investigator_id}</div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
                <MiniStat label="Total Reviewed" value={selected.profile?.total_claims_reviewed || 0} />
                <MiniStat label="Approval/Rejection Ratio" value={selected.profile?.approval_rejection_ratio || 0} />
                <MiniStat label="Agreement with System" value={pct(selected.profile?.agreement_with_system || 0)} />
                <MiniStat label="Avg Review Time" value={`${selected.profile?.average_review_time || 0}s`} />
                <MiniStat label="Risk Sensitivity" value={pct(selected.metrics?.risk_sensitivity_score || 0)} />
              </div>
              <div className="mt-4 text-sm font-semibold">Recent Decisions</div>
              <div className="mt-2 max-h-48 overflow-auto">
                <table className="cg-table">
                  <thead>
                    <tr>
                      <th>Claim</th>
                      <th>System</th>
                      <th>Investigator</th>
                      <th>Risk</th>
                      <th>Ts</th>
                      <th>Time(s)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(selected.profile?.recent_decisions || []).map((r) => (
                      <tr key={`${r.claim_id}-${r.timestamp}`}>
                        <td className="mono">{String(r.claim_id).slice(0, 12)}</td>
                        <td>{r.system_decision}</td>
                        <td>{r.investigator_decision}</td>
                        <td>{r.system_risk_level}</td>
                        <td>{r.system_Ts}</td>
                        <td>{r.review_time_seconds}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {(selected.alerts || []).length > 0 && (
                <div className="mt-4 space-y-2">
                  {selected.alerts.map((a, idx) => (
                    <div key={`${a}-${idx}`} className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                      {a}
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}
        </>
      )}
    </div>
  )
}

function MiniStat({ label, value }) {
  return (
    <div className="rounded border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
      <div className="text-xs text-[var(--text-muted)]">{label}</div>
      <div className="mt-1 text-lg font-semibold">{value}</div>
    </div>
  )
}

function FragmentRow({ row }) {
  return (
    <>
      <div className="flex items-center justify-center rounded border border-[var(--border)] bg-[var(--bg-elevated)] py-2 font-semibold">{row.risk}</div>
      {DECISIONS.map((d) => (
        <div
          key={`${row.risk}-${d}`}
          className="flex items-center justify-center rounded border border-[var(--border)] py-2 font-semibold text-white"
          style={{ background: heatColor(Math.min(1, row[d] / 5)) }}
        >
          {row[d]}
        </div>
      ))}
    </>
  )
}
