import { useCallback, useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { ethers } from 'ethers'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 20000,
})

const apiKey = import.meta.env.VITE_CLAIMAGUARD_API_KEY
if (apiKey) {
  api.defaults.headers.common['X-API-Key'] = apiKey
}

const FILTERS = ['all', 'fraud', 'valid']

/** FastAPI may return detail as a string, or validation errors as an array of {msg, loc, ...}. */
function formatApiError(error) {
  const d = error?.response?.data?.detail
  if (d == null) {
    return String(error?.message || 'Request failed.')
  }
  if (typeof d === 'string') {
    return d
  }
  if (Array.isArray(d)) {
    const parts = d.map((e) => {
      if (typeof e === 'string') return e
      if (e && typeof e === 'object') {
        if (typeof e.msg === 'string') return e.msg
        if (e.msg != null) return JSON.stringify(e.msg)
        return JSON.stringify(e)
      }
      return String(e)
    })
    return parts.filter(Boolean).join(' ') || 'Request failed.'
  }
  if (typeof d === 'object') {
    if (typeof d.msg === 'string') return d.msg
    if (d.msg != null) return JSON.stringify(d.msg)
    try {
      return JSON.stringify(d)
    } catch {
      return 'Request failed.'
    }
  }
  return String(d)
}

/** Avoid React "Objects are not valid as a React child" if any field is unexpectedly nested. */
function safeText(value) {
  if (value == null) return ''
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  try {
    return JSON.stringify(value)
  } catch {
    return '[unprintable]'
  }
}

function toIpfsUrl(ipfsHash) {
  if (!ipfsHash) return null
  return `https://gateway.pinata.cloud/ipfs/${ipfsHash}`
}

function shortHex(value) {
  if (!value) return 'N/A'
  if (value.length <= 16) return value
  return `${value.slice(0, 10)}...${value.slice(-6)}`
}

/** Group claims by UTC calendar day; count APPROVED vs REJECTED (shown as valid vs fraud). */
function aggregateClaimsByDay(claims, maxDaysBack = 90) {
  const cutoff = Date.now() - maxDaysBack * 86400000
  const map = new Map()
  for (const c of claims) {
    const ts = c?.timestamp
    if (ts == null) continue
    const d = new Date(ts)
    if (Number.isNaN(d.getTime()) || d.getTime() < cutoff) continue
    const day = d.toISOString().slice(0, 10)
    if (!map.has(day)) {
      map.set(day, { date: day, label: day.slice(5), valid: 0, fraud: 0 })
    }
    const row = map.get(day)
    if (c.decision === 'APPROVED') row.valid += 1
    else if (c.decision === 'REJECTED') row.fraud += 1
  }
  return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date))
}

async function fetchAllClaimsAllPages() {
  const items = []
  let page = 1
  const page_size = 100
  for (;;) {
    const { data } = await api.get('/claims', {
      params: { filter: 'all', page, page_size },
    })
    const batch = data?.items ?? []
    items.push(...batch)
    if (batch.length < page_size) break
    page += 1
    if (page > 200) break
  }
  return items
}

function App() {
  const [form, setForm] = useState({
    patient_id: '',
    provider_id: '',
    amount: '',
    insurance: 'CNSS',
  })
  const [selectedFiles, setSelectedFiles] = useState([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState('')
  const [lastResult, setLastResult] = useState(null)

  const [claims, setClaims] = useState([])
  const [claimsLoading, setClaimsLoading] = useState(false)
  const [claimsError, setClaimsError] = useState('')
  const [filter, setFilter] = useState('all')

  const [chartTick, setChartTick] = useState(0)
  const [dailyClaims, setDailyClaims] = useState([])
  const [chartLoading, setChartLoading] = useState(false)

  const hasValidTxHash = useMemo(() => {
    if (!lastResult?.tx_hash) return false
    return ethers.isHexString(lastResult.tx_hash)
  }, [lastResult])

  const fetchClaims = useCallback(async (activeFilter = filter) => {
    setClaimsLoading(true)
    setClaimsError('')
    try {
      const response = await api.get('/claims', {
        params: { filter: activeFilter },
      })
      setClaims(response.data?.items ?? [])
    } catch (error) {
      setClaimsError(formatApiError(error) || 'Failed to fetch admin claims list.')
    } finally {
      setClaimsLoading(false)
    }
  }, [filter])

  useEffect(() => {
    fetchClaims(filter)
  }, [filter, fetchClaims])

  useEffect(() => {
    let cancelled = false
    async function loadChart() {
      setChartLoading(true)
      try {
        const all = await fetchAllClaimsAllPages()
        if (!cancelled) setDailyClaims(aggregateClaimsByDay(all))
      } catch {
        if (!cancelled) setDailyClaims([])
      } finally {
        if (!cancelled) setChartLoading(false)
      }
    }
    loadChart()
    return () => {
      cancelled = true
    }
  }, [chartTick])

  function handleInputChange(event) {
    const { name, value } = event.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  function handleFileChange(event) {
    const fileList = Array.from(event.target.files ?? [])
    setSelectedFiles(fileList)
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setSubmitError('')
    setIsSubmitting(true)

    try {
      const formData = new FormData()
      formData.append('patient_id', form.patient_id.trim())
      formData.append('provider_id', form.provider_id.trim())
      formData.append('amount', String(Number(form.amount)))
      formData.append('insurance', form.insurance)
      formData.append('history_json', JSON.stringify([]))
      for (const file of selectedFiles) {
        formData.append('files', file)
      }
      const response = await api.post('/claim/upload', formData)
      const body = response.data
      setLastResult(body?.data != null ? body.data : body)
      await fetchClaims(filter)
      setChartTick((t) => t + 1)
    } catch (error) {
      setSubmitError(formatApiError(error) || 'Claim submission failed.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-slate-100">
      <div className="mx-auto grid w-full max-w-7xl gap-6 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <h1 className="text-2xl font-semibold">ClaimGuard Dashboard</h1>
          <p className="mt-2 text-sm text-slate-300">
            Submit insurance claims and inspect AI + blockchain validation in real time.
          </p>

          <form onSubmit={handleSubmit} className="mt-5 space-y-4">
            <div>
              <label htmlFor="patient_id" className="mb-1 block text-sm">
                Patient ID
              </label>
              <input
                id="patient_id"
                name="patient_id"
                value={form.patient_id}
                onChange={handleInputChange}
                required
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 outline-none ring-cyan-500 focus:ring"
                placeholder="e.g. 12345678"
              />
            </div>

            <div>
              <label htmlFor="provider_id" className="mb-1 block text-sm">
                Provider / facility ID
              </label>
              <input
                id="provider_id"
                name="provider_id"
                value={form.provider_id}
                onChange={handleInputChange}
                required
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 outline-none ring-cyan-500 focus:ring"
                placeholder="e.g. CHU-Rabat-001"
              />
            </div>

            <div>
              <label htmlFor="amount" className="mb-1 block text-sm">
                Amount
              </label>
              <input
                id="amount"
                name="amount"
                type="number"
                min="0"
                step="0.01"
                value={form.amount}
                onChange={handleInputChange}
                required
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 outline-none ring-cyan-500 focus:ring"
                placeholder="5000"
              />
            </div>

            <div>
              <label htmlFor="insurance" className="mb-1 block text-sm">
                Insurance Type
              </label>
              <select
                id="insurance"
                name="insurance"
                value={form.insurance}
                onChange={handleInputChange}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 outline-none ring-cyan-500 focus:ring"
              >
                <option value="CNSS">CNSS</option>
                <option value="CNOPS">CNOPS</option>
              </select>
            </div>

            <div>
              <label htmlFor="documents" className="mb-1 block text-sm">
                Document Upload
              </label>
              <input
                id="documents"
                type="file"
                multiple
                onChange={handleFileChange}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 file:mr-4 file:rounded-md file:border-0 file:bg-cyan-600 file:px-3 file:py-1 file:text-white hover:file:bg-cyan-500"
              />
              <p className="mt-2 text-xs text-slate-400">
                Files upload via multipart; the API extracts text from PDFs and plain text, and runs
                Tesseract OCR on images when installed.
              </p>
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="inline-flex w-full items-center justify-center rounded-md bg-cyan-600 px-4 py-2 font-medium text-white transition hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isSubmitting ? 'Processing claim...' : 'Submit Claim'}
            </button>
          </form>

          {submitError && (
            <p className="mt-4 rounded-md border border-red-700 bg-red-950/60 px-3 py-2 text-sm text-red-300">
              {safeText(submitError)}
            </p>
          )}
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <h2 className="text-xl font-semibold">Results Dashboard</h2>
          {!lastResult ? (
            <p className="mt-4 text-sm text-slate-400">
              Submit a claim to see decision, score, blockchain hash, and IPFS references.
            </p>
          ) : (
            <div className="mt-4 space-y-4 text-sm">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
                  <p className="text-slate-400">Decision</p>
                  <p
                    className={`mt-1 text-lg font-semibold ${
                      lastResult.decision === 'APPROVED' ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {lastResult.decision}
                  </p>
                </div>
                <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
                  <p className="text-slate-400">Score</p>
                  <p className="mt-1 text-lg font-semibold text-cyan-300">{lastResult.score}</p>
                </div>
              </div>

              <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
                <p className="text-slate-400">Claim ID</p>
                <p className="mt-1 font-mono text-xs">{lastResult.claim_id}</p>
              </div>

              <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
                <p className="text-slate-400">Blockchain tx hash</p>
                {lastResult.tx_hash ? (
                  <a
                    href={hasValidTxHash ? `https://sepolia.etherscan.io/tx/${lastResult.tx_hash}` : '#'}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-1 inline-block font-mono text-xs text-cyan-300 hover:text-cyan-200"
                  >
                    {shortHex(lastResult.tx_hash)}
                  </a>
                ) : (
                  <p className="mt-1 text-slate-400">
                    {lastResult.decision === 'REJECTED'
                      ? 'Skipped — on-chain step runs only for approved claims.'
                      : 'No on-chain tx (set CONTRACT_ADDRESS + SEPOLIA_PRIVATE_KEY and restart the API).'}
                  </p>
                )}
              </div>

              <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
                <p className="text-slate-400">IPFS link</p>
                {lastResult.ipfs_hash ? (
                  <a
                    href={toIpfsUrl(lastResult.ipfs_hash)}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-1 inline-block font-mono text-xs text-cyan-300 hover:text-cyan-200"
                  >
                    {lastResult.ipfs_hash}
                  </a>
                ) : (
                  <p className="mt-1 text-slate-400">
                    {lastResult.decision === 'REJECTED'
                      ? 'Skipped — IPFS upload runs only for approved claims.'
                      : 'No IPFS hash returned (check Pinata / DOCUMENT_ENCRYPTION_KEY).'}
                  </p>
                )}
              </div>

              <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
                <p className="mb-2 text-slate-400">Agent breakdown</p>
                <div className="space-y-2">
                  {(lastResult.agent_results ?? []).map((agent) => (
                    <div
                      key={`${lastResult.claim_id}-${agent.agent_name}`}
                      className="rounded-md border border-slate-700 bg-slate-900 p-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium">{agent.agent_name}</span>
                        <span className={agent.decision ? 'text-emerald-300' : 'text-rose-300'}>
                          {agent.decision ? 'PASS' : 'FAIL'}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-slate-300">Score: {safeText(agent.score)}</p>
                      <p className="mt-1 text-xs text-slate-400">{safeText(agent.reasoning)}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 lg:col-span-2">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-semibold">Admin View</h2>
            <div className="flex items-center gap-2">
              {FILTERS.map((entry) => (
                <button
                  key={entry}
                  type="button"
                  onClick={() => setFilter(entry)}
                  className={`rounded-md px-3 py-1 text-sm ${
                    filter === entry
                      ? 'bg-cyan-600 text-white'
                      : 'bg-slate-800 text-slate-200 hover:bg-slate-700'
                  }`}
                >
                  {entry.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-6 rounded-lg border border-slate-800 bg-slate-950/50 p-4">
            <div className="mb-3">
              <h3 className="text-sm font-semibold text-slate-200">Daily claims: valid vs fraud</h3>
              <p className="text-xs text-slate-500">
                Count of approved (valid) vs rejected (fraud) by day (UTC), last 90 days. Uses all
                claims, not the table filter.
              </p>
            </div>
            {chartLoading ? (
              <p className="py-12 text-center text-sm text-slate-500">Loading chart…</p>
            ) : dailyClaims.length === 0 ? (
              <p className="py-12 text-center text-sm text-slate-500">No claim data in this window yet.</p>
            ) : (
              <div className="h-[320px] w-full min-w-0">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dailyClaims} margin={{ top: 8, right: 12, left: 4, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis allowDecimals={false} tick={{ fill: '#94a3b8', fontSize: 11 }} width={36} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                      }}
                      labelStyle={{ color: '#e2e8f0' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                    <Bar
                      dataKey="valid"
                      name="Valid (approved)"
                      fill="#34d399"
                      radius={[4, 4, 0, 0]}
                      maxBarSize={40}
                    />
                    <Bar
                      dataKey="fraud"
                      name="Fraud (rejected)"
                      fill="#f43f5e"
                      radius={[4, 4, 0, 0]}
                      maxBarSize={40}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {claimsLoading && <p className="mt-4 text-sm text-slate-300">Loading claims...</p>}
          {claimsError && (
            <p className="mt-4 rounded-md border border-red-700 bg-red-950/60 px-3 py-2 text-sm text-red-300">
              {safeText(claimsError)}
            </p>
          )}

          {!claimsLoading && !claimsError && (
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full border-collapse text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-700 text-slate-300">
                    <th className="px-3 py-2">Claim ID</th>
                    <th className="px-3 py-2">Decision</th>
                    <th className="px-3 py-2">Score</th>
                    <th className="px-3 py-2">Tx Hash</th>
                    <th className="px-3 py-2">IPFS</th>
                  </tr>
                </thead>
                <tbody>
                  {claims.map((claim) => (
                    <tr key={claim.claim_id} className="border-b border-slate-800">
                      <td className="px-3 py-2 font-mono text-xs">{shortHex(claim.claim_id)}</td>
                      <td
                        className={`px-3 py-2 font-semibold ${
                          claim.decision === 'APPROVED' ? 'text-emerald-300' : 'text-rose-300'
                        }`}
                      >
                        {claim.decision}
                      </td>
                      <td className="px-3 py-2 text-cyan-200">{claim.score}</td>
                      <td className="px-3 py-2 font-mono text-xs">{shortHex(claim.tx_hash)}</td>
                      <td className="px-3 py-2 font-mono text-xs">{shortHex(claim.ipfs_hash)}</td>
                    </tr>
                  ))}
                  {claims.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-3 py-4 text-center text-slate-400">
                        No claims match this filter yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </main>
  )
}

export default App
