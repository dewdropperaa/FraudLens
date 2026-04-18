import { useCallback, useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { ethers } from 'ethers'
import { Icons, SidebarItem } from './components'
import Dashboard  from './pages/Dashboard'
import SubmitClaim from './pages/SubmitClaim'
import Analytics  from './pages/Analytics'
import Database   from './pages/Database'

/* ── API ─────────────────────────────────────────────────────── */
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: Number(import.meta.env.VITE_API_TIMEOUT_MS) || 120000,
})
const apiKey = import.meta.env.VITE_CLAIMAGUARD_API_KEY
if (apiKey) api.defaults.headers.common['X-API-Key'] = apiKey

/* ── Helpers ─────────────────────────────────────────────────── */
function formatApiError(error) {
  const d = error?.response?.data?.detail
  if (d == null) return String(error?.message || 'Request failed.')
  if (typeof d === 'string') return d
  if (Array.isArray(d)) return d.map((e) => (typeof e === 'string' ? e : e?.msg ?? JSON.stringify(e))).filter(Boolean).join(' ') || 'Request failed.'
  if (typeof d === 'object') { try { return d.msg ? String(d.msg) : JSON.stringify(d) } catch { return 'Request failed.' } }
  return String(d)
}
export function safeText(v) {
  if (v == null) return ''
  if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return String(v)
  try { return JSON.stringify(v) } catch { return '[unprintable]' }
}
export function toIpfsUrl(h) { return h ? `https://gateway.pinata.cloud/ipfs/${h}` : null }
export function shortHex(v) {
  if (!v) return 'N/A'
  if (v.length <= 16) return v
  return `${v.slice(0, 10)}…${v.slice(-6)}`
}
export function scorePercent(score) {
  if (score == null) return 0
  const n = typeof score === 'string' ? parseFloat(score) : score
  if (Number.isNaN(n)) return 0
  if (n <= 1) return Math.round(n * 100)
  if (n <= 100) return Math.round(n)
  return 100
}

function aggregateClaimsByDay(claims, maxDaysBack = 90) {
  const cutoff = Date.now() - maxDaysBack * 86400000
  const map = new Map()
  for (const c of claims) {
    const ts = c?.timestamp; if (ts == null) continue
    const d = new Date(ts); if (Number.isNaN(d.getTime()) || d.getTime() < cutoff) continue
    const day = d.toISOString().slice(0, 10)
    if (!map.has(day)) map.set(day, { date: day, label: day.slice(5), valid: 0, fraud: 0 })
    const row = map.get(day)
    if (c.decision === 'APPROVED') row.valid += 1
    else if (c.decision === 'REJECTED') row.fraud += 1
  }
  return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date))
}

async function fetchAllClaimsAllPages() {
  const items = []; let page = 1; const page_size = 100
  for (;;) {
    const { data } = await api.get('/claims', { params: { filter: 'all', page, page_size } })
    const batch = data?.items ?? []; items.push(...batch)
    if (batch.length < page_size) break
    page += 1; if (page > 200) break
  }
  return items
}

/* ── Nav config ──────────────────────────────────────────────── */
const NAV_LINKS = [
  { id: 'dashboard', label: 'Home',       Icon: Icons.Home      },
  { id: 'submit',    label: 'New Claim',   Icon: Icons.FileText  },
  { id: 'analytics', label: 'Analytics',   Icon: Icons.BarChart2 },
  { id: 'database',  label: 'Database',    Icon: Icons.Database  },
]

/* ── App ─────────────────────────────────────────────────────── */
export default function App() {
  /* Form */
  const [form,          setForm]          = useState({ patient_id: '', provider_id: '', amount: '', insurance: 'CNSS' })
  const [selectedFiles, setSelectedFiles] = useState([])
  const [isSubmitting,  setIsSubmitting]  = useState(false)
  const [currentClaimId, setCurrentClaimId] = useState(null)
  const [submitError,   setSubmitError]   = useState('')
  const [lastResult,    setLastResult]    = useState(null)

  /* Claims */
  const [claims,        setClaims]        = useState([])
  const [claimsLoading, setClaimsLoading] = useState(false)
  const [claimsError,   setClaimsError]   = useState('')
  const [filter,        setFilter]        = useState('all')

  /* Chart */
  const [chartTick,    setChartTick]    = useState(0)
  const [dailyClaims,  setDailyClaims]  = useState([])
  const [chartLoading, setChartLoading] = useState(false)

  /* Page routing */
  const [activePage,   setActivePage]   = useState('dashboard')

  /* ── Stats ─────────────────────────────────────────────────── */
  const stats = useMemo(() => {
    const total    = claims.length
    const approved = claims.filter(c => c.decision === 'APPROVED').length
    const rejected = claims.filter(c => c.decision === 'REJECTED').length
    return { total, approved, rejected, fraudRate: total > 0 ? Math.round((rejected / total) * 100) : 0 }
  }, [claims])

  const hasValidTxHash = useMemo(() => lastResult?.tx_hash ? ethers.isHexString(lastResult.tx_hash) : false, [lastResult])

  /* ── Fetch claims ───────────────────────────────────────────── */
  const fetchClaims = useCallback(async (activeFilter = filter) => {
    setClaimsLoading(true); setClaimsError('')
    try {
      const res = await api.get('/claims', { params: { filter: activeFilter } })
      setClaims(res.data?.items ?? [])
    } catch (err) {
      setClaimsError(formatApiError(err) || 'Failed to fetch claims.')
    } finally { setClaimsLoading(false) }
  }, [filter])

  useEffect(() => { fetchClaims(filter) }, [filter, fetchClaims])

  useEffect(() => {
    let cancelled = false
    async function loadChart() {
      setChartLoading(true)
      try { const all = await fetchAllClaimsAllPages(); if (!cancelled) setDailyClaims(aggregateClaimsByDay(all)) }
      catch { if (!cancelled) setDailyClaims([]) }
      finally { if (!cancelled) setChartLoading(false) }
    }
    loadChart()
    return () => { cancelled = true }
  }, [chartTick])

  /* ── Handlers ───────────────────────────────────────────────── */
  function handleInputChange(e) { const { name, value } = e.target; setForm(p => ({ ...p, [name]: value })) }
  function handleFileChange(e)  { setSelectedFiles(Array.from(e.target.files ?? [])) }

  async function handleSubmit(e) {
    e.preventDefault(); setSubmitError(''); setIsSubmitting(true); setLastResult(null)
    const claimId = `claim-${Date.now()}`
    setCurrentClaimId(claimId)
    try {
      const fd = new FormData()
      fd.append('patient_id', form.patient_id.trim()); fd.append('provider_id', form.provider_id.trim())
      fd.append('amount', String(Number(form.amount))); fd.append('insurance', form.insurance)
      fd.append('claim_id', claimId)
      fd.append('history_json', JSON.stringify([]))
      for (const file of selectedFiles) fd.append('files', file)
      const res = await api.post('/claim/upload', fd)
      const body = res.data
      setLastResult(body?.data != null ? body.data : body)
      await fetchClaims(filter); setChartTick(t => t + 1)
    } catch (err) { setSubmitError(formatApiError(err) || 'Claim submission failed.')
    } finally { setIsSubmitting(false) }
  }

  /* ── Shared props passed to all pages ───────────────────────── */
  const shared = { claims, claimsLoading, claimsError, filter, setFilter, fetchClaims, shortHex, toIpfsUrl, safeText }

  const submitProps = {
    form, handleInputChange, handleFileChange, handleSubmit,
    isSubmitting, submitError, selectedFiles,
    lastResult, hasValidTxHash, safeText, shortHex, toIpfsUrl, scorePercent, currentClaimId
  }

  /* ── Render page content ────────────────────────────────────── */
  function renderPage() {
    switch (activePage) {
      case 'dashboard':  return <Dashboard  {...shared} stats={stats} dailyClaims={dailyClaims} chartLoading={chartLoading} />
      case 'submit':     return <SubmitClaim {...submitProps} />
      case 'analytics':  return <Analytics  dailyClaims={dailyClaims} chartLoading={chartLoading} stats={stats} claimsLoading={claimsLoading} />
      case 'database':
      case 'admin':
      case 'claims':     return <Database   {...shared} />
      default:           return <Dashboard  {...shared} stats={stats} dailyClaims={dailyClaims} chartLoading={chartLoading} />
    }
  }

  return (
    <div className="cg-layout">

      {/* ── Navbar ─────────────────────────────────────────────── */}
      <nav className="cg-nav">
        <div className="cg-nav-brand">
          <div className="cg-nav-brand-icon"><Icons.Shield /></div>
          <span className="cg-nav-brand-name">ClaimGuard</span>
        </div>

        <div className="cg-nav-links">
          {NAV_LINKS.map(({ id, label, Icon }) => (
            <button key={id} className={`cg-nav-link${activePage === id ? ' active' : ''}`} onClick={() => setActivePage(id)}>
              <Icon />{label}
            </button>
          ))}
        </div>

        <div className="cg-nav-actions">
          <button className="cg-nav-icon-btn" aria-label="Notifications">
            <Icons.Bell />{lastResult && <span className="dot" />}
          </button>
          <div className="cg-avatar">A</div>
        </div>
      </nav>

      {/* ── Sidebar ────────────────────────────────────────────── */}
      <aside className="cg-sidebar">
        <div className="cg-sidebar-label">Overview</div>
        <SidebarItem icon={Icons.Home}       label="Dashboard"    active={activePage === 'dashboard'}  onClick={() => setActivePage('dashboard')} />
        <SidebarItem icon={Icons.PlusCircle} label="Submit Claim" active={activePage === 'submit'}     onClick={() => setActivePage('submit')} />

        <div className="cg-sidebar-label">Analysis</div>
        <SidebarItem icon={Icons.BarChart2}  label="Analytics"    active={activePage === 'analytics'}  onClick={() => setActivePage('analytics')} />
        <SidebarItem icon={Icons.Cpu}        label="AI Agents"    active={activePage === 'agents'}     onClick={() => setActivePage('agents')} />

        <div className="cg-sidebar-label">Management</div>
        <SidebarItem icon={Icons.Database}   label="Admin View"   active={activePage === 'admin'}      onClick={() => setActivePage('database')} />
        <SidebarItem
          icon={Icons.Inbox}
          label="All Claims"
          badge={stats.total > 0 ? stats.total : undefined}
          active={activePage === 'claims' || activePage === 'database'}
          onClick={() => setActivePage('database')}
        />

        <div className="cg-sidebar-label">System</div>
        <SidebarItem icon={Icons.Box}  label="Blockchain" active={activePage === 'blockchain'} onClick={() => setActivePage('database')} />
        <SidebarItem icon={Icons.Link} label="IPFS Docs"  active={activePage === 'ipfs'}      onClick={() => setActivePage('database')} />
      </aside>

      {/* ── Page content ───────────────────────────────────────── */}
      <main className="cg-main">
        {renderPage()}
      </main>
    </div>
  )
}
