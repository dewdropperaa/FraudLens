import { useCallback, useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { ethers } from 'ethers'
import { Icons, SidebarItem } from './components'
import Dashboard   from './pages/Dashboard'
import SubmitClaim  from './pages/SubmitClaim'
import Database    from './pages/Database'
import AdminReview from './pages/AdminReview'
import InvestigationClaimPage from './pages/InvestigationClaimPage'
import ProofModePage from './pages/ProofModePage'
import Login      from './pages/Login'
import { useAuth } from './context/AuthContext'

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

/* ── App ─────────────────────────────────────────────────────── */
export default function App() {
  const { user, token, logout, isAuthenticated } = useAuth()

  /* Inject JWT; remove X-API-Key so the JWT role is used on the backend */
  useEffect(() => {
    if (token) {
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`
      delete api.defaults.headers.common['X-API-Key']
    } else {
      delete api.defaults.headers.common['Authorization']
      if (apiKey) api.defaults.headers.common['X-API-Key'] = apiKey
    }
  }, [token])

  useEffect(() => {
    const interceptorId = api.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error?.response?.status === 401) {
          logout()
        }
        return Promise.reject(error)
      },
    )
    return () => {
      api.interceptors.response.eject(interceptorId)
    }
  }, [logout])

  /* Form */
  const [form,           setForm]          = useState({ patient_id: '', provider_id: '', amount: '', insurance: 'CNSS' })
  const [selectedFiles,  setSelectedFiles] = useState([])
  const [isSubmitting,   setIsSubmitting]  = useState(false)
  const [currentClaimId, setCurrentClaimId] = useState(null)
  const [submitError,    setSubmitError]   = useState('')
  const [lastResult,     setLastResult]    = useState(null)

  /* Claims */
  const [claims,        setClaims]        = useState([])
  const [claimsLoading, setClaimsLoading] = useState(false)
  const [claimsError,   setClaimsError]   = useState('')
  const [filter,        setFilter]        = useState('all')

  /* Chart (Dashboard only) */
  const [chartTick,    setChartTick]    = useState(0)
  const [dailyClaims,  setDailyClaims]  = useState([])
  const [chartLoading, setChartLoading] = useState(false)

  /* Page routing */
  const initialPath = typeof window !== 'undefined' ? window.location.pathname : '/'
  const investigationMatch = initialPath.match(/^\/investigation\/([^/]+)$/)
  const proofMatch = initialPath.match(/^\/proof\/([^/]+)$/)
  const [activePage, setActivePage] = useState(
    proofMatch
        ? 'proof-mode'
      : investigationMatch
        ? 'investigation-claim'
        : 'dashboard',
  )
  const [investigationClaimId, setInvestigationClaimId] = useState(investigationMatch?.[1] || null)
  const [proofClaimId, setProofClaimId] = useState(proofMatch?.[1] || null)

  /* ── Stats ─────────────────────────────────────────────────── */
  const stats = useMemo(() => {
    const total    = claims.length
    const approved = claims.filter(c => c.decision === 'APPROVED').length
    const rejected = claims.filter(c => c.decision === 'REJECTED').length
    return { total, approved, rejected, fraudRate: total > 0 ? Math.round((rejected / total) * 100) : 0 }
  }, [claims])

  const hasValidTxHash = useMemo(() => lastResult?.tx_hash ? ethers.isHexString(lastResult.tx_hash) : false, [lastResult])

  /* ── Fetch claims ───────────────────────────────────────────── */
  const fetchClaims = useCallback(async () => {
    if (!token) {
      setClaims([])
      setClaimsLoading(false)
      setClaimsError('')
      return
    }
    setClaimsLoading(true); setClaimsError('')
    try {
      const res = await api.get('/claims', { params: { filter: 'all', page_size: 100 } })
      setClaims(res.data?.items ?? [])
    } catch (err) {
      setClaimsError(formatApiError(err) || 'Failed to fetch claims.')
    } finally { setClaimsLoading(false) }
  }, [token])

  useEffect(() => { fetchClaims() }, [fetchClaims])

  useEffect(() => {
    if (!token) {
      setDailyClaims([])
      setChartLoading(false)
      return
    }
    let cancelled = false
    async function loadChart() {
      setChartLoading(true)
      try { const all = await fetchAllClaimsAllPages(); if (!cancelled) setDailyClaims(aggregateClaimsByDay(all)) }
      catch { if (!cancelled) setDailyClaims([]) }
      finally { if (!cancelled) setChartLoading(false) }
    }
    loadChart()
    return () => { cancelled = true }
  }, [chartTick, token])

  /* ── Handlers ───────────────────────────────────────────────── */
  function handleInputChange(e) { const { name, value } = e.target; setForm(p => ({ ...p, [name]: value })) }
  function handleFileChange(e)  { setSelectedFiles(Array.from(e.target.files ?? [])) }

  async function handleSubmit(e) {
    e.preventDefault(); setSubmitError(''); setIsSubmitting(true); setLastResult(null)
    const claimId = `claim-${Date.now()}`
    setCurrentClaimId(claimId)
    try {
      const documentsBase64 = await Promise.all(
        selectedFiles.map((file) => new Promise((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => {
            const result = String(reader.result || '')
            const base64 = result.includes(',') ? result.split(',')[1] : result
            resolve({ name: file.name, content_base64: base64 })
          }
          reader.onerror = () => reject(reader.error || new Error('Failed to read file'))
          reader.readAsDataURL(file)
        })),
      )
      const res = await api.post('/v2/claim/analyze', {
        identity: {
          cin: form.patient_id.trim(),
        },
        policy: {
          insurance: form.insurance,
        },
        metadata: {
          claim_id: claimId,
          provider: form.provider_id.trim(),
          amount: Number(form.amount),
        },
        documents: selectedFiles.map((file) => ({ id: file.name, document_type: file.type || 'uploaded_file' })),
        documents_base64: documentsBase64,
      })
      const body = res.data
      console.log('🚀 USING V2 PIPELINE')
      console.log('BACKEND RESPONSE:', body)
      console.log('FINAL DECISION FROM BACKEND:', body?.decision)
      setLastResult(body?.data != null ? body.data : body)
      const normalizedResult = body?.data != null ? body.data : body
      // Route to Admin View only for HUMAN_REVIEW — REJECTED stays on submit page
      if (normalizedResult?.claim_id && normalizedResult?.decision === 'HUMAN_REVIEW') {
        setPageAndPath('admin')
      }
      await fetchClaims(filter); setChartTick(t => t + 1)
    } catch (err) { setSubmitError(formatApiError(err) || 'Claim submission failed.')
    } finally { setIsSubmitting(false) }
  }

  /* ── Shared props ───────────────────────────────────────────── */
  const shared = { claims, claimsLoading, claimsError, filter, setFilter, fetchClaims, shortHex, toIpfsUrl, safeText }

  const submitProps = {
    form, handleInputChange, handleFileChange, handleSubmit,
    isSubmitting, submitError, selectedFiles,
    lastResult, hasValidTxHash, safeText, shortHex, toIpfsUrl, scorePercent, currentClaimId,
  }

  /* ── Render page ────────────────────────────────────────────── */
  const isAdminDemoUser = (user?.email || '').toLowerCase() === 'admin@gmail.com'
  const canInvestigate = user?.role === 'admin' || isAdminDemoUser
  const canSeeAnalytics = user?.role === 'admin' || isAdminDemoUser

  function setPageAndPath(nextPage) {
    setActivePage(nextPage)
    if (nextPage !== 'investigation-claim') {
      setInvestigationClaimId(null)
    }
    if (nextPage !== 'proof-mode') {
      setProofClaimId(null)
    }
    const path = nextPage === 'proof-mode' && proofClaimId
        ? `/proof/${proofClaimId}`
        : '/'
    if (typeof window !== 'undefined' && window.location.pathname !== path) {
      window.history.replaceState({}, '', path)
    }
  }

  function renderPage() {
    switch (activePage) {
      case 'investigation-claim':
        return canInvestigate && investigationClaimId
          ? (
            <InvestigationClaimPage
              claimId={investigationClaimId}
              user={user}
              token={token}
              onBackToDashboard={() => setPageAndPath('admin')}
            />
          )
          : (
            <div className="cg-card">
              <div className="cg-card-title">No investigation selected</div>
              <div className="cg-page-sub" style={{ marginTop: 8 }}>
                Submit a claim or pick one from the admin dashboard.
              </div>
            </div>
          )
      case 'proof-mode':
        return proofClaimId
          ? (
            <ProofModePage
              claimId={proofClaimId}
              token={token}
              onBack={() => setPageAndPath('admin')}
            />
          )
          : (
            <div className="cg-card">
              <div className="cg-card-title">No proof trace selected</div>
              <div className="cg-page-sub" style={{ marginTop: 8 }}>
                Open a proof URL like /proof/&lt;claim_id&gt; or launch from claim actions.
              </div>
            </div>
          )
      case 'dashboard': return <Dashboard  {...shared} stats={stats} dailyClaims={dailyClaims} chartLoading={chartLoading} />
      case 'submit':    return <SubmitClaim {...submitProps} />
      case 'admin':     return <AdminReview {...shared} />
      case 'database':
      case 'claims':    return <Database   {...shared} />
      default:          return <Dashboard  {...shared} stats={stats} dailyClaims={dailyClaims} chartLoading={chartLoading} />
    }
  }

  /* ── Auth gate ──────────────────────────────────────────────── */
  if (!isAuthenticated) return <Login />

  return (
    <div className="cg-layout">

      {/* ── Navbar ─────────────────────────────────────────────── */}
      <nav className="cg-nav">
        <div className="cg-nav-brand">
          <div className="cg-nav-brand-icon"><Icons.Shield /></div>
          <span className="cg-nav-brand-name">FraudLens</span>
        </div>

        <div className="cg-nav-links">
          {[
            { id: 'dashboard', label: 'Home',      Icon: Icons.Home     },
            { id: 'submit',    label: 'New Claim',  Icon: Icons.FileText },
            { id: 'database',  label: 'All Claims', Icon: Icons.Database },
          ].map(({ id, label, Icon }) => (
            <button key={id} className={`cg-nav-link${activePage === id ? ' active' : ''}`} onClick={() => setPageAndPath(id)}>
              <Icon />{label}
            </button>
          ))}
        </div>

        <div className="cg-nav-actions">
          <button className="cg-nav-icon-btn" aria-label="Notifications">
            <Icons.Bell />{lastResult && <span className="dot" />}
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.2 }}>
                {user?.full_name ?? user?.email}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.2 }}>Assureur</div>
            </div>
            <div className="cg-avatar" style={{ background: 'var(--accent)' }}>
              {(user?.full_name?.[0] ?? user?.email?.[0] ?? 'A').toUpperCase()}
            </div>
            <button
              onClick={logout}
              title="Se déconnecter"
              style={{ background: 'none', border: '1px solid var(--border)', borderRadius: '6px', padding: '5px 8px', cursor: 'pointer', color: 'var(--text-secondary)', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <Icons.LogOut style={{ width: 14, height: 14 }} />
            </button>
          </div>
        </div>
      </nav>

      {/* ── Sidebar ────────────────────────────────────────────── */}
      <aside className="cg-sidebar">
        <div className="cg-sidebar-label">Overview</div>
        <SidebarItem icon={Icons.Home}       label="Dashboard"  active={activePage === 'dashboard'} onClick={() => setPageAndPath('dashboard')} />
        <SidebarItem icon={Icons.PlusCircle} label="New Claim"  active={activePage === 'submit'}    onClick={() => setPageAndPath('submit')} />

        <div className="cg-sidebar-label">Management</div>
        <SidebarItem icon={Icons.Database} label="Admin View" active={activePage === 'admin'} onClick={() => setPageAndPath('admin')} />
        <SidebarItem
          icon={Icons.Inbox}
          label="All Claims"
          badge={stats.total > 0 ? stats.total : undefined}
          active={activePage === 'claims' || activePage === 'database'}
          onClick={() => setPageAndPath('database')}
        />
      </aside>

      {/* ── Page content ───────────────────────────────────────── */}
      <main className="cg-main">
        {renderPage()}
      </main>
    </div>
  )
}
