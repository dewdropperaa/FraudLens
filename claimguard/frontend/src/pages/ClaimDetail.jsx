import { useEffect, useState } from 'react'
import { Icons } from '../components'

export default function ClaimDetail({ claimId, onBack, api, safeText }) {
  const [claim, setClaim] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!claimId) return
    setLoading(true)
    setError('')
    api.get(`/claim/${claimId}`)
      .then(res => setClaim(res.data))
      .catch(err => setError(err?.response?.data?.detail || 'Failed to load claim details.'))
      .finally(() => setLoading(false))
  }, [claimId, api])

  const resolveIpfsUrl = (raw) => {
    if (!raw) return null
    const s = String(raw).trim()
    if (s.startsWith('https://') || s.startsWith('http://')) return s
    const bare = s.replace(/^ipfs:\/\//, '')
    return bare ? `https://gateway.pinata.cloud/ipfs/${bare}` : null
  }

  const decisionColor = (dec) => {
    if (dec === 'APPROVED') return { bg: '#f0fdf4', border: '#bbf7d0', text: '#15803d' }
    if (dec === 'HUMAN_REVIEW') return { bg: '#fffbeb', border: '#fde68a', text: '#92400e' }
    return { bg: '#fef2f2', border: '#fecaca', text: '#b91c1c' }
  }

  const getExplanation = (dec) => {
    if (dec === 'APPROVED') return {
      title: 'This claim was approved because:',
      points: [
        "The patient's identity was successfully verified",
        'The submitted documents are consistent and valid',
        'The claim complies with the insurance policy',
        'No significant fraud indicators were detected',
      ],
    }
    if (dec === 'HUMAN_REVIEW') return {
      title: 'This claim requires manual review because:',
      points: [
        'Some important information is missing or unclear',
        'Certain elements could not be fully verified automatically',
      ],
    }
    return {
      title: 'This claim was rejected because:',
      points: [
        'The provided information is inconsistent or invalid',
        'The claim does not meet the insurance requirements',
      ],
    }
  }

  if (loading) {
    return (
      <div className="cg-card">
        <div className="cg-empty"><span className="cg-spinner" style={{ width: 18, height: 18 }} /><div style={{ marginTop: 8 }}>Loading claim details…</div></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="cg-card">
        <div className="cg-alert error"><Icons.AlertTriangle />{safeText(error)}</div>
        <button onClick={onBack} className="cg-btn cg-btn-primary" style={{ marginTop: 12 }}>Back to Dashboard</button>
      </div>
    )
  }

  if (!claim) return null

  const colors = decisionColor(claim.decision)
  const explanation = getExplanation(claim.decision)
  const ipfsLink = resolveIpfsUrl(claim.ipfs_hash)

  return (
    <>
      <div className="cg-page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={onBack} style={{ background: 'none', border: '1px solid var(--border)', borderRadius: 6, padding: '6px 10px', cursor: 'pointer', color: 'var(--text-secondary)', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
            <Icons.ArrowLeft style={{ width: 14, height: 14 }} />Back
          </button>
          <div>
            <div className="cg-page-title">Claim Details</div>
            <div className="cg-page-sub mono" style={{ fontSize: 12 }}>{claim.claim_id}</div>
          </div>
        </div>
      </div>

      {/* Decision Banner */}
      <div style={{ padding: '16px 20px', borderRadius: 12, background: colors.bg, border: `1px solid ${colors.border}`, marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <div style={{ color: colors.text, width: 24, height: 24 }}>{claim.decision === 'APPROVED' ? <Icons.CheckCircle /> : <Icons.XCircle />}</div>
          <span style={{ fontWeight: 700, fontSize: 18, color: colors.text }}>{claim.decision}</span>
          <span style={{ fontSize: 14, color: '#6b7280', marginLeft: 'auto' }}>Score: {Math.round(claim.score ?? claim.Ts ?? 0)}/100</span>
        </div>
        <div style={{ fontSize: 13, color: '#374151', fontWeight: 500, marginBottom: 6 }}>{explanation.title}</div>
        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#374151', lineHeight: 1.8 }}>
          {explanation.points.map((p, i) => <li key={i}>{p}</li>)}
        </ul>
      </div>

      {/* Details Grid */}
      <div className="cg-info-row" style={{ marginBottom: 16 }}>
        <div className="cg-card" style={{ flex: 1 }}>
          <div className="cg-card-header"><div className="cg-card-title">Blockchain Record</div></div>
          <div style={{ padding: '12px 16px' }}>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 4 }}>Transaction Hash</div>
              <div className="mono" style={{ fontSize: 12, wordBreak: 'break-all', color: claim.tx_hash ? '#111827' : '#9ca3af' }}>
                {claim.tx_hash || 'N/A'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 4 }}>IPFS Document</div>
              {ipfsLink ? (
                <a href={ipfsLink} target="_blank" rel="noopener noreferrer" style={{ fontSize: 12, color: '#2563eb', textDecoration: 'underline' }}>
                  View Document
                </a>
              ) : (
                <span style={{ fontSize: 12, color: '#9ca3af' }}>N/A</span>
              )}
            </div>
          </div>
        </div>

        <div className="cg-card" style={{ flex: 1 }}>
          <div className="cg-card-header"><div className="cg-card-title">Claim Information</div></div>
          <div style={{ padding: '12px 16px' }}>
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 2 }}>Consensus Decision</div>
              <div style={{ fontSize: 13 }}>{claim.consensus_decision || claim.decision}</div>
            </div>
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 2 }}>Confidence Score</div>
              <div style={{ fontSize: 13 }}>{Math.round(claim.Ts ?? claim.score ?? 0)}/100</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 2 }}>Timestamp</div>
              <div style={{ fontSize: 12, color: '#6b7280' }}>{claim.timestamp ? new Date(claim.timestamp).toLocaleString() : 'N/A'}</div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
