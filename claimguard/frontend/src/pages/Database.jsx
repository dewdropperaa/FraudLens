import { Icons } from '../components'

const FILTERS = ['all', 'fraud', 'valid']

export default function Database({ claims, claimsLoading, claimsError, filter, setFilter, fetchClaims, shortHex, toIpfsUrl, safeText }) {
  return (
    <>
      <div className="cg-page-header">
        <div>
          <div className="cg-page-title">Database</div>
          <div className="cg-page-sub">Browse, filter and inspect all processed claims</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="cg-filter-group">
            {FILTERS.map((entry) => (
              <button key={entry} type="button" onClick={() => setFilter(entry)}
                className={`cg-filter-pill${filter === entry ? ` active-${entry}` : ''}`}>
                {entry === 'all' ? 'All' : entry === 'fraud' ? 'Fraud' : 'Valid'}
              </button>
            ))}
          </div>
          <button className="cg-btn cg-btn-ghost cg-btn-sm" onClick={() => fetchClaims(filter)}>
            <Icons.RefreshCw /> Refresh
          </button>
        </div>
      </div>

      {/* Summary totals */}
      <div className="cg-stats-row" style={{ gridTemplateColumns: 'repeat(3,1fr)' }}>
        <div className="cg-stat-card">
          <div className="cg-stat-icon-wrap blue"><Icons.Inbox /></div>
          <div className="cg-stat-label">Showing</div>
          <div className="cg-stat-value">{claimsLoading ? '—' : claims.length}</div>
          <div className="cg-stat-sub">claims ({filter})</div>
        </div>
        <div className="cg-stat-card">
          <div className="cg-stat-icon-wrap green"><Icons.CheckCircle /></div>
          <div className="cg-stat-label">Approved</div>
          <div className="cg-stat-value">{claimsLoading ? '—' : claims.filter(c => c.decision === 'APPROVED').length}</div>
        </div>
        <div className="cg-stat-card">
          <div className="cg-stat-icon-wrap red"><Icons.XCircle /></div>
          <div className="cg-stat-label">Rejected</div>
          <div className="cg-stat-value">{claimsLoading ? '—' : claims.filter(c => c.decision === 'REJECTED').length}</div>
        </div>
      </div>

      <div className="cg-card">
        <div className="cg-card-header">
          <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.Database /></div>Claims List</div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{claims.length} record{claims.length !== 1 ? 's' : ''}</span>
        </div>

        {claimsLoading && <div className="cg-empty"><span className="cg-spinner" style={{ width: 18, height: 18, color: '#6366f1' }} /><div style={{ marginTop: 8 }}>Loading claims…</div></div>}
        {claimsError && <div className="cg-alert error"><Icons.AlertTriangle />{safeText(claimsError)}</div>}

        {!claimsLoading && !claimsError && (
          <div className="cg-table-wrap">
            <table className="cg-table">
              <thead>
                <tr>
                  <th>Claim ID</th>
                  <th>Decision</th>
                  <th>Score</th>
                  <th>Blockchain Tx</th>
                  <th>IPFS</th>
                </tr>
              </thead>
              <tbody>
                {claims.map((claim) => (
                  <tr key={claim.claim_id}>
                    <td className="mono">{shortHex(claim.claim_id)}</td>
                    <td className={claim.decision === 'APPROVED' ? 'approved' : 'rejected'}>{claim.decision}</td>
                    <td className="score">{claim.score}</td>
                    <td className="mono">{shortHex(claim.tx_hash)}</td>
                    <td>
                      {claim.ipfs_hash
                        ? <a href={toIpfsUrl(claim.ipfs_hash)} target="_blank" rel="noreferrer">{shortHex(claim.ipfs_hash)}</a>
                        : <span style={{ color: 'var(--text-muted)' }}>N/A</span>}
                    </td>
                  </tr>
                ))}
                {claims.length === 0 && (
                  <tr><td colSpan={5} style={{ padding: '28px', textAlign: 'center', color: 'var(--text-muted)' }}>No claims match this filter.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
