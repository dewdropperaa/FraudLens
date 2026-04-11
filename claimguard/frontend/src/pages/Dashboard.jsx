import { Icons, StatCard } from '../components'
import {
  Bar, BarChart, CartesianGrid, Legend,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

export default function Dashboard({ stats, claimsLoading, dailyClaims, chartLoading, claims, shortHex, toIpfsUrl }) {
  return (
    <>
      <div className="cg-page-header">
        <div>
          <div className="cg-page-title">Dashboard</div>
          <div className="cg-page-sub">Real-time overview of claim processing and fraud detection</div>
        </div>
        <div className="cg-live"><span className="cg-live-dot" />Live</div>
      </div>

      <div className="cg-stats-row">
        <StatCard label="Total Claims"  value={claimsLoading ? '—' : stats.total}               sub="Processed this session" IconComp={Icons.FileText}   color="blue"   />
        <StatCard label="Approved"      value={claimsLoading ? '—' : stats.approved}            sub="Valid claims"           IconComp={Icons.CheckCircle} color="green"  />
        <StatCard label="Rejected"      value={claimsLoading ? '—' : stats.rejected}            sub="Flagged as fraud"       IconComp={Icons.XCircle}     color="red"    />
        <StatCard label="Fraud Rate"    value={claimsLoading ? '—' : `${stats.fraudRate}%`}     sub="Of all processed"       IconComp={Icons.TrendingUp}  color="purple" />
      </div>

      <div className="cg-card">
        <div className="cg-card-header">
          <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.BarChart2 /></div>Daily Claims — Valid vs Fraud</div>
        </div>
        {chartLoading ? (
          <div className="cg-empty"><span className="cg-spinner" style={{ width: 18, height: 18, color: '#6366f1' }} /><div style={{ marginTop: 8 }}>Loading chart…</div></div>
        ) : dailyClaims.length === 0 ? (
          <div className="cg-empty"><div className="cg-empty-icon"><Icons.Activity /></div>No claim data in the last 90 days.</div>
        ) : (
          <ResponsiveContainer width="100%" height={240} minWidth={280}>
            <BarChart data={dailyClaims} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
              <XAxis dataKey="label" tick={{ fill: '#9ca3af', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis allowDecimals={false} tick={{ fill: '#9ca3af', fontSize: 10 }} width={28} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e8eaed', borderRadius: '8px', fontSize: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} labelStyle={{ color: '#111827', fontWeight: 600 }} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
              <Legend wrapperStyle={{ fontSize: '11px', color: '#6b7280' }} />
              <Bar dataKey="valid" name="Valid (approved)" fill="#22c55e" radius={[4,4,0,0]} maxBarSize={32} />
              <Bar dataKey="fraud" name="Fraud (rejected)" fill="#ef4444" radius={[4,4,0,0]} maxBarSize={32} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Recent claims */}
      <div className="cg-card">
        <div className="cg-card-header">
          <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.Inbox /></div>Recent Claims</div>
        </div>
        <div className="cg-table-wrap">
          <table className="cg-table">
            <thead><tr><th>Claim ID</th><th>Decision</th><th>Score</th><th>Tx Hash</th><th>IPFS</th></tr></thead>
            <tbody>
              {claims.slice(0, 8).map((claim) => (
                <tr key={claim.claim_id}>
                  <td className="mono">{shortHex(claim.claim_id)}</td>
                  <td className={claim.decision === 'APPROVED' ? 'approved' : 'rejected'}>{claim.decision}</td>
                  <td className="score">{claim.score}</td>
                  <td className="mono">{shortHex(claim.tx_hash)}</td>
                  <td>{claim.ipfs_hash ? <a href={toIpfsUrl(claim.ipfs_hash)} target="_blank" rel="noreferrer">{shortHex(claim.ipfs_hash)}</a> : <span style={{ color: 'var(--text-muted)' }}>N/A</span>}</td>
                </tr>
              ))}
              {claims.length === 0 && <tr><td colSpan={5} style={{ padding: '28px', textAlign: 'center', color: 'var(--text-muted)' }}>No claims yet.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}
