import { Icons } from '../components'
import {
  Bar, BarChart, CartesianGrid, Legend,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

export default function Analytics({ dailyClaims, chartLoading, stats, claimsLoading }) {
  const chartData = dailyClaims

  return (
    <>
      <div className="cg-page-header">
        <div>
          <div className="cg-page-title">Analytics</div>
          <div className="cg-page-sub">Trend analysis and fraud pattern overview</div>
        </div>
        <div className="cg-live"><span className="cg-live-dot" />Last 90 days</div>
      </div>

      {/* Summary row */}
      <div className="cg-stats-row">
        <div className="cg-stat-card">
          <div className="cg-stat-icon-wrap blue"><Icons.FileText /></div>
          <div className="cg-stat-label">Total Processed</div>
          <div className="cg-stat-value">{claimsLoading ? '—' : stats.total}</div>
        </div>
        <div className="cg-stat-card">
          <div className="cg-stat-icon-wrap green"><Icons.CheckCircle /></div>
          <div className="cg-stat-label">Approval Rate</div>
          <div className="cg-stat-value">{claimsLoading || stats.total === 0 ? '—' : `${Math.round((stats.approved / stats.total) * 100)}%`}</div>
        </div>
        <div className="cg-stat-card">
          <div className="cg-stat-icon-wrap red"><Icons.AlertTriangle /></div>
          <div className="cg-stat-label">Fraud Rate</div>
          <div className="cg-stat-value">{claimsLoading ? '—' : `${stats.fraudRate}%`}</div>
        </div>
        <div className="cg-stat-card">
          <div className="cg-stat-icon-wrap purple"><Icons.Activity /></div>
          <div className="cg-stat-label">Active Days</div>
          <div className="cg-stat-value">{chartData.length}</div>
        </div>
      </div>

      {/* Main chart */}
      <div className="cg-card">
        <div className="cg-card-header">
          <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.BarChart2 /></div>Daily Claims — Valid vs Fraud</div>
          <span className="cg-card-badge normal">Last 90 days</span>
        </div>
        {chartLoading ? (
          <div className="cg-empty"><span className="cg-spinner" style={{ width: 18, height: 18, color: '#6366f1' }} /><div style={{ marginTop: 8 }}>Loading chart…</div></div>
        ) : chartData.length === 0 ? (
          <div className="cg-empty"><div className="cg-empty-icon"><Icons.Activity /></div>No claim data in the last 90 days.</div>
        ) : (
          <ResponsiveContainer width="100%" height={300} minWidth={280}>
            <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
              <XAxis dataKey="label" tick={{ fill: '#9ca3af', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis allowDecimals={false} tick={{ fill: '#9ca3af', fontSize: 10 }} width={28} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e8eaed', borderRadius: '8px', fontSize: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} labelStyle={{ color: '#111827', fontWeight: 600 }} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
              <Legend wrapperStyle={{ fontSize: '11px', color: '#6b7280' }} />
              <Bar dataKey="valid" name="Valid (approved)" fill="#22c55e" radius={[4,4,0,0]} maxBarSize={36} />
              <Bar dataKey="fraud" name="Fraud (rejected)" fill="#ef4444" radius={[4,4,0,0]} maxBarSize={36} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Day-by-day breakdown */}
      {chartData.length > 0 && (
        <div className="cg-card">
          <div className="cg-card-header">
            <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.Database /></div>Daily Breakdown</div>
          </div>
          <div className="cg-table-wrap">
            <table className="cg-table">
              <thead><tr><th>Date</th><th>Valid</th><th>Fraud</th><th>Total</th><th>Fraud Rate</th></tr></thead>
              <tbody>
                {[...chartData].reverse().map((row) => {
                  const total = row.valid + row.fraud
                  const rate  = total > 0 ? Math.round((row.fraud / total) * 100) : 0
                  return (
                    <tr key={row.date}>
                      <td style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{row.date}</td>
                      <td className="approved">{row.valid}</td>
                      <td className="rejected">{row.fraud}</td>
                      <td style={{ color: 'var(--text-primary)' }}>{total}</td>
                      <td className="score">{rate}%</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}
