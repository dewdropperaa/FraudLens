import { Icons } from '../components'

export default function SubmitClaim({ form, handleInputChange, handleFileChange, handleSubmit, isSubmitting, submitError, selectedFiles, lastResult, hasValidTxHash, safeText, shortHex, toIpfsUrl, scorePercent }) {
  return (
    <>
      <div className="cg-page-header">
        <div>
          <div className="cg-page-title">Submit New Claim</div>
          <div className="cg-page-sub">Complete the form — our AI agents will analyse it in real time</div>
        </div>
      </div>

      <div className="cg-two-col">
        {/* Form */}
        <div className="cg-card">
          <div className="cg-card-header">
            <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.FileText /></div>Claim Details</div>
            <span className="cg-card-badge normal">AI + Blockchain</span>
          </div>
          <form onSubmit={handleSubmit} className="cg-form">
            <div className="cg-field">
              <label htmlFor="patient_id">Patient ID</label>
              <input id="patient_id" name="patient_id" value={form.patient_id} onChange={handleInputChange} required placeholder="e.g. 12345678" />
            </div>
            <div className="cg-field">
              <label htmlFor="provider_id">Provider / Facility ID</label>
              <input id="provider_id" name="provider_id" value={form.provider_id} onChange={handleInputChange} required placeholder="e.g. CHU-Rabat-001" />
            </div>
            <div className="cg-field">
              <label htmlFor="amount">Claim Amount (MAD)</label>
              <input id="amount" name="amount" type="number" min="0" step="0.01" value={form.amount} onChange={handleInputChange} required placeholder="5000" />
            </div>
            <div className="cg-field">
              <label htmlFor="insurance">Insurance Type</label>
              <select id="insurance" name="insurance" value={form.insurance} onChange={handleInputChange}>
                <option value="CNSS">CNSS</option>
                <option value="CNOPS">CNOPS</option>
              </select>
            </div>
            <div className="cg-field">
              <label>Supporting Documents</label>
              <div className="cg-file-wrap">
                <label htmlFor="documents" className="cg-file-upload-label">
                  <Icons.Upload />
                  <span className="cg-file-upload-text">{selectedFiles.length > 0 ? `${selectedFiles.length} file${selectedFiles.length > 1 ? 's' : ''} selected` : 'Click to upload files'}</span>
                  <span className="cg-file-upload-hint">PDF, images, or text — Tesseract OCR applied</span>
                </label>
                <input id="documents" type="file" multiple onChange={handleFileChange} className="cg-file-input-hidden" />
              </div>
            </div>
            <button type="submit" disabled={isSubmitting} className="cg-btn cg-btn-primary cg-btn-full">
              {isSubmitting ? <><span className="cg-spinner" /> Processing claim…</> : 'Submit Claim'}
            </button>
          </form>
          {submitError && <div className="cg-alert error" style={{ marginTop: '12px' }}><Icons.AlertTriangle />{safeText(submitError)}</div>}
        </div>

        {/* Result */}
        <div className="cg-card">
          <div className="cg-card-header">
            <div className="cg-card-title"><div className="cg-card-title-icon"><Icons.Search /></div>Analysis Result</div>
            {lastResult && <span className={`cg-card-badge ${lastResult.decision === 'APPROVED' ? 'normal' : 'danger'}`}>{lastResult.decision}</span>}
          </div>
          {!lastResult ? (
            <div className="cg-empty"><div className="cg-empty-icon"><Icons.Shield /></div><div>Submit a claim to see the AI decision,</div><div style={{ marginTop: 4 }}>fraud score, blockchain hash, and IPFS link.</div></div>
          ) : (
            <>
              <div className={`cg-decision-banner ${lastResult.decision === 'APPROVED' ? 'approved' : 'rejected'}`}>
                <div className="cg-decision-icon">{lastResult.decision === 'APPROVED' ? <Icons.CheckCircle /> : <Icons.XCircle />}</div>
                <div><div className="cg-decision-label">Final Decision</div><div className="cg-decision-value">{lastResult.decision}</div></div>
              </div>
              <div className="cg-info-row">
                <div className="cg-info-box">
                  <div className="cg-info-box-label">Fraud Score</div>
                  <div className="cg-info-box-value score">{safeText(lastResult.score)}</div>
                  <div className="cg-score-bar"><div className="cg-score-bar-fill" style={{ width: `${scorePercent(lastResult.score)}%` }} /></div>
                </div>
                <div className="cg-info-box">
                  <div className="cg-info-box-label">Claim ID</div>
                  <div className="cg-info-box-value mono">{lastResult.claim_id || 'N/A'}</div>
                </div>
              </div>
              <div className="cg-info-row">
                <div className="cg-info-box">
                  <div className="cg-info-box-label">Blockchain Tx</div>
                  {lastResult.tx_hash
                    ? <a href={hasValidTxHash ? `https://sepolia.etherscan.io/tx/${lastResult.tx_hash}` : '#'} target="_blank" rel="noreferrer" className="cg-info-box-value mono link">{shortHex(lastResult.tx_hash)}</a>
                    : <div className="cg-info-box-value mono" style={{ color: 'var(--text-muted)' }}>{lastResult.decision === 'REJECTED' ? 'Skipped — rejected' : 'Not configured'}</div>}
                </div>
                <div className="cg-info-box">
                  <div className="cg-info-box-label">IPFS Document</div>
                  {lastResult.ipfs_hash
                    ? <a href={toIpfsUrl(lastResult.ipfs_hash)} target="_blank" rel="noreferrer" className="cg-info-box-value mono link">{shortHex(lastResult.ipfs_hash)}</a>
                    : <div className="cg-info-box-value mono" style={{ color: 'var(--text-muted)' }}>{lastResult.decision === 'REJECTED' ? 'Skipped — rejected' : 'Not configured'}</div>}
                </div>
              </div>
              {(lastResult.agent_results ?? []).length > 0 && (
                <>
                  <div className="cg-divider" />
                  <div className="cg-sub-heading">Agent Breakdown</div>
                  <div className="cg-agent-list">
                    {lastResult.agent_results.map((agent) => (
                      <div key={`${lastResult.claim_id}-${agent.agent_name}`} className="cg-agent-item">
                        <div className="cg-agent-row">
                          <span className="cg-agent-name">{agent.agent_name}</span>
                          <span className={`cg-agent-badge ${agent.decision ? 'pass' : 'fail'}`}>{agent.decision ? 'PASS' : 'FAIL'}</span>
                        </div>
                        <div className="cg-agent-score">Score: {safeText(agent.score)}</div>
                        <div className="cg-agent-reasoning">{safeText(agent.reasoning)}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </>
  )
}
