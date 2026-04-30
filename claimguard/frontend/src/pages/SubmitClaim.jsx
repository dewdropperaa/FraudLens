import { Icons } from '../components'

export default function SubmitClaim({ form, handleInputChange, handleFileChange, handleSubmit, isSubmitting, submitError, selectedFiles, lastResult, hasValidTxHash, safeText, shortHex, toIpfsUrl, scorePercent, currentClaimId }) {
  const resolveNumeric = (value) => {
    const n = Number(value)
    return Number.isFinite(n) ? n : null
  }

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
                  <div className="cg-info-box-label">Score de confiance</div>
                  {(() => {
                    const displayScore = resolveNumeric(lastResult?.score ?? lastResult?.Ts)
                    if (displayScore === null) {
                      return <div className="cg-alert error" style={{ marginTop: '8px' }}><Icons.AlertTriangle />Score indisponible: champs de score manquants dans la reponse.</div>
                    }
                    return (
                      <>
                        <div className="cg-info-box-value score">{safeText(displayScore)}/100</div>
                        <div className="cg-score-bar"><div className="cg-score-bar-fill" style={{ width: `${scorePercent(displayScore)}%`, backgroundColor: '#22c55e' }} /></div>
                      </>
                    )
                  })()}
                  <div className="cg-page-sub" style={{ marginTop: 6, fontSize: 11 }}>Seuil d'approbation: 65</div>
                </div>
                <div className="cg-info-box">
                  <div className="cg-info-box-label">Claim ID</div>
                  <div className="cg-info-box-value mono">{lastResult.claim_id || 'N/A'}</div>
                </div>
              </div>
              {lastResult.decision === 'APPROVED' && (
                <div className="cg-info-row">
                  <div className="cg-info-box">
                    <div className="cg-info-box-label">Blockchain Tx</div>
                    {lastResult.blockchain_tx || lastResult.tx_hash
                      ? <span className="cg-info-box-value mono" style={{ fontSize: 12, wordBreak: 'break-all' }}>{safeText(lastResult.blockchain_tx || lastResult.tx_hash)}</span>
                      : <span className="cg-info-box-value" style={{ color: '#dc2626', fontSize: 12 }}>Transaction failed</span>}
                  </div>
                  <div className="cg-info-box">
                    <div className="cg-info-box-label">IPFS Document</div>
                    {(() => {
                      const ipfs_hash = String(lastResult.ipfs_hash || lastResult.ipfs_document || '').replace(/^ipfs:\/\//, '').trim()
                      return ipfs_hash
                        ? <a href={`https://ipfs.io/ipfs/${ipfs_hash}`} target="_blank" rel="noopener noreferrer" className="cg-info-box-value mono link" style={{ fontSize: 12, wordBreak: 'break-all', color: '#2563eb', textDecoration: 'underline' }}>View Document</a>
                        : <span className="cg-info-box-value" style={{ color: '#f59e0b', fontSize: 12 }}>Non disponible</span>
                    })()}
                  </div>
                </div>
              )}
              {/* ── Human-Readable Decision Explanation ── */}
              {(() => {
                const dec = lastResult.decision
                const isApproved = dec === 'APPROVED'
                const isReview   = dec === 'HUMAN_REVIEW'

                const details = lastResult.decision_details || {}

                let explanationTitle, explanationPoints
                if (isApproved) {
                  explanationTitle = 'This claim was approved because:'
                  explanationPoints = [
                    "The patient's identity was successfully verified",
                    'The submitted documents are consistent and valid',
                    'The claim complies with the insurance policy',
                    'No significant fraud indicators were detected',
                  ]
                } else if (isReview) {
                  explanationTitle = 'This claim requires manual review because:'
                  explanationPoints = [
                    'Some important information is missing or unclear',
                    'Certain elements could not be fully verified automatically',
                  ]
                } else {
                  explanationTitle = 'This claim was rejected because:'
                  explanationPoints = [
                    'The provided information is inconsistent or invalid',
                    'The claim does not meet the insurance requirements',
                  ]
                }

                if (details.missing_fields && details.missing_fields.length > 0) {
                  explanationPoints.push(`Missing fields: ${details.missing_fields.join(', ')}`)
                }

                return (
                  <>
                    <div className="cg-divider" />
                    <div style={{ margin: '12px 0', padding: '14px 16px', borderRadius: 10, background: isApproved ? '#f0fdf4' : isReview ? '#fffbeb' : '#fef2f2', border: `1px solid ${isApproved ? '#bbf7d0' : isReview ? '#fde68a' : '#fecaca'}` }}>
                      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, color: isApproved ? '#15803d' : isReview ? '#92400e' : '#b91c1c' }}>
                        Decision Summary
                      </div>
                      <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.6, fontWeight: 500, marginBottom: 6 }}>
                        {explanationTitle}
                      </div>
                      <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#374151', lineHeight: 1.8 }}>
                        {explanationPoints.map((point, i) => (
                          <li key={i}>{point}</li>
                        ))}
                      </ul>
                      {details.confidence_reason && (
                        <div style={{ marginTop: 10, fontSize: 12, color: '#6b7280', fontStyle: 'italic' }}>
                          {details.confidence_reason}
                        </div>
                      )}
                    </div>
                  </>
                )
              })()}

            </>
          )}
        </div>
      </div>
    </>
  )
}
