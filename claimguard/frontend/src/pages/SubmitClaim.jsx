import { Icons } from '../components'

export default function SubmitClaim({ form, handleInputChange, handleFileChange, handleSubmit, isSubmitting, submitError, selectedFiles, lastResult, hasValidTxHash, safeText, shortHex, toIpfsUrl, scorePercent, currentClaimId }) {
  const getAgentBadge = (agent) => {
    const status = String(agent?.status || '').toUpperCase()
    const score = Number(agent?.score || 0)
    if (status === 'PASS') return { label: 'PASS', className: 'pass' }
    if (status === 'REVIEW') return { label: 'REVIEW', className: 'warn' }
    if (status === 'FAIL') return { label: 'FAIL', className: 'fail' }
    if (status === 'ERROR' || status === 'TIMEOUT') return { label: 'FAIL', className: 'fail' }
    if (status === 'DONE' && score >= 60) return { label: 'PASS', className: 'pass' }
    if (status === 'DONE' && score >= 40) return { label: 'REVIEW', className: 'warn' }
    return { label: 'FAIL', className: 'fail' }
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
                  <div className="cg-info-box-value score">{safeText(lastResult.score)}/100</div>
                  <div className="cg-score-bar"><div className="cg-score-bar-fill" style={{ width: `${scorePercent(lastResult.score)}%`, backgroundColor: '#22c55e' }} /></div>
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
                      : <span className="cg-info-box-value" style={{ color: '#f59e0b', fontSize: 12 }}>Calcul en cours...</span>}
                  </div>
                  <div className="cg-info-box">
                    <div className="cg-info-box-label">IPFS Document</div>
                    {lastResult.ipfs_document || lastResult.ipfs_hash
                      ? <a href={safeText(lastResult.ipfs_document || lastResult.ipfs_hash)} target="_blank" rel="noopener noreferrer" className="cg-info-box-value mono link" style={{ fontSize: 12, wordBreak: 'break-all' }}>{safeText(lastResult.ipfs_document || lastResult.ipfs_hash)}</a>
                      : <span className="cg-info-box-value" style={{ color: '#f59e0b', fontSize: 12 }}>Non disponible</span>}
                  </div>
                </div>
              )}
              {/* ── Final Analysis Summary — always shown when there's a result ── */}
              {(() => {
                const dec = lastResult.decision
                const ts  = Number(lastResult.score || lastResult.Ts || 0)
                const hasAgents = (lastResult.agent_results ?? []).length > 0
                const reasons = Array.isArray(lastResult.explanation?.reasons)
                  ? lastResult.explanation.reasons.filter(r => !r.startsWith('memory') && r.length > 3 && !r.includes(' '))
                  : []
                const isApproved = dec === 'APPROVED'
                const isReview   = dec === 'HUMAN_REVIEW'
                const isOutOfContext = reasons.includes('DOCUMENT_OUT_OF_CONTEXT') || reasons.includes('NON_CLAIM_DOCUMENT')

                const decisionText = isOutOfContext
                  ? (lastResult.explanation?.summary || `Document hors contexte — le fichier soumis n'est pas un dossier médical valide (score ${Math.round(ts)}/100). Veuillez soumettre les documents requis : rapport médical, facture et ordonnance.`)
                  : isApproved
                  ? `Ce dossier a été approuvé avec un score de confiance de ${Math.round(ts)}/100. Les agents ont validé l'identité du patient, la conformité des documents et la politique d'assurance. Aucun signal de fraude critique n'a été détecté.`
                  : isReview
                  ? `Ce dossier nécessite une révision humaine (score ${Math.round(ts)}/100). Des éléments insuffisants ou des incertitudes ont été détectés par les agents d'analyse, nécessitant une vérification complémentaire avant toute décision finale.`
                  : `Ce dossier a été rejeté (score ${Math.round(ts)}/100). Les agents ont détecté des problèmes critiques : données manquantes, incohérences documentaires ou signaux de fraude qui empêchent l'approbation automatique.`

                return (
                  <>
                    <div className="cg-divider" />
                    <div style={{ margin: '12px 0', padding: '14px 16px', borderRadius: 10, background: isApproved ? '#f0fdf4' : isReview ? '#fffbeb' : '#fef2f2', border: `1px solid ${isApproved ? '#bbf7d0' : isReview ? '#fde68a' : '#fecaca'}` }}>
                      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, color: isApproved ? '#15803d' : isReview ? '#92400e' : '#b91c1c' }}>
                        Analyse finale — {dec}
                      </div>
                      <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.6, marginBottom: reasons.length ? 10 : 0 }}>
                        {decisionText}
                      </div>
                      {reasons.length > 0 && (
                        <div style={{ marginTop: 8 }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Signaux détectés</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                            {reasons.slice(0, 6).map((r, i) => (
                              <span key={i} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 20, background: '#e5e7eb', color: '#374151' }}>{r}</span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </>
                )
              })()}

              {(lastResult.agent_results ?? []).length > 0 && (
                <>
                  <div className="cg-sub-heading" style={{ marginTop: 12 }}>Agent Breakdown</div>
                  <div className="cg-agent-list">
                    {lastResult.agent_results.map((agent) => {
                      const badge = getAgentBadge(agent)
                      const score = Number(agent?.score || 0)
                      const reason = String(agent?.explanation || agent?.reasoning || '').trim() || 'Agent completed analysis.'

                      const toolsMap = {
                        IdentityAgent:  ['identity_extractor', 'fraud_detector'],
                        DocumentAgent:  ['ocr_extractor', 'document_classifier'],
                        PolicyAgent:    ['document_classifier', 'fraud_detector'],
                        AnomalyAgent:   ['anomaly_detector', 'history_analyzer'],
                        PatternAgent:   ['pattern_detector', 'history_analyzer'],
                        GraphRiskAgent: ['fraud_ring_graph', 'network_analyzer'],
                      }
                      const tools = toolsMap[agent.agent_name] || []

                      return (
                        <div key={`${lastResult.claim_id}-${agent.agent_name}`} className="cg-agent-item">
                          <div className="cg-agent-row">
                            <span className="cg-agent-name">{agent.agent_name}</span>
                            <span className={`cg-agent-badge ${badge.className}`}>{badge.label}</span>
                          </div>
                          <div className="cg-agent-score">Score: {Math.round(score)}</div>
                          {tools.length > 0 && (
                            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', margin: '4px 0' }}>
                              {tools.map(t => (
                                <span key={t} style={{ fontSize: 10, padding: '1px 7px', borderRadius: 12, background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe', fontFamily: 'monospace' }}>
                                  {t}
                                </span>
                              ))}
                            </div>
                          )}
                          <div className="cg-agent-reasoning">{safeText(reason)}</div>
                        </div>
                      )
                    })}
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
