# FraudLens — AI-Powered Insurance Claim Analysis with Blockchain Trust

FraudLens is a production-ready system designed to analyze, validate, and secure insurance claims using artificial intelligence, human review workflows, and blockchain-based trust anchoring.

It is built with a security-first and audit-oriented approach to help insurers detect fraud, automate claim validation, and ensure data integrity through decentralized technologies.

---

## Core Concept

FraudLens combines:

* A multi-agent AI system for claim analysis
* A human-in-the-loop validation workflow for uncertain cases
* Blockchain anchoring for tamper-proof records
* IPFS storage for secure document management

The objective is to make claim validation transparent, explainable, and reliable.

---

## Features

### AI Claim Analysis

* Identity verification
* Document consistency checks (OCR and classification)
* Policy compliance validation
* Fraud and anomaly detection
* Behavioral and pattern analysis

---

### Human Review System

* Claims are routed to manual review when confidence is insufficient
* Administrative interface for approval or rejection
* Full traceability of all decisions

---

### Document Handling

* Upload and process PDF and image files
* OCR extraction using Tesseract
* Documents stored on IPFS
* Direct PDF preview in the dashboard (scrollable viewer)

---

### Blockchain Integration

* Every approved claim is anchored on the blockchain
* Stores:

  * IPFS document hash
  * Claim metadata

This ensures immutability, auditability, and proof of integrity.

---

### Dashboard and User Experience

* Clean interface designed for insurance professionals

* No technical jargon exposed

* Full claim lifecycle tracking:

  AI Analysis → Human Review → Final Decision

* Status examples:

  * Approved (AI)
  * Approved (after human review)
  * Rejected (after human review)

---

### Explainability Layer

* Human-readable explanations for every decision
* No raw AI outputs or JSON exposed

Example:

“This claim was approved because the identity is verified, the documents are consistent, and no fraud indicators were detected.”

---

### Inconsistency Detection

* Detects contradictions within claim data
* Displays them in clear, business-oriented language

Example:

“Some inconsistencies were detected between the provided identity and claim patterns. Further verification is recommended.”

---

## Architecture Overview

```text
Frontend (React)
    ↓
Backend API (FastAPI / Python)
    ↓
Orchestrator (Multi-Agent System)
    ↓
AI Agents (Identity, Document, Policy, Anomaly, Pattern, Graph Risk)
    ↓
Trust Layer
    ├── IPFS (Document Storage)
    └── Blockchain (Claim Anchoring)
```

---

## Claim Processing Flow

1. A user submits a claim with supporting documents

2. AI agents analyze the claim

3. The system decides:

   * Approve automatically
   * Send to human review
   * Reject

4. If approved:

   * Document is stored on IPFS
   * Hash is anchored on the blockchain

5. The dashboard is updated with:

   * Decision
   * Explanation
   * IPFS link
   * Blockchain transaction

---

## Tech Stack

### Backend

* Python
* FastAPI
* PyPDF2 / ReportLab
* Tesseract OCR
* Rule-based AI agents

### Frontend

* React.js

### Storage and Trust

* IPFS
* Blockchain

---

## Security and Compliance Approach

FraudLens is designed with:

* Data integrity through blockchain anchoring
* Audit trails for both AI and human decisions
* Clear explainability of outcomes
* Alignment with ISO 27001 and ISO 27005 principles

---

## Key Differentiators

* Hybrid AI and human validation
* Blockchain-backed trust layer
* Business-oriented explanations
* Full traceability of decisions
* Real-time claim analysis

---

## Example Output

```json
{
  "decision": "APPROVED",
  "decision_source": "HUMAN",
  "confidence_score": 66,
  "explanation": "The claim was validated after manual review. All required information is consistent and no fraud indicators were confirmed.",
  "ipfs_hash": "Qm...",
  "blockchain_tx": "0xabc123...",
  "trace": [
    "AI → Human Review → Approved"
  ]
}
```

---

## Future Improvements

* Smart contract automation for claim payouts
* Advanced fraud graph analysis
* Adaptive anomaly detection
* Integration with insurance core systems

---

## Author

Developed as a cybersecurity and AI project focused on real-world insurance applications and trust systems.

---

## Final Note

FraudLens is designed as a trust infrastructure for decision-making, covering the full lifecycle from detection to explanation to verifiable proof.
