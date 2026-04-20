# ClaimGuard - Multi-Agent AI Insurance Claim Verification System

A sophisticated multi-agent AI system for automated insurance claim verification using CrewAI and FastAPI.

---

## HOW TO RUN THE PROJECT (Step by Step)

### Prerequisites
- Python 3.10+ installed
- Node.js 18+ installed
- Git Bash or PowerShell terminal

---

### STEP 1 — Activate the virtual environment

Open a terminal at the **repo root** (`ClaimGuard_v2/`):

```bash
# On Windows (Git Bash / bash)
source venv/Scripts/activate

# On Windows (PowerShell)
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your prompt.

---

### STEP 2 — Install Python dependencies

```bash
pip install -r claimguard/requirements.txt
```

> This installs fastapi, uvicorn, crewai, langchain, web3, and all other backend dependencies.
> It will take several minutes the first time.

---

### STEP 3 — Set up environment variables

```bash
cp claimguard/.env.example claimguard/.env
```

Then open `claimguard/.env` and set at minimum:

```
ENVIRONMENT=development
DOCUMENT_ENCRYPTION_KEY=your32characterkeyhere1234567890
CLAIMAGUARD_API_KEYS=my-secret-key
```

> For development, you can skip Firebase, Ethereum, and Pinata variables.

---

### STEP 4 — Start the backend server

```bash
cd claimguard
python main.py
```

Or with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at: **http://localhost:8000**
API docs at: **http://localhost:8000/docs**

---

### STEP 5 — Start the frontend (new terminal)

Open a **second terminal**, then:

```bash
cd claimguard/frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:5173**

---

### STEP 6 — Set up frontend environment variables

```bash
cp claimguard/frontend/.env.example claimguard/frontend/.env
```

Default values should work for local dev:

```
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT_MS=30000
VITE_PROXY_TARGET=http://localhost:8000
VITE_CLAIMAGUARD_API_KEY=my-secret-key
```

> `VITE_CLAIMAGUARD_API_KEY` must match what you set in the backend `CLAIMAGUARD_API_KEYS`.

---

### Quick summary

| Terminal | Command | URL |
|----------|---------|-----|
| 1 (backend) | `source venv/Scripts/activate` then `cd claimguard && python main.py` | http://localhost:8000 |
| 2 (frontend) | `cd claimguard/frontend && npm run dev` | http://localhost:5173 |

---

### Run tests

```bash
cd claimguard
pytest test_claims.py -v -s
```

---

## Architecture

ClaimGuard uses a multi-agent architecture with 5 specialized AI agents:

### Agents

1. **Anomaly Agent** - Detects anomalous behavior patterns in claim submissions
   - Analyzes claim amounts vs historical averages
   - Detects high-frequency recent claims
   - Flags unusually high claim amounts
   - Checks documentation sufficiency

2. **Pattern Agent** - Identifies statistical patterns indicative of fraud
   - Calculates z-scores for amount deviations
   - Detects suspicious claim intervals
   - Identifies similar amount patterns
   - Cross-checks patient ID validity

3. **Identity Agent** - Verifies patient identity and detects identity fraud
   - Validates patient ID format (8+ digits, numeric only)
   - Detects multiple patient IDs in history
   - Checks for inconsistent patient names
   - Flags high claim frequency

4. **Document Agent** - Verifies authenticity of submitted documents
   - Checks for required documents (medical_report, invoice, prescription)
   - Validates documentation for high-amount claims
   - Detects suspicious document naming patterns

5. **Policy Agent** - Validates insurance coverage and policy compliance
   - Validates insurance provider (CNSS, CNOPS)
   - Checks coverage limits (CNSS: 30k, CNOPS: 50k)
   - Enforces annual claim limits
   - Tracks recent rejections

### Consensus System

The consensus system implements the following decision logic:
- Computes average score across all agents (0-100)
- **REJECT** if ANY agent returns negative decision
- **APPROVE** only if ALL agents approve AND average score >= 75

## Project Structure

```
claimguard/
├── main.py                 # FastAPI application and routes
├── models.py               # Pydantic data models
├── requirements.txt        # Python dependencies
├── test_claims.py          # Comprehensive test suite (10 tests)
├── agents/
│   ├── __init__.py
│   ├── base_agent.py      # Abstract base agent class
│   ├── anomaly_agent.py    # Anomaly detection agent
│   ├── pattern_agent.py    # Pattern detection agent
│   ├── identity_agent.py   # Identity verification agent
│   ├── document_agent.py   # Document verification agent
│   └── policy_agent.py     # Policy validation agent
└── services/
    ├── __init__.py
    └── consensus.py        # Consensus system implementation
```

## Installation

1. Navigate to the project directory:
```bash
cd claimguard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

Start the FastAPI server:
```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`

Alternatively, use uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /claim

Submit a new claim for verification.

**Request Body:**
```json
{
  "patient_id": "12345678",
  "amount": 5000.0,
  "documents": ["medical_report", "invoice", "prescription"],
  "history": [
    {"amount": 2000.0, "date": "2024-01-15", "recent": false}
  ],
  "insurance": "CNSS"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Claim processed successfully",
  "data": {
    "claim_id": "uuid-here",
    "decision": "APPROVED",
    "score": 96.0,
    "agent_results": [
      {
        "agent_name": "Anomaly Agent",
        "decision": true,
        "score": 100.0,
        "reasoning": "No anomalies detected",
        "details": {}
      }
    ],
    "timestamp": "2024-03-28T12:00:00"
  }
}
```

### GET /claim/{claim_id}

Retrieve claim results by ID.

**Response:**
```json
{
  "success": true,
  "message": "Claim retrieved successfully",
  "data": {
    "claim_id": "uuid-here",
    "decision": "APPROVED",
    "score": 96.0,
    "agent_results": [...],
    "timestamp": "2024-03-28T12:00:00"
  }
}
```

## Testing

Run the comprehensive test suite:
```bash
pytest test_claims.py -v -s
```

### Test Cases

The test suite includes 10 comprehensive tests:

1. **Valid CNSS Claim** - Tests approval for legitimate CNSS claim
2. **Valid CNOPS Claim** - Tests approval for legitimate CNOPS claim
3. **Fraud: High Amount** - Tests rejection for unusually high claims
4. **Fraud: Invalid Patient ID** - Tests rejection for invalid patient IDs
5. **Fraud: Insufficient Documents** - Tests rejection for missing documentation
6. **Fraud: Multiple Recent Claims** - Tests rejection for claim spamming
7. **Get Claim** - Tests claim retrieval functionality
8. **Get Nonexistent Claim** - Tests 404 error handling
9. **Invalid Insurance** - Tests rejection for invalid insurance providers
10. **Edge Case: Zero Amount** - Tests validation error handling

All tests pass successfully with proper fraud detection and approval logic.

## API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Example Usage

### Using cURL

Submit a claim:
```bash
curl -X POST "http://localhost:8000/claim" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "12345678",
    "amount": 5000.0,
    "documents": ["medical_report", "invoice", "prescription"],
    "history": [],
    "insurance": "CNSS"
  }'
```

Retrieve a claim:
```bash
curl -X GET "http://localhost:8000/claim/{claim_id}"
```

### Using Python

```python
import requests

# Submit a claim
response = requests.post(
    "http://localhost:8000/claim",
    json={
        "patient_id": "12345678",
        "amount": 5000.0,
        "documents": ["medical_report", "invoice", "prescription"],
        "history": [],
        "insurance": "CNSS"
    }
)

result = response.json()
print(f"Decision: {result['data']['decision']}")
print(f"Score: {result['data']['score']}")

# Retrieve claim
claim_id = result['data']['claim_id']
response = requests.get(f"http://localhost:8000/claim/{claim_id}")
claim = response.json()
print(claim)
```

## Decision Logic

The system uses a consensus-based approach:

1. Each agent independently analyzes the claim and returns:
   - `decision`: boolean (approve/reject)
   - `score`: float (0-100)
   - `reasoning`: string explanation
   - `details`: dict with specific findings

2. Consensus system evaluates:
   - If ANY agent returns `decision: false` → **REJECTED**
   - If ALL agents return `decision: true` AND average score >= 75 → **APPROVED**
   - Otherwise → **REJECTED**

3. Final response includes:
   - Overall decision (APPROVED/REJECTED)
   - Average score across all agents
   - Detailed breakdown from each agent

## Fraud Detection Capabilities

The system detects various fraud patterns:

- **Identity Fraud**: Invalid patient IDs, multiple identities
- **Amount Fraud**: Unusually high claims, statistical anomalies
- **Frequency Fraud**: Multiple recent claims, suspicious intervals
- **Documentation Fraud**: Missing documents, suspicious naming
- **Policy Fraud**: Invalid insurance, coverage limit violations

## Dependencies

- FastAPI - Web framework
- CrewAI - Multi-agent orchestration
- LangChain - Tool integration
- Pydantic - Data validation
- Uvicorn - ASGI server
- Pytest - Testing framework

## License

This project is part of the ClaimGuard insurance claim verification system.
