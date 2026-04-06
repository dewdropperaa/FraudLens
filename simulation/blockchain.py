"""
Connects to Polygon Amoy and provides client functions for both
deployed contracts: GovernanceAudit and AccessControl.
"""

from web3 import Web3
from dotenv import load_dotenv
import os

load_dotenv()

# ── env vars ─────────────────────────────────────────────
RPC_URL  = os.getenv("ALCHEMY_URL")
assert RPC_URL, (
    "Set ALCHEMY_URL in .env\n"
    "Get a free key at: https://alchemy.com → Ethereum → Sepolia"
)
PRIVATE_KEY      = os.getenv("PRIVATE_KEY")
GOVERNANCE_ADDR  = os.getenv("GOVERNANCE_AUDIT_ADDRESS")
ACCESS_CTRL_ADDR = os.getenv("ACCESS_CONTROL_ADDRESS")

# ── ABIs (minimal — only functions used in simulation) ───
GOVERNANCE_ABI = [
    {
        "name": "logAudit",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "ipfsHash", "type": "string"},
            {"name": "summary",  "type": "string"},
            {"name": "severity", "type": "uint8"}
        ],
        "outputs": []
    },
    {
        "name": "propose",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "description", "type": "string"}],
        "outputs": [{"name": "proposalId", "type": "uint256"}]
    },
    {
        "name": "vote",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "proposalId", "type": "uint256"},
            {"name": "support",    "type": "bool"}
        ],
        "outputs": []
    },
    {
        "name": "proposalCount",
        "type": "function",
        "stateMutability": "view",
        "inputs":  [],
        "outputs": [{"name": "", "type": "uint256"}]
    },
    {
        "name": "proposals",
        "type": "function",
        "stateMutability": "view",
        "inputs":  [{"name": "", "type": "uint256"}],
        "outputs": [
            {"name": "description",  "type": "string"},
            {"name": "votesFor",     "type": "uint256"},
            {"name": "votesAgainst", "type": "uint256"},
            {"name": "timestamp",    "type": "uint256"},
            {"name": "executed",     "type": "bool"}
        ]
    },
    {
        "name": "AuditLogged",
        "type": "event",
        "inputs": [
            {"name": "agent",     "type": "address", "indexed": True},
            {"name": "ipfsHash",  "type": "string",  "indexed": False},
            {"name": "summary",   "type": "string",  "indexed": False},
            {"name": "severity",  "type": "uint8",   "indexed": False},
            {"name": "timestamp", "type": "uint256", "indexed": False}
        ]
    },
    {
        "name": "Proposed",
        "type": "event",
        "inputs": [
            {"name": "proposalId",  "type": "uint256", "indexed": True},
            {"name": "proposer",    "type": "address", "indexed": True},
            {"name": "description", "type": "string",  "indexed": False},
            {"name": "timestamp",   "type": "uint256", "indexed": False}
        ]
    }
]

ACCESS_CONTROL_ABI = [
    {
        "name": "hasRole",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "role",    "type": "bytes32"},
            {"name": "account", "type": "address"}
        ],
        "outputs": [{"name": "", "type": "bool"}]
    },
    {
        "name": "grantRole",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "role",    "type": "bytes32"},
            {"name": "account", "type": "address"}
        ],
        "outputs": []
    }
]

# ── role constants ────────────────────────────────────────
AUDITOR_ROLE  = Web3.keccak(text="AUDITOR_ROLE")
GOVERNOR_ROLE = Web3.keccak(text="GOVERNOR_ROLE")
REPORTER_ROLE = Web3.keccak(text="REPORTER_ROLE")


class BlockchainClient:
    """Single client for both deployed contracts on Polygon Amoy."""

    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))
        assert self.w3.is_connected(), (
            "Cannot connect to Alchemy.\n"
            "Fix: check ALCHEMY_URL in .env\n"
            "Get a free key at: https://alchemy.com"
        )
        self.account = self.w3.eth.account.from_key(PRIVATE_KEY)

        assert GOVERNANCE_ADDR, "Set GOVERNANCE_AUDIT_ADDRESS in .env"
        assert ACCESS_CTRL_ADDR, "Set ACCESS_CONTROL_ADDRESS in .env"

        self.governance = self.w3.eth.contract(
            address=Web3.to_checksum_address(GOVERNANCE_ADDR),
            abi=GOVERNANCE_ABI
        )
        self.access_ctrl = self.w3.eth.contract(
            address=Web3.to_checksum_address(ACCESS_CTRL_ADDR),
            abi=ACCESS_CONTROL_ABI
        )

        # nonce tracker — incremented locally after each tx
        self._nonce = None

    # ── read functions (free, no gas) ────────────────────

    def check_role(self, role: bytes, address: str) -> bool:
        """Check if an address has a role in AccessControl contract."""
        try:
            return self.access_ctrl.functions.hasRole(
                role,
                Web3.to_checksum_address(address)
            ).call()
        except Exception as e:
            print(f"[check_role error] {e}")
            return False

    def get_proposal_count(self) -> int:
        """Return total number of proposals on-chain."""
        try:
            return self.governance.functions.proposalCount().call()
        except Exception as e:
            print(f"[get_proposal_count error] {e}")
            return -1

    def get_proposal(self, proposal_id: int) -> dict:
        """Fetch a proposal from on-chain by ID."""
        try:
            p = self.governance.functions.proposals(proposal_id).call()
            return {
                "id":           proposal_id,
                "description":  p[0],
                "votes_for":    p[1],
                "votes_against":p[2],
                "timestamp":    p[3],
                "executed":     p[4],
            }
        except Exception as e:
            return {"error": str(e)}

    def read_state(self) -> dict:
        """Read current on-chain state. No gas cost."""
        wallet = self.account.address
        return {
            "connected":       self.w3.is_connected(),
            "block":           self.w3.eth.block_number,
            "wallet":          wallet,
            "proposal_count":  self.get_proposal_count(),
            "has_auditor_role":self.check_role(AUDITOR_ROLE, wallet),
            "has_governor_role":self.check_role(GOVERNOR_ROLE, wallet),
        }

    # ── write functions (spend gas) ──────────────────────

    def _send(self, fn) -> dict:
        """Sign, send, wait for receipt. Tracks nonce locally."""
        try:
            # Fetch nonce from chain only on first call,
            # then increment locally for every subsequent call
            if self._nonce is None:
                self._nonce = self.w3.eth.get_transaction_count(
                    self.account.address
                )

            tx = fn.build_transaction({
                "from":     self.account.address,
                "nonce":    self._nonce,
                "gas":      250_000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId":  self.w3.eth.chain_id,
            })

            signed  = self.w3.eth.account.sign_transaction(
                          tx, PRIVATE_KEY)
            tx_hash = self.w3.eth.send_raw_transaction(
                          signed.raw_transaction)

            # Increment BEFORE waiting — so next tx can be built
            # immediately without waiting for this one to mine
            self._nonce += 1

            receipt = self.w3.eth.wait_for_transaction_receipt(
                          tx_hash, timeout=120)

            return {
                "tx_hash": tx_hash.hex(),
                "status":  receipt.status,
                "block":   receipt.blockNumber,
            }
        except Exception as e:
            # On error, reset nonce so next call re-fetches from chain
            self._nonce = None
            return {"error": str(e)}

    def log_audit(self, summary: str, severity: int) -> dict:
        """
        Call logAudit() on GovernanceAudit contract.
        Requires wallet to have AUDITOR_ROLE.
        severity: 0=INFO 1=LOW 2=MEDIUM 3=HIGH 4=CRITICAL
        """
        fn = self.governance.functions.logAudit(
            "QmSimulation",   # ipfsHash placeholder for simulation
            summary,
            severity
        )
        return self._send(fn)

    def create_proposal(self, description: str) -> dict:
        """Call propose() on GovernanceAudit contract."""
        fn = self.governance.functions.propose(description)
        result = self._send(fn)
        if result.get("status") == 1:
            # Read back the new proposal count
            result["proposal_id"] = self.get_proposal_count()
        return result

    def cast_vote(self, proposal_id: int, support: bool) -> dict:
        """Call vote() on GovernanceAudit contract."""
        fn = self.governance.functions.vote(proposal_id, support)
        return self._send(fn)

    def preflight_check(self) -> dict:
        """
        Run before simulation. Checks all requirements are met.
        Returns dict with pass/fail for each check.
        """
        wallet = self.account.address
        balance = self.w3.eth.get_balance(wallet)
        balance_matic = self.w3.from_wei(balance, "ether")

        detected = self.w3.eth.chain_id

        checks = {
            "rpc_connected":      self.w3.is_connected(),
            "wallet_has_balance": float(balance_matic) > 0.01,
            "chain_id":           detected,
            "has_auditor_role":   self.check_role(AUDITOR_ROLE, wallet),
            "has_governor_role":  self.check_role(GOVERNOR_ROLE, wallet),
            "governance_deployed":len(self.w3.eth.get_code(
                Web3.to_checksum_address(GOVERNANCE_ADDR))) > 2,
            "access_ctrl_deployed":len(self.w3.eth.get_code(
                Web3.to_checksum_address(ACCESS_CTRL_ADDR))) > 2,
        }

        assert detected == 11155111, (
            f"Wrong network. Expected Sepolia (11155111), "
            f"got {detected}. Check ALCHEMY_URL."
        )

        checks["all_passed"] = all(checks.values())
        return checks
