"""
Deploy ClaimValidator.sol to the currently configured EVM network.

Requirements:
    pip install web3 py-solc-x python-dotenv

Usage:
    cd ClaimGuard_v2
    python contracts/deploy.py
"""
import os
import sys

from dotenv import load_dotenv
from web3 import Web3

load_dotenv("claimguard/.env")

RPC_URL = (
    os.getenv("SEPOLIA_RPC_URL")
    or os.getenv("ALCHEMY_URL")
    or os.getenv("WEB3_PROVIDER_URL")
    or os.getenv("GANACHE_RPC_URL")
    or ""
)
PRIVATE_KEY = (
    os.getenv("SEPOLIA_PRIVATE_KEY")
    or os.getenv("PRIVATE_KEY")
    or os.getenv("GANACHE_PRIVATE_KEY")
    or ""
)

if not RPC_URL or not PRIVATE_KEY:
    print("ERROR: Set SEPOLIA_RPC_URL/ALCHEMY_URL and SEPOLIA_PRIVATE_KEY/PRIVATE_KEY in claimguard/.env")
    sys.exit(1)

# ── Compile ────────────────────────────────────────────────────────────────
try:
    from solcx import compile_source, install_solc
    install_solc("0.8.19")
except ImportError:
    print("ERROR: run  pip install py-solc-x")
    sys.exit(1)

sol_path = os.path.join(os.path.dirname(__file__), "ClaimValidator.sol")
with open(sol_path) as f:
    source = f.read()

compiled = compile_source(source, output_values=["abi", "bin"], solc_version="0.8.19")
contract_id = "<stdin>:ClaimValidator"
abi      = compiled[contract_id]["abi"]
bytecode = compiled[contract_id]["bin"]

# ── Connect ────────────────────────────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    print(f"ERROR: cannot connect to {RPC_URL}")
    sys.exit(1)

account = w3.eth.account.from_key(PRIVATE_KEY)
print(f"Deploying from: {account.address}")
print(f"Balance:        {w3.from_wei(w3.eth.get_balance(account.address), 'ether'):.4f} ETH")
print(f"Chain ID:       {w3.eth.chain_id}")

# ── Deploy ─────────────────────────────────────────────────────────────────
Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
tx = Contract.constructor().build_transaction({
    "from":     account.address,
    "nonce":    w3.eth.get_transaction_count(account.address),
    "gasPrice": int(w3.eth.gas_price * 1.1),
    "gas":      1_500_000,
    "chainId":  int(w3.eth.chain_id),

})

signed  = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
print(f"Tx sent: {w3.to_hex(tx_hash)}")
print("Waiting for receipt…")

receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
if receipt["status"] != 1:
    print("FAILED — transaction reverted")
    sys.exit(1)

contract_address = receipt["contractAddress"]
print(f"\nContract deployed at: {contract_address}")
print(f"\nAdd this to claimguard/.env:")
print(f"  CONTRACT_ADDRESS={contract_address}")
