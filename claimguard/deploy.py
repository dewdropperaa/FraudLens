"""
Deployment script for ClaimValidator smart contract on Sepolia testnet
"""
import os
import sys
import json
import time
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv

from claimguard.config import get_sepolia_private_key, get_sepolia_rpc_url

# Load environment
load_dotenv()

# Contract bytecode (compiled with solc)
# Run: solc --bin contracts/ClaimValidator.sol -o build --overwrite
CONTRACT_BYTECODE = None  # Will be loaded from file or compiled

CONTRACT_ABI = [
    {"inputs": [], "stateMutability": "nonpayable", "type": "constructor"},
    {"anonymous": False, "inputs": [
        {"indexed": True, "name": "claimIdHash", "type": "bytes32"},
        {"indexed": False, "name": "score", "type": "uint256"},
        {"indexed": False, "name": "approved", "type": "bool"},
        {"indexed": False, "name": "timestamp", "type": "uint256"},
        {"indexed": False, "name": "validator", "type": "address"}
    ], "name": "ClaimValidated", "type": "event"},
    {"inputs": [{"name": "claimIdHash", "type": "bytes32"}, {"name": "score", "type": "uint256"},
     {"name": "approved", "type": "bool"}],
     "name": "validateClaim", "outputs": [{"name": "", "type": "bool"}],
     "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "claimIdHash", "type": "bytes32"}], "name": "getClaimProof",
     "outputs": [{"name": "score", "type": "uint256"}, {"name": "approved", "type": "bool"}, {"name": "documentHash", "type": "bytes32"}, {"name": "zkProofHash", "type": "bytes32"},
     {"name": "timestamp", "type": "uint256"}, {"name": "validator", "type": "address"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "claimIdHash", "type": "bytes32"}], "name": "isClaimValidated",
     "outputs": [{"name": "", "type": "bool"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "newValidator", "type": "address"}], "name": "setAuthorizedValidator",
     "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "owner", "outputs": [{"name": "", "type": "address"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "authorizedValidator", "outputs": [{"name": "", "type": "address"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "totalValidatedClaims", "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "index", "type": "uint256"}], "name": "getClaimIdByIndex",
     "outputs": [{"name": "", "type": "bytes32"}], "stateMutability": "view", "type": "function"}
]


def compile_contract():
    """Compile Solidity contract using solc"""
    try:
        import subprocess
        result = subprocess.run(
            ["solc", "--bin", "--abi", "contracts/ClaimValidator.sol", "-o", "build", "--overwrite"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode != 0:
            print(f"Compilation error: {result.stderr}")
            return None
        
        # Load bytecode
        bytecode_path = os.path.join(os.path.dirname(__file__), "build", "ClaimValidator.bin")
        with open(bytecode_path, "r") as f:
            return "0x" + f.read().strip()
    except Exception as e:
        print(f"Compilation failed: {e}")
        return None


def get_compiled_artifact():
    """Get compiled contract bytecode from file"""
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    bytecode_path = os.path.join(build_dir, "ClaimValidator.bin")
    
    if os.path.exists(bytecode_path):
        with open(bytecode_path, "r") as f:
            return "0x" + f.read().strip()
    return None


def deploy_contract(max_retries=3):
    """Deploy contract to Sepolia with retry logic"""
    preferred_rpc = get_sepolia_rpc_url()
    private_key = get_sepolia_private_key()

    if not private_key:
        print("ERROR: Set SEPOLIA_PRIVATE_KEY or PRIVATE_KEY in .env")
        return None

    # Connect to network (Alchemy first via get_sepolia_rpc_url, then fallbacks)
    rpc_candidates = [preferred_rpc]
    for url in (
        "https://rpc.sepolia.org",
        "https://ethereum-sepolia-rpc.publicnode.com",
        "https://rpc2.sepolia.org",
    ):
        if url not in rpc_candidates:
            rpc_candidates.append(url)
    w3 = None
    chosen_rpc = None
    for rpc_url in rpc_candidates:
        try:
            candidate = Web3(Web3.HTTPProvider(rpc_url))
            if candidate.is_connected():
                w3 = candidate
                chosen_rpc = rpc_url
                break
            print(f"RPC unavailable: {rpc_url}")
        except Exception as e:
            print(f"RPC error ({rpc_url}): {e}")
    
    if not w3:
        print("ERROR: Unable to connect to Sepolia via all configured RPC endpoints.")
        return None
    
    print(f"Connected to Sepolia via {chosen_rpc} (Chain ID: {w3.eth.chain_id})")
    
    # Setup account
    try:
        account = Account.from_key(private_key)
    except Exception as e:
        print(f"ERROR: Invalid SEPOLIA_PRIVATE_KEY format: {e}")
        return None
    print(f"Deployer address: {account.address}")
    
    # Check balance
    balance = w3.eth.get_balance(account.address)
    balance_eth = w3.from_wei(balance, 'ether')
    print(f"Balance: {balance_eth} ETH")
    
    if balance == 0:
        print("ERROR: No ETH for gas. Get Sepolia ETH from faucet:")
        print("  https://sepoliafaucet.com/")
        print("  https://www.alchemy.com/faucets/ethereum-sepolia")
        return None
    
    # Get bytecode
    bytecode = get_compiled_artifact()
    if not bytecode:
        print("Attempting to compile contract...")
        bytecode = compile_contract()
    
    if not bytecode:
        print("ERROR: Could not get contract bytecode. Install solc:")
        print("  macOS: brew install solidity")
        print("  Linux: sudo snap install solc")
        print("  Windows: Download from https://github.com/ethereum/solidity/releases")
        return None
    
    # Build deployment transaction
    for attempt in range(max_retries):
        try:
            print(f"\nDeploying contract (attempt {attempt + 1}/{max_retries})...")
            
            nonce = w3.eth.get_transaction_count(account.address)
            gas_price = int(w3.eth.gas_price * 1.1)
            
            contract = w3.eth.contract(abi=CONTRACT_ABI, bytecode=bytecode)
            
            # Estimate gas
            tx = contract.constructor().build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gasPrice': gas_price,
            })
            
            # Set gas with buffer
            tx['gas'] = int(w3.eth.estimate_gas(tx) * 1.2)
            
            # Sign and send
            signed = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            tx_hash_hex = w3.to_hex(tx_hash)
            
            print(f"Transaction sent: {tx_hash_hex}")
            print(f"Explorer: https://sepolia.etherscan.io/tx/{tx_hash_hex}")
            
            # Wait for confirmation
            print("Waiting for confirmation...")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                contract_address = receipt['contractAddress']
                print(f"\n✓ Contract deployed successfully!")
                print(f"  Address: {contract_address}")
                print(f"  Block: {receipt['blockNumber']}")
                print(f"  Gas used: {receipt['gasUsed']}")
                
                # Update .env
                update_env_file(contract_address)
                
                # Verify deployment
                verify_deployment(w3, contract_address)
                
                return contract_address
            else:
                print(f"Transaction failed. Retrying...")
                
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
    
    return None


def update_env_file(contract_address):
    """Update .env file with contract address"""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    
    with open(env_path, "r") as f:
        lines = f.readlines()
    
    # Check if CONTRACT_ADDRESS already exists
    found = False
    new_lines = []
    for line in lines:
        if line.startswith("CONTRACT_ADDRESS="):
            new_lines.append(f"CONTRACT_ADDRESS={contract_address}\n")
            found = True
        else:
            new_lines.append(line)
    
    if not found:
        new_lines.append(f"\n# Blockchain\nCONTRACT_ADDRESS={contract_address}\n")
    
    with open(env_path, "w") as f:
        f.writelines(new_lines)
    
    print(f"\nUpdated .env with CONTRACT_ADDRESS={contract_address}")


def verify_deployment(w3, contract_address):
    """Verify contract deployment by calling a view function"""
    try:
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=CONTRACT_ABI
        )
        
        owner = contract.functions.owner().call()
        total = contract.functions.totalValidatedClaims().call()
        
        print(f"\nVerification:")
        print(f"  Owner: {owner}")
        print(f"  Total validated claims: {total}")
        print(f"  Status: ✓ Contract is operational")
        
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def test_contract():
    """Test deployed contract with a sample validation"""
    from claimguard.services.blockchain import get_blockchain_service
    
    print("\n" + "="*50)
    print("Testing contract with sample claim validation")
    print("="*50)
    
    try:
        service = get_blockchain_service()
        
        # Test validation
        result = service.validate_claim_on_chain(
            claim_id="test-claim-001",
            score=85.5,
            decision="APPROVED",
            ipfs_hashes=["QmTest123", "QmTest456"],
            patient_id="patient-test-001"
        )
        
        print(f"\n✓ Test validation successful!")
        print(f"  TX Hash: {result['tx_hash']}")
        print(f"  Block: {result['block_number']}")
        
        # Verify retrieval
        proof = service.get_claim_proof("test-claim-001")
        if proof:
            print(f"  Retrieved proof: score={proof['score']}, approved={proof['approved']}")
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    print("="*50)
    print("ClaimValidator Deployment Script")
    print("="*50)
    
    contract_address = deploy_contract()
    
    if contract_address:
        print("\n" + "="*50)
        print("Deployment complete!")
        print("="*50)
        
        # Ask to run test
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            test_contract()
    else:
        print("\nDeployment failed. Check errors above.")
        sys.exit(1)
