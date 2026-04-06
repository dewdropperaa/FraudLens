"""
Compile and deploy ClaimValidator contract using py-solc-x
"""
import os
import sys
import json
from solcx import compile_source, install_solc, set_solc_version
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv

from claimguard.config import get_sepolia_private_key, get_sepolia_rpc_url

# Load environment
load_dotenv()

# Install and set Solidity version
print("Installing Solidity compiler...")
install_solc('0.8.19')
set_solc_version('0.8.19')

# Read contract source
contract_path = os.path.join(os.path.dirname(__file__), "contracts", "ClaimValidator.sol")
with open(contract_path, 'r') as f:
    contract_source = f.read()

print("Compiling contract...")
# Compile contract
compiled = compile_source(
    contract_source,
    output_values=['abi', 'bin'],
    solc_version='0.8.19'
)

# Extract contract data
contract_id, contract_interface = next(iter(compiled.items()))
bytecode = contract_interface['bin']
abi = contract_interface['abi']

# Save compiled artifacts
build_dir = os.path.join(os.path.dirname(__file__), "build")
os.makedirs(build_dir, exist_ok=True)

with open(os.path.join(build_dir, "ClaimValidator.bin"), 'w') as f:
    f.write(bytecode)

with open(os.path.join(build_dir, "ClaimValidator.abi"), 'w') as f:
    json.dump(abi, f, indent=2)

print(f"Contract compiled successfully!")
print(f"  Bytecode saved to: build/ClaimValidator.bin")
print(f"  ABI saved to: build/ClaimValidator.abi")

# Deploy if private key is configured
private_key = get_sepolia_private_key()
rpc_url = get_sepolia_rpc_url()

if private_key:
    print("\n" + "="*50)
    print("Deploying to Sepolia testnet...")
    print("="*50)
    
    # Connect to network
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    if not w3.is_connected():
        print(f"ERROR: Cannot connect to {rpc_url}")
        sys.exit(1)
    
    print(f"Connected to Sepolia (Chain ID: {w3.eth.chain_id})")
    
    # Setup account
    account = Account.from_key(private_key)
    print(f"Deployer: {account.address}")
    
    # Check balance
    balance = w3.eth.get_balance(account.address)
    print(f"Balance: {w3.from_wei(balance, 'ether')} ETH")
    
    if balance == 0:
        print("\nERROR: No ETH for gas!")
        print("Get Sepolia ETH from:")
        print("  https://sepoliafaucet.com/")
        print("  https://www.alchemy.com/faucets/ethereum-sepolia")
        sys.exit(1)
    
    # Build transaction
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    nonce = w3.eth.get_transaction_count(account.address)
    gas_price = int(w3.eth.gas_price * 1.1)
    
    tx = contract.constructor().build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gasPrice': gas_price,
    })
    
    # Estimate gas
    tx['gas'] = int(w3.eth.estimate_gas(tx) * 1.2)
    
    print(f"Gas estimate: {tx['gas']}")
    print(f"Gas price: {w3.from_wei(gas_price, 'gwei')} gwei")
    
    # Sign and send
    signed = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    tx_hash_hex = w3.to_hex(tx_hash)
    
    print(f"\nTransaction sent: {tx_hash_hex}")
    print(f"Explorer: https://sepolia.etherscan.io/tx/{tx_hash_hex}")
    
    # Wait for confirmation
    print("\nWaiting for confirmation...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    
    if receipt['status'] == 1:
        contract_address = receipt['contractAddress']
        print(f"\n✓ Contract deployed successfully!")
        print(f"  Address: {contract_address}")
        print(f"  Block: {receipt['blockNumber']}")
        print(f"  Gas used: {receipt['gasUsed']}")
        
        # Update .env
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        found = False
        for line in lines:
            if line.startswith("CONTRACT_ADDRESS="):
                new_lines.append(f"CONTRACT_ADDRESS={contract_address}\n")
                found = True
            else:
                new_lines.append(line)
        
        if not found:
            new_lines.append(f"CONTRACT_ADDRESS={contract_address}\n")
        
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        
        print(f"\n✓ Updated .env with CONTRACT_ADDRESS={contract_address}")
        
        # Verify
        deployed = w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )
        
        owner = deployed.functions.owner().call()
        print(f"\nVerification:")
        print(f"  Owner: {owner}")
        print(f"  Total claims: {deployed.functions.totalValidatedClaims().call()}")
        print(f"\n✓ Contract is operational!")
    else:
        print("\nERROR: Transaction failed!")
        sys.exit(1)
else:
    print("\nTo deploy, set SEPOLIA_PRIVATE_KEY or PRIVATE_KEY in .env file")
    print("Get testnet ETH from: https://sepoliafaucet.com/")
