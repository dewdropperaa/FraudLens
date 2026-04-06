// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ClaimValidator
 * @dev Smart contract for storing claim validation proofs on blockchain
 * @notice Only stores hashes - NEVER stores personal or sensitive data
 */
contract ClaimValidator {
    // Struct to store claim validation proof
    struct ClaimProof {
        bytes32 claimIdHash;      // Hash of claim ID (not the actual ID)
        uint256 score;            // Validation score (0-100)
        bool approved;            // Decision: true = APPROVED, false = REJECTED
        bytes32 documentHash;     // Hash of IPFS document references
        bytes32 zkProofHash;      // Simulated ZK-proof hash
        uint256 timestamp;        // Block timestamp
        address validator;        // Address that submitted the validation
    }

    // Mapping from claim ID hash to validation proof
    mapping(bytes32 => ClaimProof) public claimProofs;
    
    // Array of all claim ID hashes for enumeration
    bytes32[] public claimIds;
    
    // Counter for total validated claims
    uint256 public totalValidatedClaims;
    
    // Address of the authorized backend
    address public owner;
    address public authorizedValidator;
    
    // Events for audit trail
    event ClaimValidated(
        bytes32 indexed claimIdHash,
        uint256 score,
        bool approved,
        uint256 timestamp,
        address validator
    );
    
    event ValidatorUpdated(address indexed oldValidator, address indexed newValidator);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyAuthorized() {
        require(
            msg.sender == owner || msg.sender == authorizedValidator,
            "Not authorized to validate claims"
        );
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedValidator = msg.sender;
    }

    /**
     * @dev Validate a claim and store proof on blockchain
     * @param claimIdHash Hash of the claim ID (keccak256)
     * @param score Validation score (0-100)
     * @param approved Decision: true = APPROVED, false = REJECTED
     */
    function validateClaim(
        bytes32 claimIdHash,
        uint256 score,
        bool approved
    ) external onlyAuthorized returns (bool) {
        // Validate score range
        require(score <= 100, "Score must be between 0 and 100");
        
        // Check if claim already validated
        require(claimProofs[claimIdHash].timestamp == 0, "Claim already validated");
        
        // Store the validation proof
        claimProofs[claimIdHash] = ClaimProof({
            claimIdHash: claimIdHash,
            score: score,
            approved: approved,
            documentHash: bytes32(0),
            zkProofHash: bytes32(0),
            timestamp: block.timestamp,
            validator: msg.sender
        });
        
        // Add to enumeration
        claimIds.push(claimIdHash);
        totalValidatedClaims++;
        
        // Emit event for audit trail
        emit ClaimValidated(
            claimIdHash,
            score,
            approved,
            block.timestamp,
            msg.sender
        );
        
        return true;
    }

    /**
     * @dev Get validation proof for a claim
     * @param claimIdHash Hash of the claim ID
     */
    function getClaimProof(bytes32 claimIdHash) external view returns (
        uint256 score,
        bool approved,
        bytes32 documentHash,
        bytes32 zkProofHash,
        uint256 timestamp,
        address validator
    ) {
        ClaimProof storage proof = claimProofs[claimIdHash];
        require(proof.timestamp > 0, "Claim not found");
        
        return (
            proof.score,
            proof.approved,
            proof.documentHash,
            proof.zkProofHash,
            proof.timestamp,
            proof.validator
        );
    }

    /**
     * @dev Verify if a claim has been validated
     * @param claimIdHash Hash of the claim ID
     */
    function isClaimValidated(bytes32 claimIdHash) external view returns (bool) {
        return claimProofs[claimIdHash].timestamp > 0;
    }

    /**
     * @dev Update authorized validator address
     * @param newValidator Address of the new validator
     */
    function setAuthorizedValidator(address newValidator) external onlyOwner {
        require(newValidator != address(0), "Invalid address");
        
        address oldValidator = authorizedValidator;
        authorizedValidator = newValidator;
        
        emit ValidatorUpdated(oldValidator, newValidator);
    }

    /**
     * @dev Transfer ownership
     * @param newOwner Address of the new owner
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }

    /**
     * @dev Get total number of validated claims
     */
    function getTotalValidatedClaims() external view returns (uint256) {
        return totalValidatedClaims;
    }

    /**
     * @dev Get claim ID hash by index
     * @param index Index in the claimIds array
     */
    function getClaimIdByIndex(uint256 index) external view returns (bytes32) {
        require(index < claimIds.length, "Index out of bounds");
        return claimIds[index];
    }
}
