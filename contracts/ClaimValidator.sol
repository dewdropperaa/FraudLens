// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract ClaimValidator {
    address public owner;
    address public authorizedValidator;
    uint256 public totalValidatedClaims;

    struct ClaimProof {
        uint256 score;
        bool approved;
        bytes32 documentHash;
        bytes32 zkProofHash;
        uint256 timestamp;
        address validator;
    }

    mapping(bytes32 => ClaimProof) private _claims;
    bytes32[] private _claimIds;

    event ClaimValidated(
        bytes32 indexed claimIdHash,
        uint256 score,
        bool approved,
        uint256 timestamp,
        address validator
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyValidator() {
        require(
            msg.sender == owner || msg.sender == authorizedValidator,
            "Not authorized"
        );
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedValidator = msg.sender;
    }

    function setAuthorizedValidator(address newValidator) external onlyOwner {
        authorizedValidator = newValidator;
    }

    function validateClaim(
        bytes32 claimIdHash,
        uint256 score,
        bool approved
    ) external onlyValidator returns (bool) {
        require(score <= 100, "Score must be 0-100");
        require(_claims[claimIdHash].timestamp == 0, "Claim already validated");

        _claims[claimIdHash] = ClaimProof({
            score: score,
            approved: approved,
            documentHash: bytes32(0),
            zkProofHash: bytes32(0),
            timestamp: block.timestamp,
            validator: msg.sender
        });

        _claimIds.push(claimIdHash);
        totalValidatedClaims++;

        emit ClaimValidated(claimIdHash, score, approved, block.timestamp, msg.sender);
        return true;
    }

    function getClaimProof(bytes32 claimIdHash)
        external
        view
        returns (
            uint256 score,
            bool approved,
            bytes32 documentHash,
            bytes32 zkProofHash,
            uint256 timestamp,
            address validator
        )
    {
        ClaimProof storage p = _claims[claimIdHash];
        return (p.score, p.approved, p.documentHash, p.zkProofHash, p.timestamp, p.validator);
    }

    function isClaimValidated(bytes32 claimIdHash) external view returns (bool) {
        return _claims[claimIdHash].timestamp != 0;
    }

    function getClaimIdByIndex(uint256 index) external view returns (bytes32) {
        require(index < _claimIds.length, "Index out of bounds");
        return _claimIds[index];
    }
}
