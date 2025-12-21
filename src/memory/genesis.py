"""
Agent OS Memory Vault Genesis Proof System

Implements cryptographic proofs for vault creation and integrity:
- Genesis record (vault creation proof)
- Integrity verification
- Chain of custody
- Audit proofs
"""

import hashlib
import secrets
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from .index import VaultIndex


logger = logging.getLogger(__name__)


@dataclass
class GenesisRecord:
    """
    Genesis record proving vault creation.

    This is the root of trust for the memory vault - it proves
    when and how the vault was created.
    """
    proof_id: str
    created_at: datetime
    created_by: str
    vault_id: str
    initial_hash: str  # Hash of initial state
    hardware_bound: bool
    encryption_profiles: List[str]
    constitution_hash: Optional[str]  # Hash of constitution at creation
    nonce: str
    signature: str  # Self-signed proof

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "vault_id": self.vault_id,
            "initial_hash": self.initial_hash,
            "hardware_bound": self.hardware_bound,
            "encryption_profiles": self.encryption_profiles,
            "constitution_hash": self.constitution_hash,
            "nonce": self.nonce,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenesisRecord":
        return cls(
            proof_id=data["proof_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            vault_id=data["vault_id"],
            initial_hash=data["initial_hash"],
            hardware_bound=data["hardware_bound"],
            encryption_profiles=data["encryption_profiles"],
            constitution_hash=data.get("constitution_hash"),
            nonce=data["nonce"],
            signature=data["signature"],
        )


@dataclass
class IntegrityProof:
    """
    Proof of vault integrity at a point in time.

    Used to verify the vault hasn't been tampered with.
    """
    proof_id: str
    created_at: datetime
    genesis_proof_id: str  # Reference to genesis
    blob_count: int
    state_hash: str  # Merkle root of all blob hashes
    consent_count: int
    consent_hash: str  # Hash of active consents
    previous_proof_id: Optional[str]  # Chain to previous proof
    nonce: str
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "created_at": self.created_at.isoformat(),
            "genesis_proof_id": self.genesis_proof_id,
            "blob_count": self.blob_count,
            "state_hash": self.state_hash,
            "consent_count": self.consent_count,
            "consent_hash": self.consent_hash,
            "previous_proof_id": self.previous_proof_id,
            "nonce": self.nonce,
            "signature": self.signature,
        }


class GenesisProofSystem:
    """
    Manages genesis proofs and integrity verification for the vault.

    Provides:
    - Genesis record creation
    - Integrity proof generation
    - Proof verification
    - Chain of custody
    """

    def __init__(
        self,
        index: VaultIndex,
        proof_dir: Optional[Path] = None,
    ):
        """
        Initialize genesis proof system.

        Args:
            index: Vault index instance
            proof_dir: Directory for proof storage
        """
        self._index = index
        self._proof_dir = proof_dir

        self._genesis_record: Optional[GenesisRecord] = None
        self._latest_integrity_proof: Optional[IntegrityProof] = None
        self._proof_chain: List[str] = []

        if proof_dir:
            self._load_genesis()

    def create_genesis(
        self,
        vault_id: str,
        created_by: str,
        encryption_profiles: List[str],
        hardware_bound: bool = False,
        constitution_path: Optional[Path] = None,
    ) -> GenesisRecord:
        """
        Create the genesis record for a new vault.

        Args:
            vault_id: Unique vault identifier
            created_by: Who created the vault
            encryption_profiles: Available encryption profiles
            hardware_bound: Whether vault uses hardware binding
            constitution_path: Path to constitution file

        Returns:
            Created GenesisRecord
        """
        if self._genesis_record:
            raise RuntimeError("Genesis record already exists")

        proof_id = f"genesis_{secrets.token_hex(16)}"
        nonce = secrets.token_hex(32)

        # Calculate constitution hash if provided
        constitution_hash = None
        if constitution_path and constitution_path.exists():
            content = constitution_path.read_bytes()
            constitution_hash = hashlib.sha256(content).hexdigest()

        # Calculate initial state hash
        initial_hash = self._calculate_initial_hash(
            vault_id, created_by, encryption_profiles
        )

        # Create proof data for signing
        proof_data = {
            "proof_id": proof_id,
            "vault_id": vault_id,
            "created_by": created_by,
            "initial_hash": initial_hash,
            "hardware_bound": hardware_bound,
            "encryption_profiles": encryption_profiles,
            "constitution_hash": constitution_hash,
            "nonce": nonce,
        }

        # Self-sign (in production would use hardware key)
        signature = self._sign_proof(proof_data)

        record = GenesisRecord(
            proof_id=proof_id,
            created_at=datetime.now(),
            created_by=created_by,
            vault_id=vault_id,
            initial_hash=initial_hash,
            hardware_bound=hardware_bound,
            encryption_profiles=encryption_profiles,
            constitution_hash=constitution_hash,
            nonce=nonce,
            signature=signature,
        )

        # Store in index
        self._index.record_genesis_proof(
            proof_id=proof_id,
            proof_type="genesis",
            proof_data=record.to_dict(),
        )

        # Store to file if configured
        if self._proof_dir:
            self._save_genesis(record)

        self._genesis_record = record
        self._proof_chain.append(proof_id)

        logger.info(f"Created genesis record: {proof_id}")
        return record

    def create_integrity_proof(self) -> IntegrityProof:
        """
        Create an integrity proof for current vault state.

        Returns:
            Created IntegrityProof
        """
        if not self._genesis_record:
            raise RuntimeError("Genesis record not found")

        proof_id = f"integrity_{secrets.token_hex(16)}"
        nonce = secrets.token_hex(32)

        # Get current state
        stats = self._index.get_statistics()
        blob_count = stats.get("total_blobs", 0)
        consent_count = stats.get("active_consents", 0)

        # Calculate state hash (Merkle root of blob hashes)
        state_hash = self._calculate_state_hash()

        # Calculate consent hash
        consent_hash = self._calculate_consent_hash()

        # Create proof data
        # First integrity proof links to genesis; subsequent proofs link to previous integrity proof
        previous_proof_id = (
            self._latest_integrity_proof.proof_id
            if self._latest_integrity_proof
            else self._genesis_record.proof_id
        )

        proof_data = {
            "proof_id": proof_id,
            "genesis_proof_id": self._genesis_record.proof_id,
            "blob_count": blob_count,
            "state_hash": state_hash,
            "consent_count": consent_count,
            "consent_hash": consent_hash,
            "previous_proof_id": previous_proof_id,
            "nonce": nonce,
        }

        signature = self._sign_proof(proof_data)

        proof = IntegrityProof(
            proof_id=proof_id,
            created_at=datetime.now(),
            genesis_proof_id=self._genesis_record.proof_id,
            blob_count=blob_count,
            state_hash=state_hash,
            consent_count=consent_count,
            consent_hash=consent_hash,
            previous_proof_id=previous_proof_id,
            nonce=nonce,
            signature=signature,
        )

        # Store in index
        self._index.record_genesis_proof(
            proof_id=proof_id,
            proof_type="integrity",
            proof_data=proof.to_dict(),
        )

        self._latest_integrity_proof = proof
        self._proof_chain.append(proof_id)

        logger.info(f"Created integrity proof: {proof_id}")
        return proof

    def verify_genesis(self) -> Tuple[bool, str]:
        """
        Verify the genesis record.

        Returns:
            Tuple of (is_valid, message)
        """
        if not self._genesis_record:
            return False, "No genesis record found"

        # Recalculate expected hash
        expected_hash = self._calculate_initial_hash(
            self._genesis_record.vault_id,
            self._genesis_record.created_by,
            self._genesis_record.encryption_profiles,
        )

        if expected_hash != self._genesis_record.initial_hash:
            return False, "Initial hash mismatch"

        # Verify signature
        proof_data = {
            "proof_id": self._genesis_record.proof_id,
            "vault_id": self._genesis_record.vault_id,
            "created_by": self._genesis_record.created_by,
            "initial_hash": self._genesis_record.initial_hash,
            "hardware_bound": self._genesis_record.hardware_bound,
            "encryption_profiles": self._genesis_record.encryption_profiles,
            "constitution_hash": self._genesis_record.constitution_hash,
            "nonce": self._genesis_record.nonce,
        }

        if not self._verify_signature(proof_data, self._genesis_record.signature):
            return False, "Invalid signature"

        # Mark as verified in index
        self._index.verify_genesis_proof(self._genesis_record.proof_id)

        return True, "Genesis record verified"

    def verify_integrity(self) -> Tuple[bool, str]:
        """
        Verify current vault integrity against latest proof.

        Returns:
            Tuple of (is_valid, message)
        """
        if not self._latest_integrity_proof:
            return False, "No integrity proof found"

        # Recalculate current state hash
        current_state_hash = self._calculate_state_hash()

        if current_state_hash != self._latest_integrity_proof.state_hash:
            return False, f"State hash mismatch: {current_state_hash} vs {self._latest_integrity_proof.state_hash}"

        # Verify signature
        proof_data = {
            "proof_id": self._latest_integrity_proof.proof_id,
            "genesis_proof_id": self._latest_integrity_proof.genesis_proof_id,
            "blob_count": self._latest_integrity_proof.blob_count,
            "state_hash": self._latest_integrity_proof.state_hash,
            "consent_count": self._latest_integrity_proof.consent_count,
            "consent_hash": self._latest_integrity_proof.consent_hash,
            "previous_proof_id": self._latest_integrity_proof.previous_proof_id,
            "nonce": self._latest_integrity_proof.nonce,
        }

        if not self._verify_signature(proof_data, self._latest_integrity_proof.signature):
            return False, "Invalid signature"

        return True, "Integrity verified"

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify the entire proof chain.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Verify genesis
        is_valid, message = self.verify_genesis()
        if not is_valid:
            issues.append(f"Genesis: {message}")

        # Verify chain links
        previous_id = self._genesis_record.proof_id if self._genesis_record else None

        for proof_id in self._proof_chain[1:]:  # Skip genesis
            proof_data = self._index.get_genesis_proof(proof_id)
            if not proof_data:
                issues.append(f"Missing proof: {proof_id}")
                continue

            if proof_data["proof_type"] == "integrity":
                if proof_data["proof_data"].get("previous_proof_id") != previous_id:
                    issues.append(f"Chain break at: {proof_id}")

            previous_id = proof_id

        return len(issues) == 0, issues

    def get_genesis(self) -> Optional[GenesisRecord]:
        """Get the genesis record."""
        return self._genesis_record

    def get_latest_integrity_proof(self) -> Optional[IntegrityProof]:
        """Get the latest integrity proof."""
        return self._latest_integrity_proof

    def get_proof_chain(self) -> List[str]:
        """Get the proof chain."""
        return self._proof_chain.copy()

    def export_proofs(self) -> Dict[str, Any]:
        """
        Export all proofs for external verification.

        Returns:
            Dict with all proof data
        """
        return {
            "genesis": self._genesis_record.to_dict() if self._genesis_record else None,
            "latest_integrity": (
                self._latest_integrity_proof.to_dict()
                if self._latest_integrity_proof else None
            ),
            "chain": self._proof_chain,
            "exported_at": datetime.now().isoformat(),
        }

    def _calculate_initial_hash(
        self,
        vault_id: str,
        created_by: str,
        encryption_profiles: List[str],
    ) -> str:
        """Calculate hash of initial vault state."""
        data = json.dumps({
            "vault_id": vault_id,
            "created_by": created_by,
            "encryption_profiles": sorted(encryption_profiles),
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _calculate_state_hash(self) -> str:
        """Calculate Merkle root of current vault state."""
        # Get all blob hashes
        blobs = self._index.query_blobs(limit=100000)
        hashes = sorted([b.content_hash for b in blobs])

        if not hashes:
            return hashlib.sha256(b"empty").hexdigest()

        # Simple Merkle tree
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last if odd

            new_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_level.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = new_level

        return hashes[0]

    def _calculate_consent_hash(self) -> str:
        """Calculate hash of active consents."""
        consents = self._index.get_active_consents()
        consent_ids = sorted([c.consent_id for c in consents])

        data = json.dumps(consent_ids)
        return hashlib.sha256(data.encode()).hexdigest()

    def _sign_proof(self, proof_data: Dict[str, Any]) -> str:
        """
        Sign proof data.

        In production, this would use hardware key or asymmetric crypto.
        For now, we use HMAC with a derived key.
        """
        import hmac
        data = json.dumps(proof_data, sort_keys=True).encode()
        # In production: use hardware-bound key
        key = b"vault_proof_key"  # Would be derived from secure key store
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def _verify_signature(self, proof_data: Dict[str, Any], signature: str) -> bool:
        """Verify a proof signature."""
        expected = self._sign_proof(proof_data)
        return hmac.compare_digest(expected, signature)

    def _load_genesis(self) -> None:
        """Load genesis from disk."""
        if not self._proof_dir:
            return

        genesis_path = self._proof_dir / "genesis.json"
        if not genesis_path.exists():
            return

        try:
            with open(genesis_path, 'r') as f:
                data = json.load(f)
            self._genesis_record = GenesisRecord.from_dict(data)
            self._proof_chain.append(self._genesis_record.proof_id)
            logger.info(f"Loaded genesis record: {self._genesis_record.proof_id}")
        except Exception as e:
            logger.error(f"Failed to load genesis: {e}")

    def _save_genesis(self, record: GenesisRecord) -> None:
        """Save genesis to disk."""
        if not self._proof_dir:
            return

        self._proof_dir.mkdir(parents=True, exist_ok=True)
        genesis_path = self._proof_dir / "genesis.json"

        with open(genesis_path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)


# Import hmac for signature verification
import hmac
