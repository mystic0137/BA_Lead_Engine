import hashlib
import sys
from src.config import RAW_DATA, EXPECTED_DATA_HASH

def verify_integrity():
    """Validates the raw dataset against the hard-coded contract in ."""
    if not RAW_DATA.exists():
        print(f"FAILED: Dataset missing at {RAW_DATA}")
        sys.exit(1)

    sha256_hash = hashlib.sha256()
    with open(RAW_DATA, "rb") as f:
        # Memory-efficient chunked read
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    actual_hash = sha256_hash.hexdigest()

    if actual_hash != EXPECTED_DATA_HASH:
        print("Data Integrity Violation!")
        print(f"Expected: {EXPECTED_DATA_HASH}")
        print(f"Actual:   {actual_hash}")
        sys.exit(1) # Hard stop for the pipeline