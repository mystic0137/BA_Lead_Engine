import hashlib
import logging
from src.config import RAW_DATA, EXPECTED_DATA_HASH

logger = logging.getLogger(__name__)


class DataIntegrityError(RuntimeError):
    pass


def verify_integrity():
    if not RAW_DATA.exists():
        raise DataIntegrityError(f"Dataset missing at {RAW_DATA}")

    sha256_hash = hashlib.sha256()
    with open(RAW_DATA, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    actual_hash = sha256_hash.hexdigest()

    if actual_hash != EXPECTED_DATA_HASH:
        raise DataIntegrityError(
            f"Data integrity mismatch: expected {EXPECTED_DATA_HASH}, got {actual_hash}"
        )
