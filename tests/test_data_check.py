import hashlib
import pytest
from unittest.mock import patch
from src.config import RAW_DATA, EXPECTED_DATA_HASH


def test_verify_integrity_success(tmp_path):
    content = b"test,data\n1,2\n"
    file_path = tmp_path / "customer_booking.csv"
    file_path.write_bytes(content)

    actual_hash = hashlib.sha256(content).hexdigest()

    with patch("src.data_check.RAW_DATA", file_path), \
         patch("src.data_check.EXPECTED_DATA_HASH", actual_hash):
        from src.data_check import verify_integrity
        verify_integrity()


def test_verify_integrity_fails_on_hash_mismatch(tmp_path):
    content = b"test,data\n1,2\n"
    file_path = tmp_path / "customer_booking.csv"
    file_path.write_bytes(content)

    wrong_hash = "0" * 64

    with patch("src.data_check.RAW_DATA", file_path), \
         patch("src.data_check.EXPECTED_DATA_HASH", wrong_hash):
        from src.data_check import verify_integrity
        with pytest.raises(RuntimeError, match="Data integrity mismatch"):
            verify_integrity()


def test_verify_integrity_fails_on_missing_file(tmp_path):
    missing_path = tmp_path / "nonexistent.csv"

    with patch("src.data_check.RAW_DATA", missing_path):
        from src.data_check import verify_integrity
        with pytest.raises(RuntimeError, match="Dataset missing"):
            verify_integrity()


def test_verify_integrity_large_file(tmp_path):
    content = b"A" * (4096 * 3 + 100)
    file_path = tmp_path / "large.csv"
    file_path.write_bytes(content)

    actual_hash = hashlib.sha256(content).hexdigest()

    with patch("src.data_check.RAW_DATA", file_path), \
         patch("src.data_check.EXPECTED_DATA_HASH", actual_hash):
        from src.data_check import verify_integrity
        verify_integrity()
