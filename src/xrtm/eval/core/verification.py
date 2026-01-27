# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

import json
import logging
from typing import Any, Dict, List, Tuple

# Note: BaseGraphState remains in xrtm-forecast
from xrtm.forecast.core.schemas.graph import BaseGraphState

logger = logging.getLogger(__name__)


class SovereigntyVerifier:
    r"""
    A utility for verifying the cryptographic integrity of .xrtm research proofs.
    """

    @staticmethod
    def verify_manifest(manifest: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []

        # 1. Structural Check
        required = ["version", "subject_id", "final_state_hash", "reasoning_trace"]
        for field in required:
            if field not in manifest:
                errors.append(f"Missing required manifest field: {field}")

        if errors:
            return False, errors

        # 2. Re-compute Merkle Hash of the final state
        try:
            state = BaseGraphState(
                subject_id=manifest["subject_id"],
                node_reports=manifest["reasoning_trace"],
                execution_path=manifest["execution_path"],
                cycle_count=manifest.get("cycle_count", 0),
            )
            state.context = manifest.get("context", {})

            reputed_hash = manifest["final_state_hash"]
            actual_hash = state.compute_hash()

            if reputed_hash != actual_hash:
                errors.append(
                    f"Merkle Integrity Failure: manifest hash {reputed_hash} != calculated hash {actual_hash}"
                )

        except Exception as e:
            errors.append(f"Reconstruction failed: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def verify_file(path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            success, errors = SovereigntyVerifier.verify_manifest(manifest)
            if not success:
                for err in errors:
                    logger.error(f"[VERIFIER] {err}")
            return success
        except Exception as e:
            logger.error(f"[VERIFIER] Failed to read or parse file: {e}")
            return False


__all__ = ["SovereigntyVerifier"]
