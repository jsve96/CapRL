from pathlib import Path

import pytest

from caprl.agents.checkpointing import ValidationCheckpoint, select_validation_checkpoint


def test_checkpoint_selection_excludes_mandate_failures_and_uses_turnover_tie_breaker():
    checkpoints = [
        ValidationCheckpoint(Path("invalid.pt"), 0.20, 0.1, False),
        ValidationCheckpoint(Path("high-turnover.pt"), 0.10, 2.0, True),
        ValidationCheckpoint(Path("selected.pt"), 0.10, 1.0, True),
    ]
    assert select_validation_checkpoint(checkpoints).path == Path("selected.pt")


def test_checkpoint_selection_rejects_empty_eligible_set():
    with pytest.raises(ValueError, match="no checkpoint"):
        select_validation_checkpoint([ValidationCheckpoint(Path("x"), 1.0, 0.0, False)])

