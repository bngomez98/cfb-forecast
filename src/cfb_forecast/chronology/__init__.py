"""Append-only chronological prospective forecasting ledger.

Unlike a batch forecast that scores every game from one shared as-of state, this
workflow only forecasts games whose configured pregame cutoff has actually
arrived. Each update is frozen as an immutable holdout run with the exact
model, data snapshot, market snapshot and score distribution used at that
moment.

The implementation is split across submodules by concern:

- ``_shared``: ledger schema and small ledger-file helpers.
- ``manifest``: on-disk manifest creation and compatibility checks.
- ``select``: picking which games are due to be forecast.
- ``update``: appending a new run of due games to the ledger.
- ``bootstrap``: starting a ledger from an already-frozen batch.
- ``settle``: verifying the ledger and grading it once results arrive.

External callers import the public functions from this package directly;
nothing in the import surface changed when the module was split.
"""

from __future__ import annotations

from ._shared import LEDGER_COLUMNS
from .bootstrap import bootstrap_from_frozen_batch
from .select import select_due_games
from .settle import settle_chronological_ledger, verify_chronological_ledger
from .update import run_chronological_update

__all__ = [
    "LEDGER_COLUMNS",
    "select_due_games",
    "run_chronological_update",
    "bootstrap_from_frozen_batch",
    "settle_chronological_ledger",
    "verify_chronological_ledger",
]
