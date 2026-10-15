"""External data sources beyond SportsDataverse and ESPN.

    cfb-forecast sources-list
    cfb-forecast sources-collect --season 2026 [--source pickswise ...] [--offline]
    cfb-forecast sources-import  --season 2026 --source predictions_board --file board.pdf

See ``registry.py`` for the sites and ``collection.py`` for the outputs.
"""
from .collection import collect, crosscheck, import_file, list_sources
from .registry import SOURCES

__all__ = ["SOURCES", "collect", "crosscheck", "import_file", "list_sources"]
