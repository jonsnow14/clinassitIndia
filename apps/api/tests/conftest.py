import os

import pytest


def pytest_collection_modifyitems(config, items):
    has_key = bool(os.environ.get("SARVAM_API_KEY"))
    require_chroma = os.environ.get("EVAL_REQUIRE_CHROMA") == "1"
    chroma = require_chroma
    if not chroma:
        try:
            from app.rag.ingest import chroma_ready

            chroma = chroma_ready()
        except Exception:
            chroma = False

    skip_live = pytest.mark.skip(reason="SARVAM_API_KEY not set")
    skip_retrieval = pytest.mark.skip(reason="chroma not ready (set EVAL_REQUIRE_CHROMA=1 to force)")
    for item in items:
        if item.get_closest_marker("live") and not has_key:
            item.add_marker(skip_live)
        if item.get_closest_marker("retrieval") and not chroma:
            item.add_marker(skip_retrieval)
