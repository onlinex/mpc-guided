"""Auto-mark every test in tests/legacy/ with the ``legacy`` marker.

Default pytest invocation skips this whole tree (see pyproject.toml's
``-m 'not legacy'``); run them explicitly with ``pytest -m legacy``.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/legacy/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.legacy)
