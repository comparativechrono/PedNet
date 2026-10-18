import re
from pathlib import Path

import pednet


def test_versions_agree():
    pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    if not pyproject.exists():
        return
    version = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.M).group(1)
    assert pednet.__version__ == version
