import sys
import unittest.mock

import pytest


@pytest.fixture
def preserve_sys_modules():
    with unittest.mock.patch.dict(sys.modules):
        yield
