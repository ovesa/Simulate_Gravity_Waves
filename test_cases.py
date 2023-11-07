import numpy as np
import pytest

import build_functions


@pytest.mark.parametrize(
    "cs_val, test_kx_val, expected",
    [
        (0.007, 1.0, 1.1140846016432673),
        (0.007, 4.0, 4.456338406573069),
        (0.007, 6.0, 6.684507609859605),
    ],
)
def test_Lamb_equation(cs_val, test_kx_val, expected):
    Lamb_val = build_functions.Lamb_equation(cs_val, test_kx_val) / 2 / np.pi * 1000
    assert Lamb_val == expected

