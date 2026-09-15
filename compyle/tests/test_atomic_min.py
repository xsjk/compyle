import ast

import numpy as np
import pytest

from ..array import wrap
from ..low_level import atomic_min
from ..parallel import Elementwise
from ..translator import CUDAConverter
from ..types import annotate


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("atomic_min(self.ready, -1)", "atomicMin(&self->ready, -1)"),
        ("atomic_min(votes[i], value)", "atomicMin(&votes[i], value)"),
    ],
)
def test_atomic_min_accepts_integer_lvalues(expression, expected):
    node = ast.parse(expression, mode="eval").body
    assert CUDAConverter().visit(node) == expected


def test_atomic_min_cuda_updates_shared_value_and_returns_old_value():
    import pycuda.driver as cuda

    cuda.init()
    assert cuda.Device.count() > 0

    @annotate(i="int", previous="gintp", shared="gintp", value="int")
    def vote(i, previous, shared, value):
        previous[i] = atomic_min(shared[0], value)

    n = 65536
    previous, shared = wrap(
        np.empty(n, dtype=np.int32), np.array([7], dtype=np.int32), backend="cuda"
    )
    kernel = Elementwise(vote, backend="cuda")
    kernel(previous, shared, -1)
    previous.pull()
    shared.pull()
    assert shared.data[0] == -1
    assert np.count_nonzero(previous.data == 7) == 1
    assert np.count_nonzero(previous.data == -1) == n - 1

    kernel(previous, shared, 3)
    previous.pull()
    shared.pull()
    assert shared.data[0] == -1
    assert np.all(previous.data == -1)
