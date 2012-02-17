import numpy as np
from pyll import scope, as_apply, dfs
from pyll.stochastic import *


def test_replace_repeat_stochastic():
    # just test that it runs
    # test_sample is a more demanding test
    rng = np.random.RandomState(955)
    aa = as_apply(dict(
                u = scope.uniform(0, 1),
                n = scope.normal(5, 0.1),
                l = [0, 1, scope.one_of(2, 3)]))
    foo, new_rng = replace_implicit_stochastic_nodes(aa, as_apply(rng))
    print foo
    dfs(foo)


def test_sample_deterministic():
    aa = as_apply([0, 1])
    print aa
    dd = sample(aa, np.random.RandomState(3))
    assert dd == (0, 1)


def test_repeatable():
    u = scope.uniform(0, 1)
    aa = as_apply(dict(
                u = u,
                n = scope.normal(5, 0.1),
                l = [0, 1, scope.one_of(2, 3), u]))
    dd1 = sample(aa, np.random.RandomState(3))
    dd2 = sample(aa, np.random.RandomState(3))
    dd3 = sample(aa, np.random.RandomState(4))
    assert dd1 == dd2
    assert dd1 != dd3


def test_sample():
    u = scope.uniform(0, 1)
    aa = as_apply(dict(
                u = u,
                n = scope.normal(5, 0.1),
                l = [0, 1, scope.one_of(2, 3), u]))
    print aa
    dd = sample(aa, np.random.RandomState(3))
    assert 0 < dd['u'] < 1
    assert 4 < dd['n'] < 6
    assert dd['u'] == dd['l'][3]
    assert dd['l'][:2] == (0, 1)
    assert dd['l'][2] in (2, 3)

