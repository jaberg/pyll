import numpy as np

from pyll import as_apply, vectorize, scope, rec_eval, clone, dfs
from pyll.annotate import replace_implicit_stochastic_nodes
from pyll.annotate import replace_repeat_stochastic


def test_replace_implicit_stochastic_nodes():
    a = scope.uniform(-2, -1)
    rng = np.random.RandomState(234)
    new_a, lrng = replace_implicit_stochastic_nodes(a, rng)
    print new_a
    assert new_a.name == 'getitem'
    assert new_a.pos_args[0].name == 'draw_rng'


def test_replace_implicit_stochastic_nodes_multi():
    uniform = scope.uniform
    a = as_apply([uniform(0, 1), uniform(2, 3)])
    rng = np.random.RandomState(234)
    new_a, lrng = replace_implicit_stochastic_nodes(a, rng)
    print new_a
    val_a = rec_eval(new_a)
    assert np.allclose(val_a, (0.03096734347001351, 2.254282073234248))


def config0():
    p0 = scope.uniform(0, 1)
    p1 = scope.uniform(2, 3)
    p2 = scope.one_of(-1, p0)
    p3 = scope.one_of(-2, p1)
    p4 = 1
    p5 = [3, 4, p0]
    d = locals()
    del d['p1'] # -- don't sample p1 all the time
    s = as_apply(d)
    return s

def test_clone():
    config = config0()
    config2 = clone(config)

    nodeset = set(dfs(config))
    assert not any(n in nodeset for n in dfs(config2))
    
    r = rec_eval(
            replace_implicit_stochastic_nodes(
                config,
                scope.rng_from_seed(5))[0])
    print r
    r2 = rec_eval(
            replace_implicit_stochastic_nodes(
                config2,
                scope.rng_from_seed(5))[0])

    print r2
    assert r == r2


def test_no_redundant_unions():
    config = config0()

    N = as_apply(5)
    vconfig = vectorize(config, N)
    print '=' * 80
    print 'VECTORIZED'
    print vconfig
    print '\n' * 3

    vconfig2 = replace_repeat_stochastic(vconfig)
    print '=' * 80
    print 'VECTORIZED STOCHASTIC'
    print vconfig2
    print '\n' * 3

    new_vc, lrng = replace_implicit_stochastic_nodes(vconfig2,
            scope.rng_from_seed(1))

    print '=' * 80
    print 'VECTORIZED STOCHASTIC WITH RNGS'
    print new_vc

    print rec_eval(new_vc)

