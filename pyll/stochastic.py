"""
Constructs for annotating base graphs.
"""
import sys
import numpy as np

from .base import scope, as_apply, dfs, Apply

################################################################################
################################################################################
def ERR(msg):
    print >> sys.stderr, msg


implicit_stochastic_symbols = set()


def implicit_stochastic(f):
    implicit_stochastic_symbols.add(f.__name__)
    return f


@scope.define_info(o_len=2)
def draw_rng(rng, f_name, *args, **kwargs):
    draw = scope._impls[f_name](*args, rng=rng, **kwargs)
    return draw, rng


@scope.define
def rng_from_seed(seed):
    return np.random.RandomState(seed)


# -- UNIFORM

@implicit_stochastic
@scope.define
def uniform(low, high, rng=None, size=()):
    return rng.uniform(low, high, size=size)


@implicit_stochastic
@scope.define
def loguniform(low, high, rng=None, size=()):
    draw = rng.uniform(low, high, size=size)
    return np.exp(draw)


@implicit_stochastic
@scope.define
def quniform(low, high, q, rng=None, size=()):
    draw = rng.uniform(low, high, size=size)
    return np.floor(draw/q) * q


@implicit_stochastic
@scope.define
def qloguniform(low, high, q, rng=None, size=()):
    draw = np.exp(rng.uniform(low, high, size=size))
    return np.floor(draw/q) * q


# -- NORMAL

@implicit_stochastic
@scope.define
def normal(mu, sigma, rng=None, size=()):
    return rng.normal(mu, sigma, size=size)


@implicit_stochastic
@scope.define
def qnormal(mu, sigma, q, rng=None, size=()):
    draw = rng.normal(mu, sigma, size=size)
    return np.floor(draw/q) * q


@implicit_stochastic
@scope.define
def lognormal(mu, sigma, rng=None, size=()):
    draw = rng.normal(mu, sigma, size=size)
    return np.exp(draw)


@implicit_stochastic
@scope.define
def qlognormal(mu, sigma, q, rng=None, size=()):
    draw = np.exp(rng.normal(mu, sigma, size=size))
    return np.floor(draw/q) * q


# -- CATEGORICAL


@implicit_stochastic
@scope.define
def randint(upper, rng=None, size=()):
    # this is tricky because numpy doesn't support
    # upper being a list of len size[0]
    if isinstance(upper, (list, tuple)):
        if isinstance(size, int):
            assert len(upper) == size
            return np.asarray([rng.randint(uu) for uu in upper])
        elif len(size) == 1:
            assert len(upper) == size[0]
            return np.asarray([rng.randint(uu) for uu in upper])
    return rng.randint(upper, size=size)


@implicit_stochastic
@scope.define
def categorical(p, rng=None, size=()):
    """Draws i with probability p[i]"""
    #XXX: OMG this is the craziest shit
    n_draws = numpy.prod(size)
    sample = rng.multinomial(n=1, pvals=p, size=tuple(size))
    assert sample.shape == tuple(shp) + (len(p),)
    if tuple(shp):
        rval = numpy.sum(sample * numpy.arange(len(p)), axis=len(shp))
    else:
        rval = [numpy.where(rng.multinomial(pvals=p, n=1))[0][0]
                for i in xrange(n_draws)]
        rval = numpy.asarray(rval, dtype=self.otype.dtype)
    assert (rval.shape == shp).all()
    return rval


@implicit_stochastic
@scope.define
def choice(args, rng=None):
    ii = rng.randint(len(args))
    return args[ii]


@implicit_stochastic
@scope.define
def one_of(*args, **kwargs):
    rng = kwargs.pop('rng', None)
    size = kwargs.pop('size', ())
    assert not kwargs # -- we should have got everything by now
    ii = rng.randint(len(args))
    return args[ii]


def replace_implicit_stochastic_nodes(expr, rng, scope=scope):
    """
    Make all of the stochastic nodes in expr use the rng

    uniform(0, 1) -> getitem(draw_rng(rng, 'uniform', 0, 1), 1)
    """
    lrng = as_apply(rng)
    nodes = dfs(expr)
    for ii, orig in enumerate(nodes):
        if orig.name in implicit_stochastic_symbols:
            obj = scope.draw_rng(lrng, orig.name)
            obj.pos_args += orig.pos_args
            obj.named_args += orig.named_args
            draw, new_lrng = obj
            # -- loop over all nodes that *use* this one, and change them
            for client in nodes[ii+1:]:
                client.replace_input(orig, draw)
            if expr is orig:
                expr = draw
            lrng = new_lrng
    return expr, new_lrng


def replace_repeat_stochastic(expr, return_memo=False):
    stoch = implicit_stochastic_symbols
    nodes = dfs(expr)
    memo = {}
    for ii, orig in enumerate(nodes):
        # SEE REPLACE ABOVE AS WELL
        # XXX NOT GOOD! WRITE PATTERNS FOR SUCH THINGS!
        if orig.name == 'idxs_map' and orig.pos_args[1]._obj in stoch:
            idxs = orig.pos_args[0]
            dist = orig.pos_args[1]._obj
            inputs = []
            for arg in orig.inputs()[2:]:
                assert arg.name == 'pos_args'
                assert arg.pos_args[0] is idxs, str(orig)
                inputs.append(arg.pos_args[1])
            if orig.named_args:
                raise NotImplementedError('')
            vnode = Apply(dist, inputs, orig.named_args, None)
            n_times = scope.len(idxs)
            vnode.named_args.append(['size', n_times])
            # -- loop over all nodes that *use* this one, and change them
            for client in nodes[ii+1:]:
                client.replace_input(orig, vnode)
            if expr is orig:
                expr = vnode
            memo[orig] = vnode
    if return_memo:
        return expr, memo
    else:
        return expr

