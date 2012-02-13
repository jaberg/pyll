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


@implicit_stochastic
@scope.define
def uniform(low, high, rng=None, size=()):
    return rng.uniform(low, high)


@implicit_stochastic
@scope.define
def randint(upper, size=(), rng=None):
    return rng.randint(upper, size=size)


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


@implicit_stochastic
@scope.define
def quantized_uniform(low, high, q, rng=None):
    draw = rng.uniform(low, high)
    return np.floor(draw/q) * q


@implicit_stochastic
@scope.define
def log_uniform(low, high, rng=None):
    loglow = np.log(low)
    loghigh = np.log(high)
    draw = rng.uniform(loglow, loghigh)
    return np.exp(draw)


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


def replace_repeat_stochastic(expr):
    stoch = implicit_stochastic_symbols
    nodes = dfs(expr)
    for ii, orig in enumerate(nodes):
        # SEE REPLACE ABOVE AS WELL
        # XXX NOT GOOD! WRITE PATTERNS FOR SUCH THINGS!
        if orig.name == 'idxs_map' and orig.pos_args[1]._obj in stoch:
            idxs = orig.pos_args[0]
            dist = orig.pos_args[1]._obj
            inputs = []
            for arg in orig.inputs()[2:]:
                assert arg.name == 'pos_args'
                assert arg.pos_args[0] is idxs
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
    return expr

