"""
Constructs for annotating base graphs.
"""

from .base import scope

################################################################################
################################################################################


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
def uniform(low, high, rng=None):
    return rng.uniform(low, high)


@implicit_stochastic
@scope.define
def choice(args, rng=None):
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


class VectorizeHelper(object):
    """
    Example:
        u0 = uniform(1, 2)
        u1 = uniform(2, 3)
        c = one_of(u0, u1)
        expr = {'u1': u1, 'c': c}
    becomes
        N
        expr_idxs = range(N)
        choices = randint(2, len(expr_idxs))
        c0_idxs, c1_idxs = vchoice_split(expr_idxs, choices)
        c0_vals = vdraw(len(c0_idxs), 'uniform', 1, 2)
        c1_vals = vdraw(len(c1_idxs), 'uniform', 2, 3)
    """
    def __init__(self, expr, expr_idxs):
        self.expr = expr
        self.expr_idxs = expr_idxs
        self.idxs_memo = {expr: expr_idxs}
        self.vals_memo = {}
        self.dfs_nodes = dfs(expr)

    def merge(self, idxs, node):
        if node in self.idxs_memo:
            self.idxs_memo[node] = scope.array_union(idxs, self.idxs_memo[node])
        else:
            self.idxs_memo[node] = idxs
        
    # -- separate method for testing
    def build_idxs(self):
        for node in reversed(self.dfs_nodes):
            node_idxs = self.idxs_memo[node]
            if node.name == 'one_of':
                n_options  = len(node.pos_args)
                choices = scope.randint(n_options, size=len(node_idxs))
                self.vals_memo[node] = choices
                sub_idxs = scope.vchoice_split(node_idxs, choices)
                for ii, arg in enumerate(node.pos_args):
                    self.merge(sub_idxs[ii], arg)
            else:
                for arg in node.inputs():
                    self.merge(node_idxs, arg)

    # -- separate method for testing
    def build_vals(self):
        for node in self.dfs_nodes:
            if node not in self.vals_memo:
                n_times = scope.len(self.idxs_memo[node])
                vnode = scope.repeat(n_times, node.name)
                vnode.pos_args.extend(node.pos_args)
                vnode.named_args.extend(node.named_args)
                for arg in node.inputs:
                    vnode.replace(arg, self.vals_memo[arg])
                vnode.replace
                self.vals_memo[node] = vnode


def vectorize(expr, N):
    """
    Returns:
       0. a list of the nodes in expr.
       1. a list of N evaluations of expr
       2. a list of (node, idxs, vals) for the stochastic elements of expr
    """
    expr_idxs = scope.range(N)
    vh = VectorizeHelper(expr, expr_idxs)
    vh.build_idxs()
    vh.build_vals()


