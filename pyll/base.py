# file is called AST to not collide with std lib module 'ast'
#
# It provides types to build ASTs in a simple lambda-notation style
#

from StringIO import StringIO
import numpy as np # -- for array union

class SymbolTable(object):
    """
    An object whose methods generally allocate Apply nodes.

    _impls is a dictionary containing implementations for those nodes.

    >>> self.add(a, b)          # -- creates a new 'add' Apply node
    >>> self._impl['add'](a, b) # -- this computes a + b
    """

    def __init__(self):
        # -- list and dict are special because they are Python builtins
        self._impls = {
                'pos_args': lambda *x: x,
                'dict': dict,
                'range': range,
                'len': len,
                }

    def _new_apply(self, name, args, kwargs, o_len):
        pos_args = [as_apply(a) for a in args]
        named_args = [(k, as_apply(v)) for (k, v) in kwargs.items()]
        named_args.sort()
        return Apply(name,
                pos_args=pos_args,
                named_args=named_args,
                o_len=o_len)

    def list(self, init):
        return self._new_apply('pos_args', init, {}, o_len=len(init))

    def dict(self, *args, **kwargs):
        # XXX: figure out len
        return self._new_apply('dict', args, kwargs, o_len=None)

    def range(self, *args):
        return self._new_apply('range', args, {}, o_len=None)

    def len(self, obj):
        return self._new_apply('len', [obj], {}, o_len=None)

    def define(self, f, o_len=None):
        """Decorator for adding python functions to self
        """
        name = f.__name__
        if hasattr(self, name):
            raise ValueError('Cannot override existing symbol', name)
        def apply_f(*args, **kwargs):
            return self._new_apply(name, args, kwargs, o_len)
        setattr(self, name, apply_f)
        self._impls[name] = f
        return f

    def define_info(self, o_len):
        def wrapper(f):
            return self.define(f, o_len=o_len)
        return wrapper


scope = SymbolTable()


def as_apply(obj):
    """Smart way of turning object into an Apply
    """
    if isinstance(obj, Apply):
        rval = obj
    elif isinstance(obj, (tuple, list)):
        rval = Apply('pos_args', [as_apply(a) for a in obj], {}, len(obj))
    elif isinstance(obj, dict):
        items = obj.items()
        # -- should be fine to allow numbers and simple things
        #    but think about if it's ok to allow Applys
        #    it messes up sorting at the very least.
        items.sort()
        named_args = [(k, as_apply(v)) for (k, v) in items]
        rval = Apply('dict', [], named_args, len(named_args))
    else:
        rval = Literal(obj)
    assert isinstance(rval, Apply)
    return rval


class Apply(object):
    """
    Represent a symbolic application of a symbol to arguments.
    """

    def __init__(self, name, pos_args, named_args, o_len=None):
        self.name = name
        # -- tuples or arrays -> lists
        self.pos_args = list(pos_args)
        self.named_args = [[kw, arg] for (kw, arg) in named_args]
        # -- o_len is attached this early to support tuple unpacking and
        #    list coersion.
        self.o_len = o_len
        assert all(isinstance(v, Apply) for v in pos_args)
        assert all(isinstance(v, Apply) for k, v in named_args)
        assert all(isinstance(k, basestring) for k, v in named_args)

    def inputs(self):
        rval = self.pos_args + [v for (k, v) in self.named_args]
        assert all(isinstance(arg, Apply) for arg in rval)
        return rval

    def clone_from_inputs(self, inputs, o_len='same'):
        if len(inputs) != len(self.inputs()):
            raise TypeError()
        L = len(self.pos_args)
        pos_args  = list(inputs[:L])
        named_args = [[kw, inputs[L + ii]]
                for ii, (kw, arg) in enumerate(self.named_args)]
        # -- danger cloning with new inputs can change the o_len
        if o_len == 'same':
            o_len = self.o_len
        return self.__class__(self.name, pos_args, named_args, o_len)


    def replace_input(self, old_node, new_node):
        rval = []
        for ii, aa in enumerate(self.pos_args):
            if aa is old_node:
                self.pos_args[ii] = new_node
                rval.append(ii)
        for ii, (nn, aa) in enumerate(self.named_args):
            if aa is old_node:
                self.named_args[ii][1] = new_node
                rval.append(ii + len(self.pos_args))
        return rval

    def pprint(self, ofile, indent=0):
        print >> ofile, ' ' * indent + self.name
        for arg in self.pos_args:
            arg.pprint(ofile, indent+2)
        for name, arg in self.named_args:
            print >> ofile, ' ' * indent + ' ' + name + ' ='
            arg.pprint(ofile, indent+2)

    def __str__(self):
        sio = StringIO()
        self.pprint(sio)
        return sio.getvalue()[:-1] # -- remove trailing '\n'

    def __add__(self, other):
        return scope.add(self, other)

    def __radd__(self, other):
        return scope.add(other, self)

    def __getitem__(self, idx):
        if self.o_len is not None and isinstance(idx, int):
            if idx >= self.o_len:
                #  -- this IndexError is essential for supporting
                #     tuple-unpacking syntax or list coersion of self.
                raise IndexError()
        return scope.getitem(self, idx)

    def __len__(self):
        if self.o_len is None:
            return object.__len__(self)
        return self.o_len


class Literal(Apply):
    def __init__(self, obj):
        try:
            o_len = len(obj)
        except TypeError:
            o_len = None
        Apply.__init__(self, 'literal', [], {}, o_len)
        self._obj = obj

    def pprint(self, ofile, indent=0):
        print >> ofile, ' ' * indent + ('Literal{%s}' % str(self._obj))

    def replace_input(self, old_node, new_node):
        return []

    def clone_from_inputs(self, inputs, o_len='same'):
        return self.__class__(self._obj)


def dfs(aa, seq=None, seqset=None):
    if seq is None:
        assert seqset is None
        seq = []
        seqset = set()
    # -- seqset is the set of all nodes we have seen (which may be still on
    #    the stack)
    if aa in seqset:
        return
    assert isinstance(aa, Apply)
    seqset.add(aa)
    for ii in aa.inputs():
        dfs(ii, seq, seqset)
    seq.append(aa)
    return seq


def clone(expr, memo=None):
    if memo is None:
        memo = {}
    nodes = dfs(expr)
    for node in nodes:
        if node not in memo:
            new_inputs = [memo[arg] for arg in node.inputs()]
            new_node = node.clone_from_inputs(new_inputs)
            memo[node] = new_node
    return memo[expr]


################################################################################
################################################################################



################################################################################
################################################################################


def rec_eval(node, scope=scope):
    """
    Returns nodes required by this one.
    Updates the memo by side effect. Returning [] means this node has been
    computed and the value is available as memo[id(node)]
    """
    memo = {}
    for aa in dfs(node):
        if isinstance(aa, Literal):
            memo[id(aa)] = aa._obj
    todo = [node]
    topnode = node
    while todo:
        if len(todo) > 100000:
            raise RuntimeError('Probably infinite loop in document')
        node = todo.pop()
        if id(node) not in memo:
            waiting_on = [v for v in node.inputs() if id(v) not in memo]
            if waiting_on:
                todo.extend([node] + waiting_on)
            else:
                args = [memo[id(v)] for v in node.pos_args]
                kwargs = dict([(k, memo[id(v)]) for (k, v) in node.named_args])
                memo[id(node)] = rval = scope._impls[node.name](*args, **kwargs)
                if rval is None:
                    raise Exception('really?', (node.name, args, kwargs))
    return memo[id(topnode)]


################################################################################
################################################################################


@scope.define
def getitem(obj, idx):
    return obj[idx]


@scope.define
def identity(obj):
    return obj


@scope.define
def add(a, b):
    return a + b


@scope.define
def array_union(a, b):
    sa = set(a)
    sa.update(b)
    return np.asarray(sorted(sa))


@scope.define
def repeat(n_times, obj):
    return [obj] * n_times

