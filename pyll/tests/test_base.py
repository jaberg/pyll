from pyll.base import *
import numpy as np

def test_literal_pprint():
    l = Literal(5)
    print str(l)
    assert str(l) == 'Literal{5}'


def test_literal_apply():
    l0 = Literal([1, 2, 3])
    print str(l0)
    assert str(l0) == 'Literal{[1, 2, 3]}'


def test_literal_unpacking():
    l0 = Literal([1, 2, 3])
    a, b, c = l0
    print a
    assert c.name == 'getitem'
    assert c.pos_args[0] is l0
    assert isinstance(c.pos_args[1], Literal)
    assert c.pos_args[1]._obj == 2


def test_as_apply_passthrough():
    a4 = as_apply(4)
    assert a4 is as_apply(a4)


def test_as_apply_literal():
    assert isinstance(as_apply(7), Literal)


def test_as_apply_list_of_literals():
    l = [9, 3]
    al = as_apply(l)
    assert isinstance(al, Apply)
    assert al.name == 'pos_args'
    assert len(al) == 2
    assert isinstance(al.pos_args[0], Literal)
    assert isinstance(al.pos_args[1], Literal)
    al.pos_args[0]._obj == 9
    al.pos_args[1]._obj == 3


def test_as_apply_list_of_applies():
    alist = [as_apply(i) for i in range(5)]

    al = as_apply(alist)
    assert isinstance(al, Apply)
    assert al.name == 'pos_args'
    # -- have to come back to this if Literal copies args
    assert al.pos_args == alist


def test_as_apply_dict_of_literals():
    d = {'a': 9, 'b': 10}
    ad = as_apply(d)
    assert isinstance(ad, Apply)
    assert ad.name == 'dict'
    assert len(ad) == 2
    assert ad.named_args[0][0] == 'a'
    assert ad.named_args[0][1]._obj == 9
    assert ad.named_args[1][0] == 'b'
    assert ad.named_args[1][1]._obj == 10


def test_as_apply_dict_of_applies():

    d = {'a': as_apply(9), 'b': as_apply(10)}
    ad = as_apply(d)
    assert isinstance(ad, Apply)
    assert ad.name == 'dict'
    assert len(ad) == 2
    assert ad.named_args[0][0] == 'a'
    assert ad.named_args[0][1]._obj == 9
    assert ad.named_args[1][0] == 'b'
    assert ad.named_args[1][1]._obj == 10


def test_as_apply_nested_dict():
    d = {'a': 9, 'b': {'c':11, 'd':12}}
    ad = as_apply(d)
    assert isinstance(ad, Apply)
    assert ad.name == 'dict'
    assert len(ad) == 2
    assert ad.named_args[0][0] == 'a'
    assert ad.named_args[0][1]._obj == 9
    assert ad.named_args[1][0] == 'b'
    assert ad.named_args[1][1].name == 'dict'
    assert ad.named_args[1][1].named_args[0][0] == 'c'
    assert ad.named_args[1][1].named_args[0][1]._obj == 11
    assert ad.named_args[1][1].named_args[1][0] == 'd'
    assert ad.named_args[1][1].named_args[1][1]._obj == 12


def test_lnorm():
    G = scope
    choice = G.choice
    uniform = G.uniform
    quantized_uniform = G.quantized_uniform

    inker_size = quantized_uniform(low=0, high=7.99, q=2) + 3
    # -- test that it runs
    lnorm = as_apply({'kwargs': {'inker_shape' : (inker_size, inker_size),
             'outker_shape' : (inker_size, inker_size),
             'remove_mean' : choice([0, 1]),
             'stretch' : uniform(low=0, high=10),
             'threshold' : uniform(
                 low=.1 / np.sqrt(10.),
                 high=10 * np.sqrt(10))
             }})
    assert len(str(lnorm)) == 977, len(str(lnorm))


def test_dfs():
    dd = as_apply({'c':11, 'd':12})

    d = {'a': 9, 'b': dd, 'y': dd, 'z': dd + 1}
    ad = as_apply(d)
    order = dfs(ad)
    print [str(o) for o in order]
    assert order[0]._obj == 9
    assert order[1]._obj == 11
    assert order[2]._obj == 12
    assert order[3].named_args[0][0] == 'c'
    assert order[4]._obj == 1
    assert order[5].name == 'add'
    assert order[6].named_args[0][0] == 'a'
    assert len(order) == 7


def test_o_len():
    obj = scope.draw_rng()
    x, y = obj
    assert x.name == 'getitem'
    assert x.pos_args[1]._obj == 0


