
def test_no_redundant_unions():
    s = dict(
            p0 = scope.uniform(0, 1),
            p1 = scope.uniform(2, 3),
            p2 = scope.one_of(-1, p0),
            p3 = scope.one_of(-2, p1))

    vectorize
