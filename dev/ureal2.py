indices = ('i', 'j', 'k', 'l', 'm', 'n', 'o', 'p')

class V:
    def __init__(self, rank):
        self._rank = rank
        self._indices = tuple(range(rank))
    
    def __repr__(self):
        return 'V^{{({})}}_{{{}}}'.format(self._rank, ''.join(map(lambda i: indices[i], self._indices)))
    
class D:
    def __init__(self, rank, var):
        self._rank = rank
        self._var = var

    def __repr__(self):
        letter = 'G' if self._rank == 1 else 'H'
        return f'{letter}^{{({self._var})}}'
    
class Mult:
    def __init__(self, *args):
        self._list = args
    
    def __repr__(self):
        return ' '.join('(' + x.__repr__() + ')' if isinstance(x, Sum) and len(x._list) > 1 else x.__repr__() for x in self._list)

class Sum:
    def __init__(self, *args):
        self._list = args
    
    def __repr__(self):
        return ' + '.join(x.__repr__() for x in self._list)

def gen_terms(terms, vars, l, n_2, n_1):
    if n_2 == n_1 == 0:
        terms.append(Mult(*(D(l[i], vars[i]) for i in range(len(l)))))
    else:
        if n_1 > 0: gen_terms(terms, vars, l + (1,), n_2, n_1 - 1)
        if n_2 > 0: gen_terms(terms, vars, l + (2,), n_2 - 1, n_1)

def gen_corr_base_expr(*vars):
    terms = []
    for v_rank in range(len(vars), 1 + 2 * len(vars)):
        v_terms = []
        for n_2 in range(1 + v_rank // 2):
            n_1 = len(vars) - n_2
            if n_1 + 2 * n_2 == v_rank:
                gen_terms(v_terms, vars, (), n_2, n_1)
        if v_terms:
            terms.append(Mult(Sum(*v_terms), V(v_rank)))
    return Sum(*terms)
            