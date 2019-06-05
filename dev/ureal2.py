from sympy.combinatorics.partitions import IntegerPartition
from sympy.combinatorics import Permutation
import copy
import numpy as np
from fractions import Fraction
from collections import Counter, OrderedDict

SIMPL_V_REPR = False

indices = ('i', 'j', 'k', 'l', 'm', 'n', 'o', 'p')

def isnum(x):
    return isinstance(x, int) or isinstance(x, Fraction)

def gen_index_perm(vlist, rank, indices, indices_bags):
    #ind = ' ' * 4 * len(indices)
    #print(f'{ind}gen_index_perm({len(vlist)}, {rank}, {indices}, {indices_bags})')
    if len(indices) == rank:
        v = V(rank, indices)
        vlist.append(v)
    else:
        for group_size, bags in indices_bags.items():
            #print(f'{ind} {group_size}: {bags}')
            assert(len(bags) == group_size)
            for i in range(len(bags)):
                bag = bags[i]
                #print(f'{ind}  {i}: {bag}')
                for j in range(len(bag)):
                    #print(f'{ind}   {j}: {bag[j]}')
                    new_indices_bags = copy.deepcopy(indices_bags)
                    new_indices_bags[group_size][i] = bag[:j] + bag[j + 1:]
                    if i > 0:
                        new_indices_bags[group_size][i - 1] = prev_bag + bag[j:j + 1]
                    gen_index_perm(vlist, rank, indices + bag[j:j + 1], new_indices_bags)
                    if i == len(bags) - 1:
                        break
                prev_bag = bag

class V:
    def __init__(self, rank, indices=()):
        self._rank = rank
        self._indices = tuple(indices)
    
    def sort_indices(self):
        self._indices = tuple(sorted(self._indices))
        return self

    def __repr__(self):
        if self._indices:
            rankstr = ''
            if SIMPL_V_REPR:
                c = Counter(self._indices)
                idxs = filter(lambda i: c[i] > 2, self._indices)
            else:
                idxs = self._indices
            indstr = '_' + ''.join(map(lambda i: indices[i], idxs))
        else:
            indstr = ''
            rankstr = f'{self._rank}'
        return f'V{rankstr}{indstr}'
    
    def __lt__(self, d):
        if isinstance(d, V):
            return self._rank < d._rank or self._rank == d._rank and self._indices < d._indices
        elif isinstance(d, D):
            return False
        elif isnum(d):
            return False
        else:
            raise TypeError("'<' not supported between instances of '{}' and '{}'".format(D, d.__class__))
    
    def __eq__(self, obj):
        return isinstance(obj, V) and self._rank == obj._rank and self._indices == obj._indices
    
    def index_V(self):
        p = IntegerPartition([self._rank])
        end = IntegerPartition([1] * self._rank)
        vlist = []
        while p > end:
            if p.as_dict().get(1, 0) == 0:
                indices_bags = dict()
                i = 0
                for group_size, group_count in p.as_dict().items():
                    bags = list(() for i in range(group_size))
                    bags[-1] = tuple(i + j for j in range(group_count))
                    i += group_count
                    indices_bags[group_size] = bags
                gen_index_perm(vlist, self._rank, (), indices_bags)
            p = p.prev_lex()
        return Sum(*vlist)
    
class D:
    def __init__(self, rank, var, indices=()):
        self._rank = rank
        self._var = var
        self._indices = tuple(indices)
    
    def sort_indices(self):
        self._indices = tuple(sorted(self._indices))
        return self

    def __repr__(self):
        letter = 'G' if self._rank == 1 else 'H'
        if self._indices:
            indstr = '_' + ''.join(map(lambda i: indices[i], self._indices))
        else:
            indstr = ''
        if str(self._var) == '':
            varstr = ''
        else:
            varstr = f'({self._var})'
        return f'{letter}{varstr}{indstr}'
    
    def __lt__(self, d):
        if isinstance(d, D):
            return self._var < d._var or self._var == d._var and self._rank < d._rank or self._rank == d._rank and self._indices < d._indices
        elif isinstance(d, V):
            return True
        elif isnum(d):
            return False
        else:
            raise TypeError("'<' not supported between instances of '{}' and '{}'".format(D, d.__class__))
    
    def __eq__(self, obj):
        return isinstance(obj, D) and self._rank == obj._rank and self._var == obj._var and self._indices == obj._indices

class Reductor:
    def __init__(self, *args):
        self._list = list(args)

    def recursive(self, method):
        # print(f'recursive {method} on {self}')
        for i in range(len(self._list)):
            if hasattr(self._list[i], 'recursive'):
                self._list[i] = self._list[i].recursive(method)
            elif hasattr(self._list[i], method):
                # print(f'    {method} on {self._list[i]}')
                self._list[i] = getattr(self._list[i], method)()
        if hasattr(self, method):
            # print(f'    {method} on {self}')
            self = getattr(self, method)()
        return self
    
    def concat(self):
        l = []
        for obj in self._list:
            if isinstance(obj, self.__class__):
                l += obj._list
            else:
                l.append(obj)
        self._list = l
        return l[0] if len(l) == 1 else self
    
    def sort(self):
        try:
            self._list = sorted(self._list)
        except TypeError:
            pass
        return self
    
    def __lt__(self, obj):
        return isinstance(obj, self.__class__) and self._list[::-1] < obj._list[::-1]
    
    def __eq__(self, obj):
        return isinstance(obj, self.__class__) and len(self._list) == len(obj._list) and all(x == y for x, y in zip(self._list, obj._list))
    
class Mult(Reductor):
    def __repr__(self):
        return ' '.join('(' + x.__repr__() + ')' if isinstance(x, Reductor) else x.__repr__() for x in self._list)
    
    def reduce(self):
        numbers = []
        l = []
        for obj in self._list:
            if isnum(obj):
                numbers.append(obj)
            else:
                l.append(obj)
        p = 1
        for n in numbers:
            p *= n
        if int(p) == p:
            p = int(p)
        if p == 1:
            self._list = l
        elif p == 0:
            return 0
        else:
            self._list = [p] + l
        if len(self._list) > 1:
            return self
        elif len(self._list) == 1:
            return self._list[0]
        else:
            return 1
    
    def gen_V(self):
        rank = sum(map(lambda x: x._rank, filter(lambda x: isinstance(x, D), self._list)))
        if rank:
            self._list.append(V(rank))
        return self
    
    def expand(self):
        for i in range(len(self._list)):
            obj = self._list[i]
            if isinstance(obj, Sum):
                terms = []
                for t in obj._list:
                    l_left = copy.deepcopy(self._list[:i])
                    l_right = copy.deepcopy(self._list[i + 1:])
                    terms.append(Mult(*l_left, t, *l_right).expand())
                return Sum(*terms)
        return self
    
    def index_D(self):
        for Vobj in self._list:
            if isinstance(Vobj, V):
                break
        else:
            return self
        Dobjs = []
        for obj in self._list:
            if isinstance(obj, D):
                Dobjs.append(obj)
        i = 0
        for d in Dobjs:
            d._indices = Vobj._indices[i:i + d._rank]
            i += d._rank
        assert(i == Vobj._rank)
        return self
    
    def split_D(self):
        Dobjs = []
        other = []
        for obj in self._list:
            if isinstance(obj, D):
                Dobjs.append(obj)
            else:
                other.append(obj)
        if not Dobjs:
            return self
        indices = ()
        for obj in Dobjs:
            indices += obj._indices
        pfirst = Permutation(list(range(len(indices))))
        p = pfirst
        terms = []
        while True:
            l = copy.deepcopy(Dobjs)
            i = 0
            for d in l:
                d._indices = tuple(indices[(i + j)^p] for j in range(len(d._indices)))
                i += len(d._indices)
            terms.append(Mult(*l))
            p += 1
            if p == pfirst:
                break
        self._list = other + [Fraction(1, len(terms)), Sum(*terms)]
        return self
    
    def normalize_D(self):
        Dobjs = []
        for obj in self._list:
            if isinstance(obj, D):
                Dobjs.append(obj)
        if not Dobjs:
            return self
        indices = ()
        r2s_indices = ()
        for d in Dobjs:
            indices += obj._indices
            if d._rank == 2 and d._indices and d._indices[0] == d._indices[1]:
                r2s_indices += d._indices[:1]
        if not r2s_indices:
            return self
        # print(f'normalize_D on {self}')
        count = Counter(indices)
        count_count = Counter(sorted(count.values()))
        p = list(range(len(set(indices))))
        assert(set(indices) == set(range(len(set(indices)))))
        for c, cc in filter(lambda ccc: ccc[1] > 1, count_count.items()):
            swappable_indices = set(filter(lambda i: count[i] == c, indices))
            # print(f'    can swap {swappable_indices}')
            assert(len(swappable_indices) > 1)
            first_indices = tuple(filter(lambda i: i in swappable_indices, OrderedDict(zip(r2s_indices, (None,) *len(r2s_indices)))))
            second_indices = tuple(filter(lambda i: i in swappable_indices and not i in r2s_indices, OrderedDict(zip(indices, (None,) *len(indices)))))
            old_indices = first_indices + second_indices
            new_indices = sorted(swappable_indices)
            for i, j in zip(old_indices, new_indices):
                # print(f'        send {i} -> {j}')
                p[i] = j
        # print(f'    final perm is {p}')
        assert(set(p) == set(indices))
        for d in Dobjs:
            d._indices = tuple(p[i] for i in d._indices)
        return self

    def classify(self):
        for obj in self._list:
            assert(not isinstance(obj, Reductor))
    
        indices = []
        for obj in self._list:
            if isinstance(obj, D):
                assert(obj._indices)
                indices.append(obj._indices)
    
        nindices = len(set(sum(indices, ())))
    
        if nindices == 1:
            self._cycle = 'A'
            return self
    
        if nindices == 2:
            for idxs in indices:
                if len(set(idxs)) == 2:
                    self._cycle = 'B'
                    return self
            self._cycle = 'AA'
            return self
    
        if nindices == 3:
            c = Counter(map(lambda i: len(set(i)), indices))
            if not 2 in c:
                self._cycle = 'AAA'
                return self
            
            
        

def stripfactor(x):
    if isinstance(x, Mult) and x._list and isnum(x._list[0]):
        return Mult(*x._list[1:])
    else:
        return x

def getfactor(x):
    if isinstance(x, Mult) and x._list and isnum(x._list[0]):
        return x._list[0]
    else:
        return 1

class Sum(Reductor):
    def __repr__(self, sep=''):
        return f' + {sep}'.join('(' + x.__repr__() + ')' if isinstance(x, Sum) else x.__repr__() for x in self._list)
        
    def reduce(self):
        numbers = []
        l = []
        for obj in self._list:
            if isnum(obj):
                numbers.append(obj)
            else:
                l.append(obj)
        s = sum(numbers)
        if int(s) == s:
            s = int(s)
        if s == 0:
            self._list = l
        else:
            self._list = [s] + l
        if len(self._list) > 1:
            return self
        elif len(self._list) == 1:
            return self._list[0]
        else:
            return 0
    
    def harvest(self):
        terms = []
        counts = []
        while self._list:
            obj = stripfactor(self._list[-1])
            count = getfactor(self._list[-1])
            for i in range(len(terms)):
                if terms[i] == obj:
                    counts[i] += count
                    break
            else:
                terms.append(obj)
                counts.append(count)
            self._list.pop()
        self._list = []
        # print(grouped_terms)
        for i in reversed(range(len(terms))):
            self._list.append(Mult(counts[i], terms[i]) if counts[i] != 1 else terms[i])
        return self

# def gen_terms(terms, vars, l, n_2, n_1):
#     if n_2 == n_1 == 0:
#         terms.append(Mult(*(D(l[i], vars[i]) for i in range(len(l)))))
#     else:
#         if n_1 > 0: gen_terms(terms, vars, l + (1,), n_2, n_1 - 1)
#         if n_2 > 0: gen_terms(terms, vars, l + (2,), n_2 - 1, n_1)
#
# def gen_corr_base_expr(*vars):
#     terms = []
#     for v_rank in range(len(vars), 1 + 2 * len(vars)):
#         v_terms = []
#         for n_2 in range(1 + v_rank // 2):
#             n_1 = len(vars) - n_2
#             if n_1 + 2 * n_2 == v_rank:
#                 gen_terms(v_terms, vars, (), n_2, n_1)
#         if v_terms:
#             terms.append(Mult(Sum(*v_terms), V(v_rank)))
#     return Sum(*terms)

def gen_corr(*vars):
    # e = gen_corr_base_expr(*vars)
    e = Mult(*[Sum(D(1, v), D(2, v)) for v in vars])
    
    # do multiplication
    e = e.recursive('expand')
    e = e.recursive('concat')
    
    # group equal terms in case there are repeated variables
    e = e.recursive('sort')
    e = e.recursive('harvest')
    
    # put V tensors
    e = e.recursive('gen_V')
    
    # split summations into diagonal and off-diagonal
    e = e.recursive('index_V')
    e = e.recursive('expand')
    e = e.recursive('concat')
    
    # put indices on D tensors
    e = e.recursive('index_D')
    e = e.recursive('sort_indices')
    e = e.recursive('sort')
    e = e.recursive('harvest')
    e = e.recursive('concat')
    e = e.recursive('reduce')
    
    # normalize usage of mute indices to spot equal terms
    e = e.recursive('normalize_D')
    e = e.recursive('sort_indices')
    e = e.recursive('sort')
    e = e.recursive('harvest')
    e = e.recursive('concat')

    return e