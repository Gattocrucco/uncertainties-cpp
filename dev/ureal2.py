from sympy.combinatorics.partitions import IntegerPartition
import copy

indices = ('i', 'j', 'k', 'l', 'm', 'n', 'o', 'p')

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
            indstr = '_{' + ''.join(map(lambda i: indices[i], self._indices)) + '}'
        else:
            indstr = ''
        return f'V^{{({self._rank})}}{indstr}'
    
    def __lt__(self, d):
        if isinstance(d, V):
            return self._rank < d._rank or self._rank == d._rank and self._indices < d._indices
        elif isinstance(d, D):
            return False
        elif isinstance(d, int):
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
            indstr = '_{' + ''.join(map(lambda i: indices[i], self._indices)) + '}'
        else:
            indstr = ''
        return f'{letter}^{{({self._var})}}{indstr}'
    
    def __lt__(self, d):
        if isinstance(d, D):
            return self._var < d._var or self._var == d._var and self._rank < d._rank or self._rank == d._rank and self._indices < d._indices
        elif isinstance(d, V):
            return True
        elif isinstance(d, int):
            return False
        else:
            raise TypeError("'<' not supported between instances of '{}' and '{}'".format(D, d.__class__))
    
    def __eq__(self, obj):
        return isinstance(obj, D) and self._rank == obj._rank and self._var == obj._var and self._indices == obj._indices

class Reductor:
    def __init__(self, *args):
        self._list = list(args)

    def recursive(self, method):
        for i in range(len(self._list)):
            if hasattr(self._list[i], 'recursive'):
                self._list[i] = self._list[i].recursive(method)
            if hasattr(self._list[i], method):
                self._list[i] = getattr(self._list[i], method)()
        if hasattr(self, method):
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
    
    def __eq__(self, obj):
        return isinstance(obj, self.__class__) and len(self._list) == len(obj._list) and all(x == y for x, y in zip(self._list, obj._list))

class Mult(Reductor):
    def __repr__(self):
        return ' '.join('(' + x.__repr__() + ')' if isinstance(x, Reductor) else x.__repr__() for x in self._list)
    
    def simplify(self):
        if any(isinstance(x, Sum) and not x._list for x in self._list):
            self._list = []
        return self
    
    def reduce(self):
        numbers = []
        l = []
        for obj in self._list:
            if isinstance(obj, int):
                numbers.append(obj)
            else:
                l.append(obj)
        p = 1
        for n in numbers:
            p *= n
        if p == 1:
            self._list = l
        elif p == 0:
            self._list = []
        else:
            self._list = [p] + l
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

class Sum(Reductor):
    def __repr__(self, sep=''):
        return f' + {sep}'.join('(' + x.__repr__() + ')' if isinstance(x, Sum) else x.__repr__() for x in self._list)
        
    def simplify(self):
        self._list = list(filter(lambda x: not isinstance(x, Mult) or x._list, self._list))
        return self
    
    def reduce(self):
        numbers = []
        l = []
        for obj in self._list:
            if isinstance(obj, int):
                numbers.append(obj)
            else:
                l.append(obj)
        s = sum(numbers)
        if s == 0:
            self._list = l
        else:
            self._list = [s] + l
        return self
    
    def harvest(self):
        grouped_terms = []
        while self._list:
            for l in grouped_terms:
                if l and l[0] == self._list[-1]:
                    l.append(self._list.pop())
                    break
            else:
                grouped_terms.append([self._list.pop()])
        self._list = [Mult(len(l), l[0]) if len(l) > 1 else l[0] for l in reversed(grouped_terms)]
        return self

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

def gen_corr(*vars):
    e = gen_corr_base_expr(*vars)
    e = e.recursive('sort')
    e = e.recursive('harvest')
    e = e.recursive('index_V')
    e = e.recursive('expand')
    e = e.recursive('concat')
    e = e.recursive('index_D')
    e = e.recursive('sort_indices')
    e = e.recursive('sort')
    e = e.recursive('harvest')
    e = e.recursive('concat')
    e = e.recursive('reduce')
    return e