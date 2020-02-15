First of all, always remember the golden rules: before optimizing everything,
find the bottlenecks on an array of concrete examples. And before optimizing
everything, make the thing work.

## Generic

See how to put package on homebrew.

Python interface. Use pybind11. The name of the module should probably be
different, because of my poor initial choice. Maybe the non-agnostic fitting
interface could be implemented only at Python level. Possible name: YALSI for
"Yet Another Least Squares Interface".

## UReal and UReal2

### Inplace operations

Implement `UReal2::binary_assign`.

Think a sensible interface to allow inplace unary operations (implement
`UReal(2)::unary_inplace`). (Second optional argument to all the unary
functions in `math.hpp`? => No because they have to return something in one
case and nothing in the other.)

Expression templates. Is there something generic available? Should I prepare
definitions for using it? Or would it be so straightforward that I would just
need to give advice on it? It would be probably better to be self-contained
since I guess it is not so difficult to implement.

Expression templates ought to be extendible to user-defined functions.

If I use expression templates, should I concatenate formulas getting directly
the partial derivatives and then give them to `nary` instead of applying
`binary` repeatedly?

### Computing derivatives

#### Summary

  * Use `autograd` to write an initial proof-of-concept python library for
    least squares with propagation only from inputs to output of single fit and
    maxentropy done with `scipy`.
    
  * Write my own custom automatic differentiation in C++, doing both a forward
    greedy like I'm doing now, and a backward with a computational graph made
    naively with heap objects. Fw/bw is decided by the class, but for firsttime
    users I could make it that udouble2 is aliased to one or the other based on
    a macro.

#### Forward implementation

  * Do it greedily like I'm doing now, to keep things simple and low memory.

  * Use expression templates: they are very important since otherwise for each
    operation the entire gradient/hessian is copied.

  * The approximation of computing only the hessian diagonal is decided by a
    template parameter. The implementation is specialized since I can avoid
    putting the out-of-diagonal hessian class variable.

Should I use hashmaps instead of trees? The advatage of trees is that I can
take advantage of the implicit sorting when using two variables at a time, i.e.
in binary operations and covariance computation. The point is if it is faster
to merge hashmaps than trees, and if it is actually more memory efficient to
use hashmaps since my data type is so small. The memory access sparsity is
always bad with trees, while when I can iterate an hashmap without order (for
example in unary operations) it may be less sparse. I need to check how C++
hashmaps work.

#### Backward implementation

Build the computational graph in a classic object oriented way. The variables
hold a smart reference to a node in a graph, not necessarily the root. So if a
higher graph level variable gets deleted, the subgraphs can remain alive. Since
they can be shared this way, they are immutable for safety. Each node stores
its partial derivatives. It's fully OO so I can have subclasses around with
different class variables (example: a summation won't store a whole hessian
matrix).

The gradient/hessians use a sparse representation like I'm doing now with
trees. Backward accumulation starts with a empty sparse gradient. If I am a
node in the graph during accumulation, I get a factor from my parent, and then
I ask each child to accumulate its derivatives multiplying by the factor I give
them, and I give them my parent's factor multiplied by partial derivatives
respect to them. Leaves do the actual accumulation, possibly adding a new
element to the gradient. I still have to think about the hessian.

When a node is directly requested to compute the gradient, it caches the
result, so if afterwards a gradient evaluation is requested from higher-up, it
just uses the cache.

For efficiency it would be appropriate to fuse automatically some operations.
Like if I do a summation of a lot of variables, it makes more sense to have a
"summation" node than a deep tree of sums. But this contrasts with the policy
of being able to access the graph at any point, so I have to make a fusing
specialization on rvalues. But this could still break things if I move a
variable that I already used as input to something, then the node would be fused
without the other supergraph knowing about. I could mark with a bool the nodes
and fuse only the true ones (the roots).

#### Autodiff programs I have thought about

I found most of them on [autodiff.org](http://www.autodiff.org). I concluded
that none of them suits my interface needs, although of course for production
code here there's more than one can hope for.

  * `autograd`: can compute the hessian, but works on functions and needs the
    input all at once in an array. I could use it just to implement propagation
    on a single least squares fit in pure Python. It would be useful as a quick
    to write starting point for the fitting interface. It could be also in
    general be a "production" way of doing propagation: once my analysis is
    fixed and I know all the input variables and all the outputs I care about,
    I package everything in a function. This is not user friendly at all when
    you are writing the code the first time and experimenting. The single least
    squares implementation would also be useful for people who maybe don't care
    at all about the general propagation stuff. Just a curve fitter with
    bonuses. Technical advantage: it is no longer updated, but maintained, so
    no surprise incompatibilities!
    
  * `tensorflow`: well it seems they are copying `autograd` these days, so
    stick with `autograd`.

  * `ADMB`: a huge framework! Not what I need.
  
  * `ADEL`: C++, hessian, backward. But not maintained, no documentation. Nope.
    
  * `adept`: C++, only first order, forward and backward. But it works context
    dependent and I guess with the usual array gradient instead of sparse
    like I need. Note: `FastAD` claims to be its far superior successor.
  
  * `adnumber`: C++, higher orders, expression tree. No doc, no maintain.
    Probably not efficient.
    
  * `ADOL-C`: BOILERPLAAAAAATE and it is a large project.
  
  * `audi`: C++ and Python, higher orders, very cool... Still not well
    documented. In the Python examples they add variables as they wish and then
    require derivatives sparsily, and at each step it can be inspected, and the
    comment says "no additional computations when we request the derivative",
    which suggests forward prop. So I guess it is not very efficient for first
    and second order, it is optimized for very high orders. Also, under heavy
    development.
    
  * `autodiff_library`: hessian, fw & bw. You need to have the list of
    variables to compute the gradient (not good). Mentions the "Edge_pushing
    algorithm by Rober Gower" for backward full hessian, maybe look it up.
  
  * `CoDiPack`: C++, jacobian, hessian, fw, bw, higher orders, well documented,
    currently maintained. The most promising C++ self-contained up to now. But
    it still has the problem that you have to do either context-dependent
    taping or pass a function, and the gradient is dense.

  * `CppAD`: C++, jacobian, hessian, fw, bw, sparsity. As usual tape/function
    behaviour, and seems less easy to use than CoDiPack, although it boasts
    the additional feature of sparsity evaluation.
    
  * `CppADCodeGen`: version of `CppAD` that JITs with LLVM. So, nope again.
  
  * `FastAD`: C++17, easy to use, well maintained, claims to be verrry fast,
    jac hes fw bw, but: usual pattern of known input-outputs.

  * `libtaylor`: taylored to taylor... Very unconfortable API for my needs.
  
### Testing

Check higher order correlation functions using relations with lower order
correlations if there are identical arguments.

Do serious systematic tests of `UReal` interface, so that I can mess up freely.

### Other

Function to parse strings.

In math functions with singularities in the derivative, check that the mean is
not close to the singularity and warn the user on cerr. => No because it
would require to compute the standard deviation at each step.

Implement `numeric_limits`. In namespace `uncertainties` or `std`? Look for
what `boost::mp` does and what `Eigen` expects.

Check if `UReal` works as user type to Eigen.

Find a C++ template library for complex numbers with arbitrary numerical type
(maybe in Boost?) (std::complex need to be defined only for builtin types) and
check UReal(2) works with that.

Use the `UNCERTAINTIES_EXTERN_*` macros in all headers, and for friend
functions.

In the formatting functions, use the `<<` operator instead of the manual
computation I'm doing because probably it is more efficient and handles
unusual cases better. For example, probably the current implementation would
fail if more digits than what 64 bit supports were requested.

Use balanced sum to compute the standard deviation of UReal, and the mean
correction of UReal2? Maybe there is an efficient way since I'm already storing
coefficients in trees.

## UReal2

### Moment access interface redesign

Class function for returning at once the 4 moments.

Allow first order propagation for all moments with function `first_order_m()`.

Make it that m(1) returns the mean because it is more intuitive for the end
user. Add a class function that supersedes the current m(1) i.e. computes the
second-order correction to the mean.

The complexity to compute a kurtosis is O(n^4) where n is the number of
variables. It is very large. Maybe the interface of UReal2 should not allow the
user to compute by default a kurtosis to second order. For example: the
functions m, moments, or whatever may have an order parameter that defaults to
1, and a complexity parameter k that in any case caps the complexity to n^k by
truncating the series. This complexity capping may not give sensible results
however, it may break positive definiteness, so computed moments no longer
correspond to a distribution. NOTE: see below for idea to lower the
computational complexity.

Don't forget all this applies also to cov and corr.

### Moment computation efficiency

Other problem with O(n^4): since the number of terms summed is large, the
numerical precision is low. Example: with 1000 variables, it's 10^12 terms,
so it is a problem even for a double.

Idea for O(n^4): is there a way to simplify expressions? For example in the 4th
moment the last term is `H_ii H_jj H_kk H_ll V_iijjkkll`, which actually just
amounts to `H_ii H_jj H_kk H_ll` in computations. The summation is with the
constraint that indices can not take the same values. So I can write it as
`(sum_i H_ii)^4 - (terms with overlapping indices)`. The terms with overlapping
indices are O(n^3). Is there a way to exploit this ahead of expanding the
series?

For second order it is straightforward, I can do it by hand right now. Although
the lower bound is O(n^2) which already is the complexity of second order, so
it is just a constant factor optimization.

The order is at least O(n^2) anyway because that's the size of hessians. This
means that numerical precision is anyway a problem, a signifitive one if we are
using float. I can already see that the O(n^2) terms in the variance and
covariance could be computed with tree summation.

Idea: the O(n^2) terms in variance and covariance are those that contain out of
diagonal terms of the hessian. If it turns out it is not possible in general
to step down the O(n^k) complexity, it may still be that it is possible if I
assume diagonal hessians. This would guarantee consistency since I'm modifying
the function through which I propagate moments consistently across different
moments. By quick inspection, it appears that it is actually possible to stay
O(n) if I ignore off-diagonal hessian. Is there a sort of mean-field
approximation to sum off-diagonal terms into diagonal terms in O(n^2)? O(n^2)
is the lower bound because it is the size of the hessian, so this is probably
a difficult problem. It is easy to think about examples in which ignoring
off-diagonal is bad, like xy, but I should check how often they happen in real
life. Best case, it turns out in least squares is almost always fine to do that.

#### Why hessian diagonalization does not help

What if I diagonalize the hessian? The way I'm writing the function is

`f(x) = f0 + Gx + x^THx`

where `x` is the variables column vector, `G` is the gradient, `H` is half the
hessian. I diagonalize `H` so `H = U^TDU` (`H` is symmetrical so `U` is
orthogonal). Then

`f(x) = f0 + GU^TUx + (Ux)^TDUx`

So I've reduced the hessian to diagonal, but I lost the indipendency assumption
on the variables, which means that now all the moments are nontrivial. So it
is not helpful.

#### Summary

Vague deas I have up to now, in order of computational complexity:

  * O(n): first order (baseline)

  * O(n): only hessian diagonal

  * O(n^2) derivatives, O(n) moments: in some way, sum out of diagonal terms
    over the diagonal terms (I call this "mean field")

  * O(n^2) or O(n^3): approximate the hessian. jacobian? effective jacobian?

  * O(n^2) derivatives, O(n^k) moments: full computation

### Construction

Move `std_moments` and `central_moments` to `UReal2` constructor with option to
center moments.

Should I allow a negative threshold in the construction of UReal2 with this
meaning: that moments close to the boundary of allowed moments shall be
refused?

### Missing higher order functionality

Third and fourth order correlation functions (automatical generation). Do it
first storing all ids and coefficients in arrays then iterating?

Implement bias correction for higher order moments.

Ma ha veramente senso fare la propagazione unbiased per i momenti superiori?
Anzi di solito è roba positiva di cui voglio una stima positiva, insomma una
stima "bayesiana" (pensa i casini quando devi fare la media pesata e la matrice
di covarianza non è definita positiva). Idem per la curtosi immagino. Fare la
stima unbiased sembra una questione di completezza delle funzionalità del
software ma probabilmente non è quello che la gente dovrebbe fare, e questo è
uno dei miei requisiti, far fare cose sensate agli utenti. Inoltre altrimenti
per avere stima unbiased insieme a stima bayesiana dei momenti superiori dovrei
aggiungere parametri, complicando l'interfaccia e la comprensibilità del
codice. Tutti questi casini esistono per colpa dei frequentisti, però il bias
in vari casi è utile perché è una cosa che tende a conservarsi. Però comunque
mi rimane il problema che nel caso "correzione del bias" devo dare i momenti
superiori che tengono conto della correzione che c'ha la sua distribuzione
dopotutto. Va bene quello che sto già facendo o sbucano dei termini? E come
sbucano? E fanno venire positive le cose che devono essere positive?

Non è che per magia per correggere il bias anziché propagare mi basta fare il
ricentramento con la correzione al contrario? (mi sa che me l'ero già chiesto e
la risposta era no). Dunque: la correzione nel centrare la varianza compare al
quadrato quindi no spero comunque che in qualche modo ci sia un trucco magico
perché altrimenti devo riscrivere a mano di nuovo tutta la serie.

### Other

Design a version of `ureals` for `UReal2`. Linear transformation? Same
standardized moments for all variables? Or can I obtain arbitrary moments up
to the fourth order with a quadratical transformation of independent variables?

Maximum entropy pdf given the moments. See R.V. Abramov 2010 paper. Question:
doing maxentropy on only one variable is equivalent to doing on two and then
marginalize?

Fit with propagation like `lsqfit` but at second order. Do a generic wrapper
of a least squares procedure. Look at lsqfit for how to diagonalize the data
covariance matrix efficiently because it is O(n^3), and I probably start with
implicit block diagonal.

In the documentation explain the bayesian-frequentist interpretation of
propagation kind.

nella propagazione al secondo ordine riesco a fare qualcosa che in qualche
senso approssima risultati bayesiani? tipo come lsqfit che si considera
bayesiano in approssimazione gaussiana. allora: la propagazione di tipo M è
quella da usare per propagare i momenti di un posteriore. però si applica al
fit ai minimi quadrati? cioè, posso considerare il fit ai minimi quadrati come
un'approssimazione della media del posteriore (e quindi quando è biased dire e
sticazzi?) minimi quadrati mi dà la moda del posteriore assumendo che dati e
priori siano gaussiani. assumendo che il posteriore sia gaussiano, mi dà allora
la media. se voglio andare al secondo ordine cosa devo fare? non minimi
quadrati ma minimo log p? È impraticabile perché per ricavare log p dai momenti
dovrei fare maxentropy che è computazionalmente infattibile. se io faccio
minimi quadrati e poi propago M, quello che sto facendo è approssimare la media
del posteriore? Non vedo direttamente perché, però il fatto che non facendo
nulla ho il primo ordine, e che facendo propagazione E ho effettivamente la
correzione del bias, mi suggerisce che facendo propagazione M sto stimando la
media del posteriore.

