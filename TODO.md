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

I'm using forward automatic differentiation. Would it be better to use backward
differentiation? In neural networks backward is better because I have short
output/long input and I already know that input will propagate through almost
all nodes of the computational graph. Here it may not be the case, it may
reasonably happen that nodes depend on separate sets of variables, so I guess
forward mode is better, although it still holds that most usages I can think of
have short output/long input. For now I will stick with forward mode because it
is simpler and uses less memory. Although to make forward mode really efficient
I need expression templates, which is the compile time equivalent of
implementing backward differentiation. Although I have the impression backward
mode is useful when I know already the set of variables over which I'm
differentiating, which would require something like forward propagation to keep
track of.

Should I use hashmaps instead of trees? The advatage of trees is that I can take
advantage of the implicit sorting when using two variables at a time, i.e. in
binary operations and covariance computation. The point is if it is faster to
merge hashmaps than trees, and if it is actually more memory efficient to use
hashmaps since my data type is so small. The memory access sparsity is always
bad with trees, while when I can iterate an hashmap without order (for example
in unary operations) it may be less sparse. I need to check how C++ hashmaps
work.

### Testing

Check higher order correlation functions using relations with lower order
correlations if there are identical arguments.

Do serious systematic tests of `UReal` interface, so that I can mess up freely.

### Other

Function to parse strings.

In math functions with singularities in the derivative, check that the mean is
not close to the singularity and warn the user on cerr. => No because it
would require to compute the standard deviation at each step.

Implement `numeric_limits`. In namespace `uncertainties` or `std`? Look for what
`boost::mp` does and what `Eigen` expects.

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

What if I diagonalize the hessian? The way I'm writing the function is

`f(x) = f0 + Gx + x^THx`

where `x` is the variables column vector, `G` is the gradient, `H` is half the
hessian. I diagonalize `H` so `H = U^TDU` (`H` is symmetrical so `U` is
orthogonal). Then

`f(x) = f0 + GU^TUx + (Ux)^TDUx`

So I've reduced the hessian to diagonal, but I lost the indipendency assumption
on the variables, which means that now all the moments are nontrivial. So it
is not helpful.

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

