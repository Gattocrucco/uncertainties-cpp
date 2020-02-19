# TODO for uncertainties-cpp

First of all, always remember the golden rules: before optimizing everything,
find the bottlenecks on an array of concrete examples. And before optimizing
everything, make the thing work.

## Roadmap

  * Implement a pure python version with `autograd` and `scipy`. Name: YALSI
    for Yet Another Least Squares Interface. Features: moment propagation with
    diagonal hessian, using full function specification (because we are using
    `autograd`), least squares with propagation from input to output, but it
    can be used as function in an `autograd` graph, and finally maxentropy.
    Possible icon: a gradient bimodal in a circle of radial errorbars.
 
  * Complete the test suite and fix the interface and code bloat issues in
    UReal2.
  
  * Implement a sensible internal interface in UReal2 for computing the moments
    with different algorithms. First store gradient and hessian in Eigen
    matrices, apart from simple algorithms that can just iterate once on
    everything (like, I guess, diagonal hessian approx).
  
  * Implement diagonal hessian approx. Superclass does diagonal, subclass does
    full.
  
  * Write python wrapper with `pybind11`.
  
  * Add mutiple variable higher order moments.
  
  * Port Abramov's code for maxentropy.
  
  * Implement expression templates.
  
  * Implement computational graph type.

## Packaging

See how to put package on homebrew.

Python interface. Use `pybind11`.

## UReal and UReal2

### Inplace operations

Finish implementing `UReal2::binary_assign`.

Think a sensible interface to allow inplace unary operations (implement
`UReal(2)::unary_inplace`). (Second optional argument to all the unary
functions in `math.hpp`? => No because they have to return something in one
case and nothing in the other.) This may not be needed if I implement
expression templates.

Expression templates. Is there something generic available? Should I prepare
definitions for using it? Or would it be so straightforward that I would just
need to give advice on it? It would be probably better to be self-contained
since I guess it is not so difficult to implement.

Expression templates ought to be extendible to user-defined functions.

If I use expression templates, should I concatenate formulas getting directly
the partial derivatives and then give them to `nary` instead of applying
`binary` repeatedly?

### Computing derivatives

Write my own custom automatic differentiation in C++, doing both a forward
greedy like I'm doing now, and a backward with a computational graph made
naively with heap objects. Fw/bw is decided by the class, but for firsttime
users I could make it that udouble2 is aliased to one or the other based on a
macro.

#### Forward implementation

  * Do it greedily like I'm doing now, to keep things simple and low memory.

  * Use expression templates: they are very important since otherwise for each
    operation the entire gradient/hessian is copied.

  * There is a `UReal2` class that computes only the hessian diagonal and a
    subclass that does full hessian, since in the first case I can avoid
    storing the out-of-diagonal map.
    
#### Backward implementation

Build the computational graph in a classic object oriented way. The variables
hold a smart reference to a node in a graph, not necessarily the root. So if a
higher graph level variable gets deleted, the subgraphs can remain alive. Since
they can be shared this way, they are immutable for safety. Each node stores
its partial derivatives. It's fully OO so I can have subclasses around with
different class variables (example: a summation won't store a whole hessian
matrix).

The gradient uses a sparse representation like I'm doing now with trees.
Backward accumulation starts with a empty sparse gradient. If I am a node in
the graph during accumulation, I get a factor from my parent, and then I ask
each child to accumulate its derivatives multiplying by the factor I give them,
and I give them my parent's factor multiplied by partial derivatives respect to
them. Leaves do the actual accumulation, possibly adding a new element to the
gradient.

Doing backward the hessian this way is not possible because in a non-unary node
the off diagonal terms in the partial hessian need accumulation with the
product of the gradients of two different nodes. Doing the hessian backward
effectively means doing the backward gradient of the forward gradient, so it is
not quite efficient. Look at the stochastic hessian estimation from 1206.6464,
they claim to reduce the computational complexity while achieving reasonable
precision.

When a node is directly requested to compute the gradient, it caches the
result, so if afterwards a gradient evaluation is requested from higher-up, it
just uses the cache. Example where this is very useful: I do a fit, and then
use the fit result to compute the model prediction at many points to draw a
plot.

For efficiency it would be appropriate to fuse automatically some operations.
Like if I do a summation of a lot of variables, it makes more sense to have a
"summation" node than a deep tree of sums. But this contrasts with the policy
of being able to access the graph at any point, so I have to make a fusing
specialization on rvalues. But this could still break things if I move a
variable that I already used as input to something, then the node would be
fused without the other supergraph knowing about. Since I would be using smart
pointers, use `use_count()` to check if the node in uniquely owned (not thread
safe though). It is also convenient to fuse unary operations, i.e. compute the
product of derivatives directly.

#### Alternatives to `std::map`

Should I use hashmaps instead of trees? The memory access is not better because
`std::unordered_map` is implemented with pointers. And I lose ordered access,
which is useful when merging.

Other alternative: use Eigen's sparse matrices. The problems are that the
storage is not completely ordered, it is inefficient to grow one element at a
time, and size must be specified.

#### Autodiff programs I have thought about

I found most of them on [autodiff.org](http://www.autodiff.org). I concluded
that none of them suits my interface needs, although of course for production
code here there's more than one can hope for.

  * **`autograd`**: any order derivatives, backward & forward, pure python. But
    it is difficult to adapt it efficiently to an `uncertainties`-like
    interface, it wants all the input at a time, so I would have to build a
    computational graph and then accumulate all the unique inputs.
    
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
  
  * **`CoDiPack`**: C++, jacobian, hessian, fw, bw, higher orders, well documented,
    currently maintained. The most promising C++ self-contained up to now. But
    it still has the problem that you have to do either context-dependent
    taping or pass a function, and the gradient is dense.

  * `CppAD`: C++, jacobian, hessian, fw, bw, sparsity. As usual tape/function
    behaviour, and seems less easy to use than CoDiPack, although it boasts
    the additional feature of sparsity evaluation.
    
  * `CppADCodeGen`: version of `CppAD` that JITs with LLVM. So, nope again.
  
  * **`FastAD`**: C++17, easy to use, well maintained, claims to be verrry
    fast, jac hes fw bw, but: usual pattern of known input-outputs.

  * `libtaylor`: taylored to taylor... Very unconfortable API for my needs.
  
### Testing

Check higher order correlation functions using relations with lower order
correlations if there are identical arguments.

Do serious systematic tests of `UReal` interface, so that I can mess up freely.

The test I'm doing in `distr.cpp` is on a degenerate rank 1 hessian. It is
possible that this gets fooled, although I guess not.

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

## UReal2 only

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

### Moment computation optimization

Ideas I have up to now, in order of computational complexity (`n` is the number
of independent variables). Note that the number of terms in moment computation
grows quickly with order `k` and I'm ignoring that, there's always something
like `k!` in the moment computation.

  * O(n): first order (baseline).

  * O(n): compute and use only hessian diagonal. Collect terms in the
    moment summation, example: `∑i≠j Hii Hjj = (∑i Hii)^2 - ∑i Hii^2`.

  * O(n^2) derivatives, O(n^2) transform, O(n) moments: in some way, sum
    off-diagonal hessian terms over the diagonal terms (I call this "mean
    field"). The trace should be preserved because it gives the bias
    correction. Basic idea: `Hii -> ∑j Hij - 1/n ∑j≠k Hjk`. Intuitively this
    reweights the variables but won't increase the variance as it should. Think
    in 2D the case `H11 = H22 = 0`, `H12 = 1`. The transformation would give
    `H = 0`! But if I allowed the trace to vary, then the bias would be nonzero,
    which violates the symmetry.

  * O(n^2) derivatives, O(n^2 m) transform, O(n m^k) moments: trace preserving
    rank m approximation of the hessian. Problematic example from above: the
    eigenvalues are 1 and -1. The eigenvectors are `[1 1]`, `[1 -1]`. Their
    outer products are `[1 1; 1 1]` and `[1 -1; -1 1]`. If I want to make
    either with zero trace with a scaling, again I obtain `H = 0`. I guess when
    the trace is zero there's no way around than having `H = 0` if I'm linear.
    
  * O(n^2) derivatives, O(n^k) moments: full computation. I do not exclude
    there is a way to reduce the O(n^k) but I suspect no.

Does mean field makes sense really? And is there a O(n^2) way to decide if it
is better to use the diagonal or a lower rank?

#### Computing the lower rank approximation

How should I pick the m eigenvalues in general?

m = 1 case: I have to preserve the trace and it makes sense to preserve the
curvature sign in the direction I pick, so choose the maximum absolute value
eigenvalue with same sign of the trace.

For general m: find the first as in the m = 1 case, subtract from the trace,
and so on.

The algorithm I should use is Lanczos. C++ header only implementations I found:

  * `lambda-lanczos`: very small and simple. Not really a supported project.
    I would use it to implement directly my algorithm by finding an eigenvalue
    at a time and each time I subtract it from the matrix.
  
  * `spectra`: a large library based on Eigen, a port of ARPACK. I could use
    it this way: I ask for 2m eigenvalues with option BOTH_ENDS, so in the
    extreme case where I need all eigenvalues with the same sign I surely have
    m of them.

I guess I'll go with the second, mainly because what I write by hand is less
dependable than well-tested algorithms. Don't forget to check if it is actually
faster to do a full diagonalization than using this! There will be a threshold
on the matrix size. Also: should I use SVD?

#### Numerical error

For a O(n^2) computation, I guess the digits I lose are as many as the digits
of n. Example: 10000 variables, I use floats, I lose 5 out digits of 7. This
assuming a random behaviour with the result always on the order of the
operands. So I should care about using good summation algorithms, probably
having the out of diagonal hessian in its own tree I should use binary
summation. At least, I'm sure I can do this for second order moments, I don't
know for higher ones.

A O(n^4) computation may be problematic even for a double, but when it becomes
so long I think the first problem is computational time, since then it would be
hugely convenient to do a monte carlo.

#### Why hessian diagonalization does not help

What if I diagonalize the hessian? The way I'm approximating the function is

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
refused? Because maybe then it happens that the output moments can fall past
the boundary or give numerical problems anyway. If I do that, I guess I should
remove altogether the possibility of allowing numerically past-the-boundary
moments? But it may be interesting to, say, input a bunch of deltas in the
computation and see what comes out, since then results are probably analytical.
Idea: I can always perturbate output moments if they turn out nonsensical, this
works in general while hoping that input ok => output ok may fail.

### Missing higher order functionality

Third and fourth order correlation functions (automatical generation). Do it
first storing all ids and coefficients in arrays then iterating? Reconsider
this after I've implemented efficient moment computation.

Implement bias correction for higher order moments. In the sense that I compute
to second order the unbiased estimates of the moments of the bias corrected
estimator. This means that they can come out non positive definite. Is there a
way to do non-unbiased estimates for the bias corrected estimator that are
positive definite and still better than bare propagation? The truth is I do not
want to write the CAS for these new more complicated series and implement them.

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

Least squares with second order propagation as approximate bayesian inference?
Check this empirically.

Avoid code bloat by putting shared functionality in a superclass
`UReal2Base<Real>` and the subclassing to `UReal2E` and `UReal2P`. Because
actually all the difference now is only in the function `n` and in conversion
rules.
