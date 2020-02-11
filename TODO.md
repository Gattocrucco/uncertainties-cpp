## Generic

See how to put package on homebrew.

Python interface. Use pybind11. The name of the module should probably be
different, because of my poor initial choice. Maybe the non-agnostic fitting
interface could be implemented only at Python level.

## UReal and UReal2

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

Expression templates. Is there something generic available? Should I prepare
definitions for using it? Or would it be so straightforward that I would just
need to give advice on it?

## UReal

Use balanced sum to compute the standard deviation?

## UReal2

Allow first order propagation for all moments with function `first_order_m()`.

Implement bias correction for higher order moments.

Design a version of `ureals` for `UReal2`. Linear transformation? Same
standardized moments for all variables?

Third and fourth order correlation functions (automatical generation). Do it
first storing all ids and coefficients in arrays then iterating.

Check higher order correlation functions using relations with lower order
correlations if there are identical arguments.

Implement `UReal2::binary_assign`.

Think a sensible interface to allow inplace unary operations (implement
`UReal(2)::unary_inplace`). (Second optional argument to all the unary
functions in `math.hpp`? => No because they have to return something in one
case and nothing in the other.)

Maximum entropy pdf given the moments. See RV Abramov paper.

Fit with propagation like `lsqfit` but at second order.

Move `std_moments` and `central_moments` to `UReal2` constructor with option to
center moments.

Class function for returning at once the 4 moments.

Make it that m(1) returns the mean because it is more intuitive for the end
user. Add a class function that supersedes the current m(1) i.e. computes the
second-order correction to the mean.

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
