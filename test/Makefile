CC = c++
CFLAGS = -std=c++11 -g -I.. -I/usr/local/include/eigen3
CFLAGS += -DUNCERTAINTIES_EXTERN_UDOUBLE2 -DUNCERTAINTIES_EXTERN_UDOUBLE
LDFLAGS = -lc++

########################################################################

tests = urealhpp iohpp stathpp mathhpp urealshpp functionshpp ureal2hpp distrhpp
tests += generic format ureals math indep
tests += mpfrformat mpfrmath mpfrureals
tests += cast2 generic2 deriv2 distr distre math2 checkmom
benchmarks = fillvector accumulate smallureals

programs = $(tests) $(benchmarks)

objects = $(patsubst %,%.cpp.o,$(programs)) uncertainties.cpp.o
benchmark_objects = $(patsubst %,%.cpp.o,$(benchmarks))
run_tests = $(patsubst %,run_%,$(tests))
run_benchmarks = $(patsubst %,run_%,$(benchmarks))

.PHONY: all test benchmark $(run_tests) $(run_benchmarks)
all : test benchmark
test: $(run_tests)
benchmark: $(benchmarks) $(run_benchmarks)

$(run_tests) $(run_benchmarks): run_%: %
	$<

$(programs): %: %.cpp.o uncertainties.cpp.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^
$(objects): %.cpp.o: %.cpp ../uncertainties/*.hpp ../uncertainties/internal/*.hpp
	$(CC) $(CFLAGS) -c -o $@ $<
$(benchmark_objects): CFLAGS += -O2
# I should do a separate compilation of uncertainties.cpp for the benchmarks.
mpfrformat mpfrmath mpfrureals: LDFLAGS += -lmpfr
