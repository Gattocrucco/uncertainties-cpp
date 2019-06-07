CC = c++
CFLAGS = -std=c++11 -g -I.. -I/usr/local/include/eigen3 -DUNCERTAINTIES_EXTERN_UDOUBLE2
LDFLAGS = -lc++

########################################################################

tests = urealhpp iohpp stathpp mathhpp urealshpp functionshpp ureal2hpp distrhpp
tests += generic format ureals math
tests += mpfrformat mpfrmath mpfrureals
tests += cast2 generic2 deriv2 distr math2 checkmom
benchmarks = fillvector accumulate smallureals

programs = $(tests) $(benchmarks)

objects = $(patsubst %,%.cpp.o,$(programs))
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
uncertainties.cpp.o: uncertainties.cpp ../uncertainties/impl.hpp ../uncertainties/core.hpp
	$(CC) $(CFLAGS) -c -o $@ $<
$(objects): %.cpp.o: %.cpp ../uncertainties/*.hpp ../uncertainties/internal/*.hpp
	$(CC) $(CFLAGS) -c -o $@ $<
$(benchmark_objects): CFLAGS += -O2
mpfrformat mpfrmath mpfrureals: LDFLAGS += -lmpfr
