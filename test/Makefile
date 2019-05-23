CPP = c++
CFLAGS = -std=c++11 -g -I..
LDFLAGS = -lc++

tests = urealhpp iohpp stathpp mathhpp urealshpp functionshpp ureal2hpp distrhpp generic format ureals math mpfrformat mpfrmath mpfrureals cast2 generic2 deriv2
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
	$(CPP) $(CFLAGS) $(LDFLAGS) -o $@ $^
uncertainties.cpp.o: uncertainties.cpp ../uncertainties/impl.hpp ../uncertainties/core.hpp
	$(CPP) $(CFLAGS) -c -o $@ $<
$(objects): %.cpp.o: %.cpp ../uncertainties/*.hpp ../uncertainties/internal/*.hpp
	$(CPP) $(CFLAGS) -c -o $@ $<
urealshpp.cpp.o ureals.cpp.o smallureals.cpp.o mpfrureals.cpp.o: CFLAGS += -I/usr/local/include/eigen3
$(benchmark_objects): CFLAGS += -O2
mpfrformat mpfrmath mpfrureals: LDFLAGS += -lmpfr
