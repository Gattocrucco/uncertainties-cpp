CPP = c++
CFLAGS = -std=c++11 -g -I..
LDFLAGS = -lc++

tests = generic urealhpp iohpp stathpp  mathhpp format urealshpp ureals

objects = $(patsubst %,%.cpp.o,$(tests))
run_tests = $(patsubst %,run_%,$(tests))

.PHONY: test $(run_tests)
test: $(run_tests)

$(run_tests): run_%: %
	$<

$(tests): %: %.cpp.o uncertainties.cpp.o
	$(CPP) $(CFLAGS) $(LDFLAGS) -o $@ $^
uncertainties.cpp.o: uncertainties.cpp ../uncertainties/*.hpp
	$(CPP) $(CFLAGS) -c -o $@ $<
$(objects): %.cpp.o: %.cpp ../uncertainties/*.hpp
	$(CPP) $(CFLAGS) -c -o $@ $<
urealshpp.cpp.o ureals.cpp.o: CFLAGS += -I/usr/local/include/eigen3
