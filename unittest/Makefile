# -*- Makefile -*-
include ../common.mk

.PHONY: default clean cleanall

### target
LIBS    :=
SRCS    := test_serial.cpp test_parallel.cpp test_chunk.cpp test_jsonio1.cpp
#SRCS    := application.cpp chunkmap.cpp test_serial.cpp test_chunk.cpp test_jsonio.cpp hilbert.cpp

### OBJS/DEPS
EXES    := $(SRCS:%.cpp=%)
OBJS    := $(SRCS:%.cpp=%.o)
DEPS    := $(SRCS:%.cpp=%.d)


### default
#default: $(EXES)
default: test_chunk test_jsonio1 test_jsonio2

# individual executables
test_chunk: test_serial.o test_chunk.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lpk3

test_jsonio1: test_serial.o test_jsonio1.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lpk3

test_jsonio2: test_parallel.o test_jsonio2.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lpk3

application: application.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

chunkmap: chunkmap.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

hilbert: hilbert.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

### clean
clean:
	rm -f *.o *.d *.out

cleanall:
	rm -f *.o *.d *.out *.dat
	rm -f $(EXES)


### dependency
-include $(DEPS)
