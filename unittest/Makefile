# -*- Makefile -*-
include ../common.mk

.PHONY: default clean cleanall

### target
LIBS    := $(BASEDIR)/libpk3.a
SRCS    := test_serial.cpp test_parallel.cpp test_jsonio1.cpp test_jsonio2.cpp test_sfc.cpp test_chunk.cpp test_chunkmap.cpp


### OBJS/DEPS
EXES    := $(SRCS:%.cpp=%)
OBJS    := $(SRCS:%.cpp=%.o)
DEPS    := $(SRCS:%.cpp=%.d)


### default
#default: $(EXES)
default: test_jsonio1 test_jsonio2 test_sfc test_chunk test_chunkmap

# individual executables
test_jsonio1: $(LIBS) test_serial.o test_jsonio1.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lpk3

test_jsonio2: $(LIBS) test_parallel.o test_jsonio2.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lpk3

test_sfc: $(LIBS) test_serial.o test_sfc.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lpk3

test_chunk: $(LIBS) test_serial.o test_chunk.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lpk3

test_chunkmap: $(LIBS) test_serial.o test_chunkmap.o
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
