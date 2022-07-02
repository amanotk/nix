# -*- Makefile -*-
.PHONY: default clean cleanall

### target
LIBS    :=
SRCS    := application.cpp chunkmap.cpp test_chunk.cpp hilbert.cpp

### OBJS/DEPS
EXES    := $(SRCS:%.cpp=%)
OBJS    := $(SRCS:%.cpp=%.o)
DEPS    := $(SRCS:%.cpp=%.d)


### include system specific configuration
include Makefile.inc


### default
#default: $(EXES)
default: test_chunk

# individual executables
application: application.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

chunkmap: chunkmap.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

test_chunk: test_chunk.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lpk3

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
