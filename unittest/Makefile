# -*- Makefile -*-
.PHONY: default clean cleanall

### target
LIBS    :=
SRCS    := application.cpp patchmap.cpp patch.cpp hilbert.cpp

### OBJS/DEPS
EXES    := $(SRCS:%.cpp=%)
OBJS    := $(SRCS:%.cpp=%.o)
DEPS    := $(SRCS:%.cpp=%.d)


### include system specific configuration
include Makefile.inc


### default
default: $(EXES)

# individual executables
application: application.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

patchmap: patchmap.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

patch: patch.o
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
