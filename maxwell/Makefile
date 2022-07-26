# -*- Makefile -*-
include ../common.mk

.PHONY: default clean cleanall

### target
LIBS    := $(BASEDIR)/libnix.a
SRCS    := fdtd.cpp maxwell.cpp main.cpp

### OBJS/DEPS
EXES    := main.out
OBJS    := $(SRCS:%.cpp=%.o)
DEPS    := $(SRCS:%.cpp=%.d)

### default
default: $(EXES)

main.out: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L$(BASEDIR) -lnix

### clean
clean:
	rm -f *.o *.d *.out *.stdout *.stderr

cleanall:
	rm -f *.o *.d *.out *.stdout *.stderr
	rm -f $(EXES)


### dependency
-include $(DEPS)
