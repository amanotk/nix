# -*- Makefile -*-
include ../common.mk

.PHONY: default clean cleanall

### target
LIBS    := $(BASEDIR)/libnix.a
SRCS    := test_serial.cpp test_parallel.cpp test_jsonio1.cpp test_jsonio2.cpp \
	test_sfc.cpp test_chunk.cpp test_chunkmap.cpp test_particle.cpp \
	test_application.cpp

### OBJS/DEPS
EXES    := test_jsonio1 test_jsonio2 test_sfc test_chunk test_chunkmap \
	test_particle test_application test_sendrecv
OBJS    := $(SRCS:%.cpp=%.o)
DEPS    := $(SRCS:%.cpp=%.d)

### default
default: $(EXES)

# individual executables
test_jsonio1: test_serial.o test_jsonio1.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lnix

test_jsonio2: test_parallel.o test_jsonio2.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lnix

test_sfc: test_serial.o test_sfc.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lnix

test_chunk: test_serial.o test_chunk.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lnix

test_chunkmap: test_serial.o test_chunkmap.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lnix

test_particle: test_serial.o test_particle.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lnix

test_application: test_serial.o test_application.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lnix

test_sendrecv: test_serial.o test_sendrecv.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../ -lnix

### clean
clean:
	rm -f *.o *.d *.out *.stdout *.stderr

cleanall:
	rm -f *.o *.d *.out *.stdout *.stderr
	rm -f $(EXES)

### dependency
-include $(DEPS)
