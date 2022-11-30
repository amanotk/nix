# -*- Makefile -*-
include common.mk

SUBDIRS = unittest maxwell
SRCS    = balancer.cpp chunk.cpp chunkmap.cpp jsonio.cpp sfc.cpp
OBJS    = $(SRCS:%.cpp=%.o)
DEPS    = $(SRCS:%.cpp=%.d)

default: libnix.a

### library
libnix.a: $(OBJS)
	$(AR) r $@ $(OBJS)

### test
testall: libnix.a
	$(MAKE) testall -C unittest
	$(MAKE) testall -C maxwell

### clean
clean:
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done

cleanall:
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) cleanall -C $$dir; \
	done

### dependency
-include $(DEPS)
