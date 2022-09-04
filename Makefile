# -*- Makefile -*-
include common.mk

SUBDIRS = unittest maxwell
SRCS    = chunk.cpp chunk3d.cpp chunkmap.cpp jsonio.cpp sfc.cpp
OBJS    = $(SRCS:%.cpp=%.o)
DEPS    = $(SRCS:%.cpp=%.d)

default: libnix.a

### test
testall: libnix.a
	make testall -C unittest
	make testall -C maxwell

### library
libnix.a: $(OBJS)
	$(AR) r $@ $(OBJS)

### clean
clean:
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		make clean -C $$dir; \
	done

cleanall:
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		make cleanall -C $$dir; \
	done

### dependency
-include $(DEPS)
