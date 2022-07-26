# -*- Makefile -*-
include ../common.mk

SUBDIRS =
SRCS    = chunk.cpp chunk3d.cpp chunkmap.cpp jsonio.cpp sfc.cpp
OBJS    = $(SRCS:%.cpp=%.o)
DEPS    = $(SRCS:%.cpp=%.d)

default: libnix.a

### library
libnix.a: $(OBJS)
	$(AR) r $@ $(OBJS)

### clean
clean :
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done

### dependency
-include $(DEPS)
