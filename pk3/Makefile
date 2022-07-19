# -*- Makefile -*-
include ../common.mk

SUBDIRS =
SRCS    = chunk.cpp chunkmap.cpp jsonio.cpp sfc.cpp
OBJS    = $(SRCS:%.cpp=%.o)
DEPS    = $(SRCS:%.cpp=%.d)

default: libpk3.a

### library
libpk3.a: $(OBJS)
	$(AR) r libpk3.a $(OBJS)

### clean
clean :
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done

### dependency
-include $(DEPS)
