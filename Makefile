# -*- Makefile -*-
include common.mk

SUBDIRS =
SRCS   = jsonio.o sfc.o chunk.o chunkmap.o
OBJS   = $(SRCS:%.cpp=%.o)

default: libpk3.a

libpk3.a: $(OBJS)
	$(AR) r libpk3.a $(OBJS)

clean :
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done


# dependency
chunk.o: chunk.cpp chunk.hpp
