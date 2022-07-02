# -*- Makefile -*-
include common.mk

SUBDIRS =
SRCS   = jsonio.o chunk.o chunkmap.o application.o
OBJS   = $(SRCS:%.cpp=%.o)

default: $(OBJS)
	$(AR) r libpk3.a $(OBJS)

clean :
	rm -f $(OBJS) *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done


# dependency
chunk.o: chunk.cpp chunk.hpp
