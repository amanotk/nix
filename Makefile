# -*- Makefile -*-
include common.mk

SUBDIRS =
SRCS   = jsonio.o chunk.o chunkmap.o application.o
OBJS   = $(SRCS:%.cpp=%.o)

default: $(OBJS)

clean :
	rm -f $(OBJS)
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done
