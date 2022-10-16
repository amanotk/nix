# -*- Makefile -*-

# base directory
BASEDIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# include compilers
include $(BASEDIR)/compiler.mk

# add options
CXXFLAGS += -I$(BASEDIR) -I$(BASEDIR)/thirdparty

# default
.PHONY : all
.PHONY : clean

.SUFFIXES :
.SUFFIXES : .o .cpp

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@
