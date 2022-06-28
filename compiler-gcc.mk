# -*- Makefile -*-

# compilers and arguments
AR       = ar
CXX      = mpicxx
CXXFLAGS = -std=c++14 -UNDEBUG -MMD -I$(BASEDIR)/mdarray -I$(BASEDIR)/utils
LDFLAGS  =
