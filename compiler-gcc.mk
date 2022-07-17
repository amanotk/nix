# -*- Makefile -*-

INCLUDE_FLAGS=$(shell pkg-config --cflags xtensor xtl xsimd)

# compilers and arguments
AR       = ar
CXX      = mpicxx
CXXFLAGS = -std=c++14 -MMD $(INCLUDE_FLAGS)
LDFLAGS  =
