# -*- Makefile -*-

# include path
CXX_INCLUDE = -I../ -I../base -I../mdarray

# compilers
CXX		= mpicxx -g
CXXFLAGS	= -std=c++14 -UNDEBUG -O3 -MMD $(CXX_INCLUDE)
LDFLAGS	=


%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $<

