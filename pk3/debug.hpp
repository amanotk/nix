// -*- C++ -*-
#ifndef _DEBUG_HPP_
#define _DEBUG_HPP_

#include "tinyformat.hpp"
#include <cstring>
#include <iomanip>
#include <iostream>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

//
// error printing
//
#define ERRORPRINT(fmt, ...)                                                                       \
  tfm::format(std::cerr, "<<< ERROR BEGIN (File = %s, Line = %05d) >>>\n", __FILENAME__,           \
              __LINE__);                                                                           \
  tfm::format(std::cerr, fmt, ##__VA_ARGS__);                                                      \
  tfm::format(std::cerr, "<<< ERROR END >>>\n");                                                   \
  std::cerr.flush();

//
// debug printing
//
#ifdef _DEBUGPRINT

#define DEBUGPRINT(out, fmt, ...)                                                                  \
  tfm::format(out, "DEBUG (File = %s, Line = %05d): ", __FILENAME__, __LINE__);                    \
  tfm::format(out, fmt, ##__VA_ARGS__);                                                            \
  out.flush();

#else

#define DEBUGPRINT(out, fmt, ...)

#endif

//
// log printing
//
#if !defined(_LOG)
// no logging
#define LOGPRINT0(out, fmt, ...)
#define LOGPRINT1(out, fmt, ...)
#define LOGPRINT2(out, fmt, ...)

#elif _LOG >= 0
// logging level = 0
#define LOGPRINT0(out, fmt, ...) tfm::format(out, "Log: " fmt, ##__VA_ARGS__)
#define LOGPRINT1(out, fmt, ...)
#define LOGPRINT2(out, fmt, ...)

#elif _LOG >= 1
// logging level 1
#define LOGPRINT0(out, fmt, ...) tfm::format(out, "Log: " fmt, ##__VA_ARGS__)
#define LOGPRINT1(out, fmt, ...) tfm::format(out, "Log: " fmt, ##__VA_ARGS__)
#define LOGPRINT2(out, fmt, ...)

#elif _LOG >= 2
// logggin level 2
#define LOGPRINT0(out, fmt, ...) tfm::format(out, "Log: " fmt, ##__VA_ARGS__)
#define LOGPRINT1(out, fmt, ...) tfm::format(out, "Log: " fmt, ##__VA_ARGS__)
#define LOGPRINT2(out, fmt, ...) tfm::format(out, "Log: " fmt, ##__VA_ARGS__)

#endif

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
