// -*- C++ -*-
#ifndef _CMDPARSER_HPP_
#define _CMDPARSER_HPP_

///
/// @brief Common Command-line Parser
///
/// $Id: cmdparser.hpp,v 1a555f91e3aa 2015/11/15 15:03:56 amano $
///
#include "cmdline.hpp"
#include "common.hpp"

///
/// Common command-line Parser class
///
class CmdParser : public cmdline::parser
{
public:
  CmdParser(std::string config="default.cfg")
  {
    add_default(config);
  }

  void add_default(std::string config)
  {
    const float64 oneday = 60*60*24;
    const float64 tmax   = common::HUGEVAL;

    this->add<std::string>("config", 'c',
                           "configuration file", true, config);
    this->add<std::string>("load", 'l',
                           "load file for restart", false, "");
    this->add<std::string>("save", 's',
                           "save file for restart", false, "");
    this->add<float64>("tmax", 't',
                       "maximum physical time in simulation unit",
                       false, tmax);
    this->add<float64>("emax", 'e',
                       "maximum elased time limit [sec]",
                       false, oneday);
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
