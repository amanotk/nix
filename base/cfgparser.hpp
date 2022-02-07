// -*- C++ -*-
#ifndef _CFGPARSER_HPP_
#define _CFGPARSER_HPP_

///
/// @brief Simple Configuration File Parser
///
/// $Id: configparser.hpp,v d2349ecddf6d 2015/11/16 03:32:29 amano $
///
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <regex.h>

//
// Simple Configuration File Parser
//
class CfgParser
{
private:
  std::string m_filename;
  std::map<std::string, std::string> m_pair;

  // read config file and store (key, val) to pair to m_pair
  void read_file(const char* filename)
  {
    static const int max_num_match = 5;
    static const char pattern[] =
      "[[:space:]]*([A-Za-z0-9_/\\.\\+\\-]+)[[:space:]]*="
      "[[:space:]]*([A-Za-z0-9_/\\.\\+\\-]+)[[:space:]]*";

    int status;
    std::string line;

    // open file
    std::ifstream ifs(filename);

    if( !ifs ) {
      std::cerr << "Error: Could not open file : " << filename << std::endl;
    }

    // prepare for regex match
    regex_t buf;
    regmatch_t match[max_num_match];

    status = regcomp(&buf, pattern, REG_EXTENDED);
    if(status != 0) {
      std::cerr << "Error: regcomp failed" << std::endl;
      exit(-1);
    }

    // parse for each line
    while( std::getline(ifs, line) ) {
      line = discard_comment(line);
      parse_line(line, &buf, match, max_num_match);
    }

    // close
    ifs.close();

    // free buffer
    regfree(&buf);
  }

  // discard comment (after "#" for each line)
  std::string discard_comment(std::string &str)
  {
    std::string::size_type last = str.find_first_of("#");
    return str.substr(0, last);
  }

  // perform regex match and store (key, val) pair
  void parse_line(std::string &str, regex_t *buf, regmatch_t *match, size_t n)
  {
    int status = regexec(buf, str.c_str(), n, match, 0);

    // ignore unrecognized line
    if(status != 0 || n < 3) {
      return;
    }

    // insert into m_pair
    std::string key = trim(str.substr(match[1].rm_so, match[1].rm_eo));
    std::string val = trim(str.substr(match[2].rm_so, match[2].rm_eo));
    m_pair.insert(std::make_pair(key, val));
  }

  // trim white space
  std::string trim(std::string str)
  {
    std::string::size_type first = str.find_first_not_of(" ");
    std::string::size_type last  = str.find_last_not_of (" ");

    return str.substr(first, last+1);
  }

  // template declaration
  template <class T> T type_cast(std::string &str);

public:
  // default constructor; do nothing
  CfgParser()
  {
  }

  // constructor: read given file
  CfgParser(const char* filename)
  {
    read(filename);
  }

  // read given file
  void read(const char* filename)
  {
    m_filename = std::string(filename);
    read_file(filename);
  }

  // get value for given key
  template <class T>
  T getAs(const char* key)
  {
    if( m_pair.find(std::string(key)) == m_pair.end() ) {
      std::cerr << "Error: cannot find key : " << key << std::endl;
      exit(-1);
    }

    return type_cast<T>(m_pair[key]);
  }

  // print all (key, val) pairs
  void print()
  {
    std::map<std::string, std::string>::iterator it = m_pair.begin();

    std::cout << std::setw(20) << std::left << "key"
              << std::setw( 2) << ":"
              << std::setw(20) << std::left << "val" << std::endl;
    std::cout << std::setw(42) << std::setfill('-') << "" << std::endl;
    std::cout << std::setfill(' ');
    while( it != m_pair.end() ) {
      std::cout << std::setw(20) << std::left << (*it).first
                << std::setw( 2) << ":"
                << std::setw(20) << std::left << (*it).second << std::endl;
      ++it;
    }
  }

  // return file content as string
  std::string get_content()
  {
    std::ifstream     ifs(m_filename.c_str());
    std::stringstream ss;

    if( !ifs ) {
      std::cerr << "Error: Could not open file : " << m_filename << std::endl;
    }

    // copy
    ss << ifs.rdbuf();

    // close
    ifs.close();

    return ss.str();
  }
};

// for string
template <>
std::string CfgParser::type_cast<std::string>(std::string &str)
{
  return str;
}

// for int
template <>
int CfgParser::type_cast<int>(std::string &str)
{
  return std::atoi(str.c_str());
}

// for float
template <>
float CfgParser::type_cast<float>(std::string &str)
{
  return std::atof(str.c_str());
}

// for double
template <>
double CfgParser::type_cast<double>(std::string &str)
{
  return std::atof(str.c_str());
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
