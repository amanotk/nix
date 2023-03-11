// -*- C++ -*-
#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#include "nix.hpp"
#include "tinyformat.hpp"
#include <nlohmann/json.hpp>

NIX_NAMESPACE_BEGIN

static constexpr int logger_flush_interval = 10;

///
/// @brief Simple Logger class
///
class Logger
{
private:
  int  thisrank; ///< MPI rank
  json config;   ///< configuration
  json content;  ///< log content

public:
  Logger(json& cfg)
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
    config  = cfg;
    content = {};
  }

  virtual void log(int curstep)
  {
    int  interval      = config.value("interval", 1);
    bool is_final_step = (curstep + 1) % interval == 0;

    // save file
    this->save(curstep, is_final_step);

    // clear
    if (is_final_step)
      content = {};
  }

  virtual void append(int curstep, const char* name, json& obj)
  {
    if (content.contains(name) == false) {
      // step
      content[name]["step"] = json::array();
      // data
      for (auto it = obj.begin(); it != obj.end(); ++it) {
        content[name][it.key()] = json::array();
      }
    }

    // step
    content[name]["step"].push_back(curstep);

    // data
    for (auto it = obj.begin(); it != obj.end(); ++it) {
      content[name][it.key()].push_back(it.value());
    }
  }

  virtual void save(int curstep, bool force = false)
  {
    int         interval     = config.value("interval", 1);
    int         step         = interval * (curstep / interval);
    std::string path         = config.value("path", ".");
    std::string prefix       = config.value("prefix", "log");
    std::string filename     = tfm::format("%s/%s_%s.json", path, prefix, format_step(step));
    float64     last_flushed = content.value("flushed", 0.0);
    float64     wclock       = wall_clock();

    if (thisrank == 0 && (force == true || (wclock - last_flushed > logger_flush_interval))) {
      content["flushed"] = wclock;

      std::ofstream ofs(filename);
      ofs << std::setw(2) << content << std::flush;
      ofs.close();
    }
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
