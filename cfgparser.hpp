// -*- C++ -*-
#ifndef _CFGPARSER_HPP_
#define _CFGPARSER_HPP_

#include "debug.hpp"
#include "nix.hpp"

NIX_NAMESPACE_BEGIN

class CfgParser
{
protected:
  json root;

public:
  json get_root()
  {
    return root;
  }

  json get_application()
  {
    return root["application"];
  }

  json get_parameter()
  {
    return root["parameter"];
  }

  json get_diagnostic()
  {
    return root["diagnostic"];
  }

  bool parse_file(std::string filename)
  {
    std::ifstream ifs(filename.c_str());
    root = json::parse(ifs, nullptr, true, true);

    bool status = validate(root);

    if (status == true) {
      // make sure that the option section exists
      if (root["application"]["option"].is_null() == true) {
        root["application"]["option"] = {};
      }
    }

    return status;
  }

  void overwrite(json& object)
  {
    assert(validate(object) == true);

    root = object;
  }

  virtual bool validate(json& object)
  {
    bool status = true;

    status = status & check_mandatory_sections(object);

    if (object["parameter"].is_null() == false) {
      status = status & check_dimensions(object["parameter"]);
    } else {
      status = status & false;
    }

    return status;
  }

  virtual bool check_mandatory_sections(json& object)
  {
    bool status = true;

    std::vector<std::string> mandatory_sections = {"application", "diagnostic", "parameter"};

    for (auto section : mandatory_sections) {
      if (object[section].is_null()) {
        ERROR << tfm::format("Configuration misses `%s` section", section);
        status = false;
      }
    }

    return status;
  }

  virtual bool check_dimensions(json& parameter)
  {
    int  nx     = parameter.value("Nx", 1);
    int  ny     = parameter.value("Ny", 1);
    int  nz     = parameter.value("Nz", 1);
    int  cx     = parameter.value("Cx", 1);
    int  cy     = parameter.value("Cy", 1);
    int  cz     = parameter.value("Cz", 1);
    bool status = (nz % cz == 0) && (ny % cy == 0) && (nx % cx == 0);

    if (status == false) {
      ERROR << tfm::format("Number of grid must be divisible by number of chunk");
      ERROR << tfm::format("Nx, Ny, Nz = [%4d, %4d, %4d]", nx, ny, nz);
      ERROR << tfm::format("Cx, Cy, Cz = [%4d, %4d, %4d]", cx, cy, cz);
    }

    return status;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
