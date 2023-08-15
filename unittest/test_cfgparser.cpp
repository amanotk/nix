// -*- C++ -*-

#include "cfgparser.hpp"

#include "catch.hpp"

using namespace nix;

TEST_CASE("Basic")
{
  CfgParser parser;
}

TEST_CASE("check_mandatory_sections")
{
  CfgParser parser;

  SECTION("successful")
  {
    json root = {{"application", {}}, {"diagnostic", {}}, {"parameter", {}}};

    REQUIRE(parser.check_mandatory_sections(root) == true);
  }

  SECTION("application is missing")
  {
    json root = {{"diagnostic", {}}, {"parameter", {}}};

    REQUIRE(parser.check_mandatory_sections(root) == false);
  }

  SECTION("diagnostic is missing")
  {
    json root = {{"application", {}}, {"parameter", {}}};

    REQUIRE(parser.check_mandatory_sections(root) == false);
  }

  SECTION("parameter is missing")
  {
    json root = {{"application", {}}, {"diagnostic", {}}};

    REQUIRE(parser.check_mandatory_sections(root) == false);
  }
}

TEST_CASE("check_dimensions")
{
  CfgParser parser;

  SECTION("successful")
  {
    json parameter = json::array({
        // 0
        {
            {"Nx", 2},
            {"Ny", 2},
            {"Nz", 2},
            {"Cx", 1},
            {"Cy", 1},
            {"Cz", 1},
        },
        // 1
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 2},
            {"Cy", 2},
            {"Cz", 2},
        },
        // 2
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 1},
            {"Cy", 1},
            {"Cz", 8},
        },
    });

    REQUIRE(parser.check_dimensions(parameter[0]) == true);
    REQUIRE(parser.check_dimensions(parameter[1]) == true);
    REQUIRE(parser.check_dimensions(parameter[2]) == true);
  }

  SECTION("failure")
  {
    json parameter = json::array({
        // 0
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 3},
            {"Cy", 1},
            {"Cz", 1},
        },
        // 1
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 1},
            {"Cy", 3},
            {"Cz", 1},
        },
        // 2
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 1},
            {"Cy", 3},
            {"Cz", 1},
        },
    });

    REQUIRE(parser.check_dimensions(parameter[0]) == false);
    REQUIRE(parser.check_dimensions(parameter[1]) == false);
    REQUIRE(parser.check_dimensions(parameter[2]) == false);
  }
}

TEST_CASE("parse_file")
{
  CfgParser parser;

  std::string filename = "test_cfgparser.json";
  std::string content  = R"(
  {
    "application": {
      "rebalance": {},
      "log": {}
    },
    "diagnostic": [
      {},
      {},
      {}
    ],
    "parameter": {
        "Nx": 16,
        "Ny": 16,
        "Nz": 16,
        "Cx": 4,
        "Cy": 4,
        "Cz": 4,
        "delt": 1.0,
        "delh": 1.0
    }
  }
  )";

  // save
  {
    std::ofstream ofs(filename);
    ofs << content;
  }

  REQUIRE(parser.parse_file(filename) == true);;

  // cleanup
  std::filesystem::remove(filename);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
