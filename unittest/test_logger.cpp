// -*- C++ -*-

#include "logger.hpp"

#include "catch.hpp"

using namespace nix;

class TestLogger : public Logger
{
public:
  using Logger::Logger;

  void test_config(std::string prefix, std::string path, int interval, float64 flush)
  {
    REQUIRE(config["prefix"] == prefix);
    REQUIRE(config["path"] == path);
    REQUIRE(config["interval"] == interval);
    REQUIRE(config["flush"] == flush);
  }

  void test_is_flush_required(float64 last_flushed, float64 flush, bool expected)
  {
    config["flush"]    = flush;
    this->last_flushed = last_flushed;
    REQUIRE(is_flush_required() == expected);
  }
};

TEST_CASE("test_config")
{
  SECTION("null")
  {
    json object = {};

    TestLogger logger(0, object);

    logger.test_config("log", ".", 100, 10.0);
  }

  SECTION("fully specified")
  {
    json object = {{"prefix", "foo"}, {"path", "bar"}, {"interval", 1}, {"flush", 1.0}};

    TestLogger logger(0, object);

    logger.test_config("foo", "bar", 1, 1.0);
  }

  SECTION("partially specified")
  {
    json object = {{"interval", 10}, {"flush", 60.0}};

    TestLogger logger(0, object);

    logger.test_config("log", ".", 10, 60.0);
  }
}

TEST_CASE("test_is_flush_required")
{
  TestLogger logger(0, {});

  SECTION("flush required")
  {
    logger.test_is_flush_required(wall_clock() - 10, 0.0, true);
  }

  SECTION("flush not required")
  {
    logger.test_is_flush_required(wall_clock() + 10, 100, false);
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
