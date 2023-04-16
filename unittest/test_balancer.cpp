// -*- C++ -*-

#include "balancer.hpp"
#include <iostream>

#include "catch.hpp"

using namespace nix;

//
// for assignment with binary search
//
TEST_CASE("AssignBinarySearch")
{
  const int            Nr = 10;
  const int            Nc = 20;
  std::vector<float64> load(Nr * Nc);
  std::vector<int>     boundary(Nr + 1);

  std::unique_ptr<Balancer> balancer = std::make_unique<Balancer>();

  SECTION("HomogeneousLoad")
  {
    // initialize
    std::fill(load.begin(), load.end(), 1.0);

    // test
    balancer->assign(load, boundary, true);

    // check
    REQUIRE(balancer->validate_boundary(Nr * Nc, boundary) == true);
    for (int i = 0; i < Nr; i++) {
      REQUIRE(boundary[i] == i * Nc);
    }
  }

  SECTION("InhomogeneousLoad")
  {
    std::random_device                      seed;
    std::mt19937                            engine(seed());
    std::uniform_real_distribution<float64> dist(0.5, 1.5);

    std::vector<float64> cumload(Nr * Nc + 1);

    // initialize
    for (int i = 0; i < Nr * Nc; i++) {
      load[i] = dist(engine);
    }

    // calculate cumulative sum of load
    cumload[0] = 0;
    for (int i = 0; i < Nr * Nc; i++) {
      cumload[i + 1] = cumload[i] + load[i];
    }

    // test
    balancer->assign(load, boundary, true);

    // check
    REQUIRE(balancer->validate_boundary(Nr * Nc, boundary) == true);
    for (int i = 1; i < Nr; i++) {
      int     index1   = boundary[i] - 1;
      int     index2   = boundary[i];
      float64 bestload = i * cumload[Nr * Nc] / Nr;

      REQUIRE((cumload[index1] <= i * bestload && cumload[index2] > i * cumload[Nr * Nc] / Nr));
    }
  }
}

//
// for assignment with default algorithm
//
TEST_CASE("Assign")
{
  const int            Nr = 10;
  const int            Nc = 20;
  std::vector<float64> load(Nr * Nc);
  std::vector<int>     boundary(Nr + 1);
  std::vector<float64> cumload(Nr * Nc + 1);

  std::random_device                      seed;
  std::mt19937                            engine(seed());
  std::uniform_real_distribution<float64> dist(0.5, 1.5);

  std::unique_ptr<Balancer> balancer = std::make_unique<Balancer>();

  // initialize
  for (int i = 0; i < Nr * Nc; i++) {
    load[i] = dist(engine);
  }

  for (int i = 0; i < Nr + 1; i++) {
    boundary[i] = Nc * i;
  }

  // calculate cumulative sum
  cumload[0] = 0;
  for (int i = 0; i < Nr * Nc; i++) {
    cumload[i + 1] = cumload[i] + load[i];
  }

  // test
  balancer->assign(load, boundary);

  // check
  REQUIRE(balancer->validate_boundary(Nr * Nc, boundary) == true);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
