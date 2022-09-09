// -*- C++ -*-
#include "chunk.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int N>                                                                                 \
  type Chunk<N>::name

DEFINE_MEMBER(void, initialize)(const int dims[N], const int id)
{
  int shift;

  // endian flag
  if (common::get_endian_flag() == 1) {
    shift = 31 - DIRTAG_BIT;
  } else {
    shift = 0;
  }

  // set directional message tag
  tagmask = (DIRTAG_SIZE - 1) << shift;
  for (int i = 0; i < DIRTAG_SIZE; i++) {
    dirtag[i] = i << shift;
  }

  // set ID
  set_id(id);

  // set dimensions
  for (int i = 0; i < N; i++) {
    this->dims[i] = dims[i];
  }

  initialize_load();
}

DEFINE_MEMBER(, Chunk)()
{
  const int dims[N] = {};
  initialize(dims, 0);
}

DEFINE_MEMBER(, Chunk)(const int dims[N], const int id)
{
  initialize(dims, id);
}

DEFINE_MEMBER(, ~Chunk)()
{
}

DEFINE_MEMBER(void, initialize_load)()
{
  static std::random_device                      rd;
  static std::mt19937                            mt(rd());
  static std::uniform_real_distribution<float64> rand(0.75, +1.25);

  load = rand(mt);
}

DEFINE_MEMBER(float64, get_load)()
{
  static std::random_device                      rd;
  static std::mt19937                            mt(rd());
  static std::uniform_real_distribution<float64> rand(0.75, +1.25);

  load = rand(mt);
  return load;
}

DEFINE_MEMBER(int, pack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += memcpy_count(buffer, &myid, sizeof(int), count, 0);
  count += memcpy_count(buffer, &nbid[0], nbsize * sizeof(int), count, 0);
  count += memcpy_count(buffer, &nbrank[0], nbsize * sizeof(int), count, 0);
  count += memcpy_count(buffer, &load, sizeof(float64), count, 0);

  return count;
}

DEFINE_MEMBER(int, unpack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += memcpy_count(&myid, buffer, sizeof(int), 0, count);
  count += memcpy_count(&nbid[0], buffer, nbsize * sizeof(int), 0, count);
  count += memcpy_count(&nbrank[0], buffer, nbsize * sizeof(int), 0, count);
  count += memcpy_count(&load, buffer, sizeof(float64), 0, count);

  return count;
}

template class Chunk<1>;
template class Chunk<2>;
template class Chunk<3>;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
