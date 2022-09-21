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

  load.resize(1);
  reset_load();
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

DEFINE_MEMBER(void, reset_load)()
{
  load.assign(load.size(), 0.0);
}

DEFINE_MEMBER(std::vector<float64>, get_load)()
{
  return load;
}

DEFINE_MEMBER(float64, get_total_load)()
{
  return std::accumulate(load.begin(), load.end(), 0.0);
}

DEFINE_MEMBER(int, pack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += memcpy_count(buffer, &myid, sizeof(int), count, 0);
  count += memcpy_count(buffer, &nbid[0], nbsize * sizeof(int), count, 0);
  count += memcpy_count(buffer, &nbrank[0], nbsize * sizeof(int), count, 0);

  // load
  {
    int size = load.size();
    count += memcpy_count(buffer, &size, sizeof(int), count, 0);
    count += memcpy_count(buffer, load.data(), sizeof(float64) * size, count, 0);
  }

  return count;
}

DEFINE_MEMBER(int, unpack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += memcpy_count(&myid, buffer, sizeof(int), 0, count);
  count += memcpy_count(&nbid[0], buffer, nbsize * sizeof(int), 0, count);
  count += memcpy_count(&nbrank[0], buffer, nbsize * sizeof(int), 0, count);

  // load
  {
    int size = 0;
    count += memcpy_count(&size, buffer, sizeof(int), 0, count);
    load.resize(size);
    count += memcpy_count(load.data(), buffer, sizeof(float64) * size, 0, count);
  }

  return count;
}

DEFINE_MEMBER(int, pack_diagnostic)(const int mode, void *buffer, const int address)
{
  return 0;
}

DEFINE_MEMBER(bool, set_boundary_query)(const int mode)
{
  return true;
}

DEFINE_MEMBER(void, set_boundary_physical)(const int mode)
{
}

DEFINE_MEMBER(void, set_boundary_begin)(const int mode)
{
}

DEFINE_MEMBER(void, set_boundary_end)(const int mode)
{
}

// explicit instantiation
template class Chunk<1>;
template class Chunk<2>;
template class Chunk<3>;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
