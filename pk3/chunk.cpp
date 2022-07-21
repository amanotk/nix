// -*- C++ -*-
#include "chunk.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int N>                                                                                 \
  type BaseChunk<N>::name

DEFINE_MEMBER(int, pack_base)(const int mode, void *buffer)
{
  using common::memcpy_count;

  int   count = 0;
  char *ptr   = static_cast<char *>(buffer);

  switch (mode) {
  case PackAll:
    count += memcpy_count(&ptr[count], &myid, sizeof(int), false);
    count += memcpy_count(&ptr[count], &nbid[0], nbsize * sizeof(int), false);
    count += memcpy_count(&ptr[count], &nbrank[0], nbsize * sizeof(int), false);
    count += memcpy_count(&ptr[count], &load, sizeof(float64), false);
    break;
  case PackAllQuery:
    count += memcpy_count(&ptr[count], &myid, sizeof(int), true);
    count += memcpy_count(&ptr[count], &nbid[0], nbsize * sizeof(int), true);
    count += memcpy_count(&ptr[count], &nbrank[0], nbsize * sizeof(int), true);
    count += memcpy_count(&ptr[count], &load, sizeof(float64), true);
    break;
  default:
    count = -1;
    break;
  }

  return count;
}

DEFINE_MEMBER(int, unpack_base)(const int mode, void *buffer)
{
  using common::memcpy_count;

  int   count = 0;
  char *ptr   = static_cast<char *>(buffer);

  switch (mode) {
  case PackAll:
    count += memcpy_count(&myid, &ptr[count], sizeof(int), false);
    count += memcpy_count(&nbid[0], &ptr[count], nbsize * sizeof(int), false);
    count += memcpy_count(&nbrank[0], &ptr[count], nbsize * sizeof(int), false);
    count += memcpy_count(&load, &ptr[count], sizeof(float64), false);
    break;
  case PackAllQuery:
    count += memcpy_count(&myid, &ptr[count], sizeof(int), true);
    count += memcpy_count(&nbid[0], &ptr[count], nbsize * sizeof(int), true);
    count += memcpy_count(&nbrank[0], &ptr[count], nbsize * sizeof(int), true);
    count += memcpy_count(&load, &ptr[count], sizeof(float64), true);
    break;
  default:
    count = -1;
    break;
  }

  return count;
}

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

  // set shape
  for (int i = 0; i < N; i++) {
    shape[i]  = dims[i];
    offset[i] = 0;
  }

  initialize_load();
}

DEFINE_MEMBER(, BaseChunk)()
{
  const int dims[N] = {};
  initialize(dims, 0);
}

DEFINE_MEMBER(, BaseChunk)(const int dims[N], const int id)
{
  initialize(dims, id);
}

DEFINE_MEMBER(, ~BaseChunk)()
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

DEFINE_MEMBER(int, pack)(const int mode, void *buffer)
{
  return pack_base(mode, buffer);
}

DEFINE_MEMBER(int, unpack)(const int mode, void *buffer)
{
  return unpack_base(mode, buffer);
}

template class BaseChunk<1>;
template class BaseChunk<2>;
template class BaseChunk<3>;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
