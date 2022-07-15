// -*- C++ -*-
#include "chunk.hpp"

#define DEFINE_MEMBER(type, name)                                              \
  template <int N>                                                             \
  type BaseChunk<N>::name

DEFINE_MEMBER(int, pack_base)(const int flag, char *buffer)
{
  char *buffer0 = buffer;

  std::memcpy(buffer, &myid, sizeof(int));
  buffer += sizeof(int);

  std::memcpy(buffer, &nbid[0], sizeof(int) * nbsize);
  buffer += sizeof(int) * nbsize;

  std::memcpy(buffer, &nbrank[0], sizeof(int) * nbsize);
  buffer += sizeof(int) * nbsize;

  std::memcpy(buffer, &load, sizeof(float64));
  buffer += sizeof(float64);

  return buffer - buffer0;
}

DEFINE_MEMBER(int, unpack_base)(const int flag, char *buffer)
{
  char *buffer0 = buffer;

  std::memcpy(&myid, buffer, sizeof(int));
  buffer += sizeof(int);

  std::memcpy(&nbid[0], buffer, sizeof(int) * nbsize);
  buffer += sizeof(int) * nbsize;

  std::memcpy(&nbrank[0], buffer, sizeof(int) * nbsize);
  buffer += sizeof(int) * nbsize;

  std::memcpy(&load, buffer, sizeof(float64));
  buffer += sizeof(float64);

  return buffer - buffer0;
}


DEFINE_MEMBER(void, initialize)(const int dims[N], const int id)
{
  int shift;

  // endian flag
  if (get_endian_flag() == 1) {
    shift = 32 - DIRTAG_BIT;
  } else {
    shift = 0;
  }

  // set directional message tag
  tagmask = (DIRTAG_SIZE-1) << shift;
  for (int i = 0; i < DIRTAG_SIZE; i++) {
    dirtag[i] = i << shift;
  }

  // set ID
  set_id(id);

  // set shape
  for(int i=0; i < N ;i++) {
    shape[i] = dims[i];
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
  static std::random_device rd;
  static std::mt19937 mt(rd());
  static std::uniform_real_distribution<float64> rand(0.75, +1.25);

  load = rand(mt);
}

DEFINE_MEMBER(float64, get_load)()
{
  static std::random_device rd;
  static std::mt19937 mt(rd());
  static std::uniform_real_distribution<float64> rand(0.75, +1.25);

  load = rand(mt);
  return load;
}

DEFINE_MEMBER(int, pack)(const int flag, char *buffer)
{
  return pack_base(flag, buffer);
}

DEFINE_MEMBER(int, unpack)(const int flag, char *buffer)
{
  return unpack_base(flag, buffer);
}

template class BaseChunk<1>;
template class BaseChunk<2>;
template class BaseChunk<3>;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
