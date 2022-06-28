// -*- C++ -*-
#include "chunk.hpp"

#define DEFINE_MEMBER(type, name) template <int N> type BaseChunk<N>::name

DEFINE_MEMBER(int, pack_base)(const int flag, char *buffer) {
  char *buffer0 = buffer;

  std::memcpy(buffer, &myid, sizeof(int));
  buffer += sizeof(int);

  std::memcpy(buffer, &nbid[0][0][0], sizeof(int) * 27);
  buffer += sizeof(int) * 27;

  std::memcpy(buffer, &nbrank[0][0][0], sizeof(int) * 27);
  buffer += sizeof(int) * 27;

  std::cout << tfm::format("pack_base  : %3d (%4d byte)\n", myid,
                           buffer - buffer0);
  return buffer - buffer0;
}

DEFINE_MEMBER(int, unpack_base)(const int flag, char *buffer) {
  char *buffer0 = buffer;

  std::memcpy(&myid, buffer, sizeof(int));
  buffer += sizeof(int);

  std::memcpy(&nbid[0][0][0], buffer, sizeof(int) * 27);
  buffer += sizeof(int) * 27;

  std::memcpy(&nbrank[0][0][0], buffer, sizeof(int) * 27);
  buffer += sizeof(int) * 27;

  std::cout << tfm::format("unpack_base: %3d (%4d byte)\n", myid,
                           buffer - buffer0);
  return buffer - buffer0;
}

DEFINE_MEMBER(, BaseChunk)(const int id, const int dim[N]) : myid(id) {
  initialize_load();
}

DEFINE_MEMBER(void, initialize_load)() {
  static std::random_device                      rd;
  static std::mt19937                            mt(rd());
  static std::uniform_real_distribution<float64> rand(0.75, +1.25);

  load = rand(mt);
}

DEFINE_MEMBER(float64, get_load)() {
  static std::random_device                      rd;
  static std::mt19937                            mt(rd());
  static std::uniform_real_distribution<float64> rand(0.75, +1.25);

  load = rand(mt);
  return load;
}

DEFINE_MEMBER(int, pack)(const int flag, char *buffer) {
  return pack_base(flag, buffer);
}

DEFINE_MEMBER(int, unpack)(const int flag, char *buffer) {
  return unpack_base(flag, buffer);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
