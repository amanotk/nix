// -*- C++ -*-
#ifndef _MPISTREAM_HPP_
#define _MPISTREAM_HPP_

///
/// MPI stream
///
#define MPICH_IGNORE_CXX_SEEK
#include "common.hpp"
#include "thirdparty/cmdline.hpp"
#include <mpi.h>

using namespace common;

/// template for Singleton class
template <class T>
class Singleton
{
private:
  Singleton(const Singleton &);
  Singleton &operator=(const Singleton &);

protected:
  Singleton(){};
  virtual ~Singleton(){};

public:
  static T *getInstance()
  {
    static T instance;
    return &instance;
  }
};

/// stream buffer mimicking "tee" command
class teebuf : public std::streambuf
{
private:
  std::streambuf *m_sb1;
  std::streambuf *m_sb2;

  virtual int overflow(int c)
  {
    if (c == EOF) {
      return !EOF;
    } else {
      int const r1 = m_sb1->sputc(c);
      int const r2 = m_sb2->sputc(c);
      return r1 == EOF || r2 == EOF ? EOF : c;
    }
  }

  virtual int sync()
  {
    int const r1 = m_sb1->pubsync();
    int const r2 = m_sb2->pubsync();
    return r1 == 0 && r2 == 0 ? 0 : -1;
  }

public:
  teebuf(std::streambuf *sb1, std::streambuf *sb2) : m_sb1(sb1), m_sb2(sb2)
  {
  }
};

///
/// MPI stream class
///
class mpistream : public Singleton<mpistream>
{
  friend class Singleton<mpistream>;

private:
  // for stdout/stderr
  std::string     m_outf;   ///< dummy standard output file
  std::string     m_errf;   ///< dummy standard error file
  std::ofstream  *m_out;    ///< dummy standard output
  std::ofstream  *m_err;    ///< dummy standard error
  std::streambuf *m_errbuf; ///< buffer of original cerr
  std::streambuf *m_outbuf; ///< buffer of original cout
  teebuf         *m_outtee; ///< buffer for replicating cout and file
  teebuf         *m_errtee; ///< buffer for replicating cerr and file

  mpistream(){};
  ~mpistream(){};

  // remain undefined
  mpistream(const mpistream &);
  mpistream &operator=(const mpistream &);

public:
  /// initialize MPI call
  static void initialize(const char *header)
  {
    mpistream *instance = getInstance();
    int        thisrank;

    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);

    // open dummy standard output stream
    instance->m_outf   = tfm::format("%s_PE%04d.stdout", header, thisrank);
    instance->m_out    = new std::ofstream(instance->m_outf.c_str());
    instance->m_outtee = new teebuf(std::cout.rdbuf(), instance->m_out->rdbuf());

    // open dummy standard error stream
    instance->m_errf   = tfm::format("%s_PE%04d.stderr", header, thisrank);
    instance->m_err    = new std::ofstream(instance->m_errf.c_str());
    instance->m_errtee = new teebuf(std::cerr.rdbuf(), instance->m_err->rdbuf());

    if (thisrank == 0) {
      // stdout/stderr are replicated for rank==0
      instance->m_outbuf = std::cout.rdbuf(instance->m_outtee);
      instance->m_errbuf = std::cerr.rdbuf(instance->m_errtee);
    } else {
      instance->m_outbuf = std::cout.rdbuf(instance->m_out->rdbuf());
      instance->m_errbuf = std::cerr.rdbuf(instance->m_err->rdbuf());
    }
  }

  /// finalize MPI call
  static void finalize(int cleanup = 0)
  {
    mpistream *instance = getInstance();

    // close dummy standard output
    instance->m_out->flush();
    instance->m_out->close();
    std::cout.rdbuf(instance->m_outbuf);
    delete instance->m_outtee;
    delete instance->m_out;

    // close dummy standard error
    instance->m_err->flush();
    instance->m_err->close();
    std::cerr.rdbuf(instance->m_errbuf);
    delete instance->m_errtee;
    delete instance->m_err;

    switch (cleanup) {
    case 0: // remove file
      std::remove(instance->m_outf.c_str());
      std::remove(instance->m_errf.c_str());
      break;
    default:
      break;
    }
  }

  /// flush
  static void flush()
  {
    mpistream *instance = getInstance();

    instance->m_out->flush();
    instance->m_err->flush();
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
