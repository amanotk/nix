// -*- C++ -*-
#ifndef _MPISTREAM_HPP_
#define _MPISTREAM_HPP_

#include "debug.hpp"
#include "tinyformat.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>

// format for temporary stdout and stderr files
static constexpr char filename_format[] = "%06d";
static constexpr char dev_null[]        = "/dev/null";

///
/// @brief singleton class
/// @tparam T typename
///
template <typename T>
class Singleton
{
private:
  Singleton(const Singleton&);
  Singleton& operator=(const Singleton&);

protected:
  Singleton(){};
  virtual ~Singleton(){};

public:
  static T* getInstance()
  {
    static T instance;
    return &instance;
  }
};

///
/// @brief stream buffer mimicking "tee" command
///
class teebuf : public std::streambuf
{
private:
  std::streambuf* m_sb1;
  std::streambuf* m_sb2;

  virtual int overflow(int c) override
  {
    if (c == EOF) {
      return !EOF;
    } else {
      int const r1 = m_sb1->sputc(c);
      int const r2 = m_sb2->sputc(c);
      return r1 == EOF || r2 == EOF ? EOF : c;
    }
  }

  virtual int sync() override
  {
    int const r1 = m_sb1->pubsync();
    int const r2 = m_sb2->pubsync();
    return r1 == 0 && r2 == 0 ? 0 : -1;
  }

public:
  teebuf(std::streambuf* sb1, std::streambuf* sb2) : m_sb1(sb1), m_sb2(sb2)
  {
  }
};

///
/// @brief MPI stream class
///
class MpiStream : public Singleton<MpiStream>
{
  friend class Singleton<MpiStream>;

protected:
  // for stdout/stderr
  std::string                    m_outf;   ///< dummy standard output file
  std::string                    m_errf;   ///< dummy standard error file
  std::unique_ptr<std::ofstream> m_out;    ///< dummy standard output
  std::unique_ptr<std::ofstream> m_err;    ///< dummy standard error
  std::unique_ptr<teebuf>        m_outtee; ///< buffer for replicating cout and file
  std::unique_ptr<teebuf>        m_errtee; ///< buffer for replicating cerr and file
  std::streambuf*                m_errbuf; ///< buffer of original cerr
  std::streambuf*                m_outbuf; ///< buffer of original cout

  MpiStream(){};
  ~MpiStream(){};

  // remain undefined
  MpiStream(const MpiStream&);
  MpiStream& operator=(const MpiStream&);

public:
  static void initialize(const char* dirname, int thisrank, int nprocess, int max_file_per_dir = -1)
  {
    MpiStream* instance = getInstance();

    // open dummy standard output stream
    instance->m_outf   = get_stdout_filename(dirname, thisrank, nprocess, max_file_per_dir);
    instance->m_out    = std::make_unique<std::ofstream>(instance->m_outf.c_str());
    instance->m_outtee = std::make_unique<teebuf>(std::cout.rdbuf(), instance->m_out->rdbuf());

    // open dummy standard error stream
    instance->m_errf   = get_stderr_filename(dirname, thisrank, nprocess, max_file_per_dir);
    instance->m_err    = std::make_unique<std::ofstream>(instance->m_errf.c_str());
    instance->m_errtee = std::make_unique<teebuf>(std::cerr.rdbuf(), instance->m_err->rdbuf());

    if (thisrank == 0) {
      // stdout/stderr are replicated for rank==0
      instance->m_outbuf = std::cout.rdbuf(instance->m_outtee.get());
      instance->m_errbuf = std::cerr.rdbuf(instance->m_errtee.get());
    } else {
      instance->m_outbuf = std::cout.rdbuf(instance->m_out->rdbuf());
      instance->m_errbuf = std::cerr.rdbuf(instance->m_err->rdbuf());
    }
  }

  static void finalize(int cleanup = 0)
  {
    MpiStream* instance = getInstance();

    // close dummy standard output
    instance->m_out->flush();
    instance->m_out->close();
    std::cout.rdbuf(instance->m_outbuf);

    // close dummy standard error
    instance->m_err->flush();
    instance->m_err->close();
    std::cerr.rdbuf(instance->m_errbuf);

    // remove files
    if (cleanup == 0 && instance->m_outf != dev_null) {
      std::remove(instance->m_outf.c_str());
    }
    if (cleanup == 0 && instance->m_errf != dev_null) {
      std::remove(instance->m_errf.c_str());
    }
  }

  static void flush()
  {
    MpiStream* instance = getInstance();

    instance->m_out->flush();
    instance->m_err->flush();
  }

  static std::string get_filename_pattern(int thisrank, int nprocess, int max_file_per_dir = -1)
  {
    if (max_file_per_dir == -1) {
      max_file_per_dir = nprocess;
    }

    // compute recursive directory level
    int directory_level = 0;

    while (nprocess > max_file_per_dir) {
      nprocess /= max_file_per_dir;
      directory_level++;
    }

    // make directory name
    std::filesystem::path path;

    for (int level = 0; level < directory_level; level++) {
      int max_file_level = std::pow(max_file_per_dir, directory_level - level);

      path = path / tfm::format(filename_format, (thisrank / max_file_level) * max_file_level);
    }
    path = path / tfm::format(filename_format, thisrank);

    return path.string();
  }

  static std::string get_filename(const char* dirname, const char* extension, int thisrank,
                                  int nprocess, int max_file_per_dir = -1)
  {
    if (std::filesystem::exists(dirname) == false) {
      if (thisrank == 0) {
        std::filesystem::create_directory(dirname);
      }
    } else if (std::filesystem::is_directory(dirname) == false) {
      ERROR << tfm::format("Error: %s exists but not a directory.", dirname) << std::endl;
    }

    std::filesystem::path path(dirname);
    path = path / get_filename_pattern(thisrank, nprocess, max_file_per_dir);

    return path.string() + extension;
  }

  static std::string get_stdout_filename(const char* dirname, int thisrank, int nprocess,
                                         int max_file_per_dir = -1)
  {
    bool use_null_stream = (dirname == nullptr) || std::string("") == dirname;

    if (use_null_stream) {
      return dev_null;
    } else {
      return get_filename(dirname, ".stdout", thisrank, nprocess, max_file_per_dir);
    }
  }

  static std::string get_stderr_filename(const char* dirname, int thisrank, int nprocess,
                                         int max_file_per_dir = -1)
  {
    bool use_null_stream = (dirname == nullptr) || std::string("") == dirname;

    if (use_null_stream) {
      return dev_null;
    } else {
      return get_filename(dirname, ".stderr", thisrank, nprocess, max_file_per_dir);
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
