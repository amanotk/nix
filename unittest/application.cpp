// -*- C++ -*-

///
/// Pseudo Code
///
/// $Id$
///
#include "application.hpp"
#include "patch.hpp"
#include "patchmap.hpp"

typedef BaseApplication<BasePatch<3>,BasePatchMap> T_app;


int main(int argc, char **argv)
{
  //
  // + parse command line
  // + read configuration fie
  // + initialize MPI
  //
  T_app *app = new T_app(argc, argv);

  // info
  app->print_info(std::cout, 1);

  //
  // + setup solvers
  //
  app->setup();

  //
  // + setup initial condition or load a previous snapshot
  //
  app->initialize();

  //
  // + output diagnostic if needed
  //
  app->diagnostic();

  //
  // time integration loop
  //
  while( app->need_push() ) {
    // advance everything by one step
    app->push();

    // output diagnostic if needed
    app->diagnostic();

    // exit if elapsed time exceed a limit
    if( app->available_etime() < 0 ) {
      break;
    }

    app->rebuild_patchmap();
  }

  // info again
  app->print_info(std::cout, 1);

  //
  // + save current snapshot if needed
  // + finalize MPI
  //
  app->finalize();

  delete app;

  return 0;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
