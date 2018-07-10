#include <iostream>
#include <string>

#include <boost/property_tree/ptree.hpp>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

#include "log_times.hpp"
#include "argh.h"

namespace amgcl { profiler<> prof; }

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using namespace amgcl;

    argh::parser cmdl(argc, argv);
    double tol;
    cmdl({"e", "tol"}, "1e-4") >> tol;

    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    vex::profiler<> prof(ctx);

    prof.tic_cpu("reading");
    size_t rows, n, m;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs;

    io::read_crs(cmdl({"A", "matrix"}, "A.bin").str(), rows, ptr, col, val);
    io::read_dense(cmdl({"f", "rhs"}, "b.bin").str(), n, m, rhs);
    prof.toc("reading");

    typedef backend::vexcl<double> Backend;
    typedef make_solver<
        preconditioner::schur_pressure_correction<
            make_solver<
                relaxation::as_preconditioner<Backend, relaxation::damped_jacobi>,
                solver::bicgstab<Backend>
            >,
            make_solver<
                amg<
                    Backend,
                    coarsening::smoothed_aggregation,
                    relaxation::ilu0
                    >,
                solver::fgmres<Backend>
                >
            >,
        solver::fgmres<Backend>
        > Solver;

    Backend::params bprm;
    bprm.q = ctx;

    Solver::params prm;
    prm.solver.tol = tol;
    prm.precond.usolver.solver.tol = tol * 10;
    prm.precond.psolver.solver.tol = tol * 10;

    cmdl("pressure-iters", "16") >> prm.precond.psolver.solver.maxiter;
    prm.precond.psolver.precond.relax.solve.iters = 3;

    prm.precond.pmask.resize(n, 0);
    for(size_t i = 0; i < rows; i += 4)
        prm.precond.pmask[i] = 1;


    prof.tic_cl("setup");
    Solver solve(std::tie(rows, ptr, col, val), prm, bprm);
    double tm_setup = prof.toc("setup");

    std::cout << solve << std::endl;

    vex::vector<double> f(ctx, rhs);
    vex::vector<double> x(n); x = 0.0;

    int iters;
    double error;

    prof.tic_cl("solve");
    std::tie(iters, error) = solve(f, x);
    double tm_solve = prof.toc("solve");

    std::cout
        << "iters: " << iters << std::endl
        << "error: " << error << std::endl
        << prof << std::endl;

#ifdef VEXCL_BACKEND_OPENCL
    log_times("amgcl-schur-vexcl-opencl.txt", 1, rows, iters, tm_setup, tm_solve);
#else
    log_times("amgcl-schur-vexcl-cuda.txt", 1, rows, iters, tm_setup, tm_solve);
#endif
}
