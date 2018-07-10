#include <iostream>
#include <vector>

#include <boost/property_tree/ptree.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/solver/lgmres.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif
#include "log_times.hpp"
#include "argh.h"

namespace amgcl { amgcl::profiler<> prof; }
using amgcl::prof;

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using namespace amgcl;

    typedef backend::builtin<double> Backend;
    typedef make_solver<
        amg<Backend, coarsening::aggregation, relaxation::damped_jacobi>,
        solver::lgmres<Backend>
        > Solver;

    argh::parser cmdl(argc, argv);

    Solver::params prm;
    prm.solver.maxiter = 500;
    cmdl({"e", "tol"}, "1e-4") >> prm.solver.tol;
    prm.precond.coarsening.aggr.block_size = 4;

    size_t rows, n, m;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs;

    prof.tic("read");
    io::read_crs(cmdl({"A", "matrix"}, "A.bin").str(), rows, ptr, col, val);
    io::read_dense(cmdl({"f", "rhs"}, "b.bin").str(), n, m, rhs);
    prof.toc("read");

    assert(n == rows && m == 1);

    prof.tic("setup");
    Solver solve(std::tie(rows, ptr, col, val), prm);
    double tm_setup = prof.toc("setup");

    std::cout << solve << std::endl;

    backend::numa_vector<double> f(rhs);
    backend::numa_vector<double> x(n, true);

    int iters;
    double error;

    prof.tic("solve");
    std::tie(iters, error) = solve(f, x);
    double tm_solve = prof.toc("solve");

    std::cout
        << "iters: " << iters << std::endl
        << "error: " << error << std::endl
        << prof << std::endl;

#ifdef _OPENMP
    int nt = omp_get_max_threads();
#else
    int nt = 1;
#endif
    log_times("amgcl-scalar.txt", nt, rows, iters, tm_setup, tm_solve);
}
