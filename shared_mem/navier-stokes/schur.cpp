#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
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

#ifdef _OPENMP
#  include <omp.h>
#endif
#include "log_times.hpp"

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using namespace amgcl;
    namespace po = boost::program_options;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "matrix,A",
         po::value<std::string>()->default_value("A.bin"),
         "The system matrix in MatrixMarket format"
        )
        (
         "rhs,f",
         po::value<std::string>()->default_value("b.bin"),
         "The right-hand side in MatrixMarket format"
        )
        (
         "tol,e",
         po::value<double>()->default_value(1e-4),
         "Tolerance"
        )
        (
         "pressure-iters",
         po::value<int>()->default_value(16),
         "Number of iterations for the pressure subproblem"
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    po::notify(vm);

    double tol = vm["tol"].as<double>();

    size_t rows, n, m;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs;

    profiler<> prof;

    prof.tic("reading");
    io::read_crs(vm["matrix"].as<std::string>(), rows, ptr, col, val);
    io::read_dense(vm["rhs"].as<std::string>(), n, m, rhs);
    prof.toc("reading");

    typedef backend::builtin<double> Backend;
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

    Solver::params prm;

    prm.solver.tol = tol;
    prm.precond.usolver.solver.tol = tol * 10;
    prm.precond.psolver.solver.tol = tol * 10;
    prm.precond.psolver.solver.maxiter = vm["pressure-iters"].as<int>();

    prm.precond.pmask.resize(n, 0);
    for(size_t i = 0; i < rows; i += 4)
        prm.precond.pmask[i] = 1;


    prof.tic("setup");
    Solver solve(boost::tie(rows, ptr, col, val), prm);
    double tm_setup = prof.toc("setup");

    std::cout << solve << std::endl;

    std::vector<double> x(n, 0.0);

    int iters;
    double error;

    prof.tic("solve");
    boost::tie(iters, error) = solve(rhs, x);
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
    log_times("amgcl-schur.txt", nt, rows, iters, tm_setup, tm_solve);
}
