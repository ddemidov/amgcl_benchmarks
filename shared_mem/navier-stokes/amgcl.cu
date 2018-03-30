#include <iostream>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>

#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/backend/cuda.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/solver/lgmres.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

#include "log_times.hpp"

namespace amgcl { profiler<backend::cuda_clock> prof; }
using amgcl::prof;

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using namespace amgcl;
    namespace po = boost::program_options;

    typedef backend::cuda<double> Backend;
    typedef make_solver<
        amg<Backend, coarsening::aggregation, relaxation::damped_jacobi>,
        solver::lgmres<Backend>
        > Solver;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Show this help.")
        ("matrix,A",
         po::value<std::string>()->default_value("A.bin"),
         "System matrix in binary format."
        )
        (
         "rhs,f",
         po::value<std::string>()->default_value("b.bin"),
         "The RHS vector in binary format."
        )
        (
         "tol,e",
         po::value<double>()->default_value(1e-4),
         "Tolerance"
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    Backend::params bprm;
    cusparseCreate(&bprm.cusparse_handle);

    Solver::params prm;
    prm.solver.maxiter = 500;
    prm.solver.tol = vm["tol"].as<double>();
    prm.precond.coarsening.aggr.block_size = 4;

    size_t rows, n, m;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, f;

    prof.tic("read");
    io::read_crs(vm["matrix"].as<std::string>(), rows, ptr, col, val);
    io::read_dense(vm["rhs"].as<std::string>(), n, m, f);
    prof.toc("read");

    assert(n == rows && m == 1);

    prof.tic("setup");
    Solver solve(boost::tie(rows, ptr, col, val), prm, bprm);
    double tm_setup = prof.toc("setup");

    std::cout << solve << std::endl;

    thrust::device_vector<double> F = f;
    thrust::device_vector<double> X(n, 0.0);

    int iters;
    double error;

    prof.tic("solve");
    boost::tie(iters, error) = solve(F, X);
    double tm_solve = prof.toc("solve");

    std::cout
        << "iters: " << iters << std::endl
        << "error: " << error << std::endl
        << prof << std::endl;

    log_times("amgcl-cuda.txt", 1, rows, iters, tm_setup, tm_solve);
}
