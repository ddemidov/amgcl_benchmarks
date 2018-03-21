#include <iostream>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>

#include <amgcl/value_type/static_matrix.hpp>
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

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using namespace amgcl;
    namespace po = boost::program_options;

    typedef static_matrix<double, 4, 4> value_type;
    typedef static_matrix<double, 4, 1> rhs_type;
    typedef backend::builtin<value_type> Backend;
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

    Solver::params prm;
    prm.solver.maxiter = 500;
    prm.solver.tol = vm["tol"].as<double>();

    size_t rows, n, m;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, f;

    amgcl::profiler<> prof;

    prof.tic("read");
    io::read_crs(vm["matrix"].as<std::string>(), rows, ptr, col, val);
    io::read_dense(vm["rhs"].as<std::string>(), n, m, f);
    prof.toc("read");

    assert(n == rows && m == 1);

    int nb = rows / 4;

    prof.tic("setup");
    Solver solve(adapter::block_matrix<value_type>(boost::tie(rows, ptr, col, val)), prm);
    double tm_setup = prof.toc("setup");

    std::cout << solve << std::endl;

    rhs_type const * fptr = reinterpret_cast<rhs_type const *>(&f[0]);
    backend::numa_vector<rhs_type> F(fptr, fptr + nb);
    backend::numa_vector<rhs_type> X(nb, true);

    int iters;
    double error;

    prof.tic("solve");
    boost::tie(iters, error) = solve(F, X);
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
    log_times("amgcl.txt", nt, rows, iters, tm_setup, tm_solve);
}
