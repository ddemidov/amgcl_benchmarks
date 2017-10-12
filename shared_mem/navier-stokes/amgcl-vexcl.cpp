#include <iostream>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>
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

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using namespace amgcl;
    namespace po = boost::program_options;

    typedef static_matrix<double, 4, 4> value_type;
    typedef static_matrix<double, 4, 1> rhs_type;
    typedef backend::vexcl<value_type> Backend;
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

    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    // Enable static matrix value types for the vexcl backend:
    vex::scoped_program_header header(ctx,
            amgcl::backend::vexcl_static_matrix_declaration<double,4>());

    Backend::params bprm;
    bprm.q = ctx;

    Solver::params prm;
    prm.solver.maxiter = 500;
    prm.solver.tol = vm["tol"].as<double>();

    size_t rows, n, m;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, f;

    vex::profiler<> prof(ctx);

    prof.tic_cpu("read");
    io::read_crs(vm["matrix"].as<std::string>(), rows, ptr, col, val);
    io::read_dense(vm["rhs"].as<std::string>(), n, m, f);
    prof.toc("read");

    assert(n == rows && m == 1);

    int nb = rows / 4;

    prof.tic_cl("setup");
    Solver solve(adapter::block_matrix<4, value_type>(boost::tie(rows, ptr, col, val)), prm, bprm);
    double tm_setup = prof.toc("setup");

    std::cout << solve << std::endl;

    vex::vector<rhs_type> F(ctx, nb, reinterpret_cast<rhs_type const *>(&f[0]));
    vex::vector<rhs_type> X(ctx, nb);

    X = amgcl::math::zero<rhs_type>();

    int iters;
    double error;

    prof.tic_cl("solve");
    boost::tie(iters, error) = solve(F, X);
    double tm_solve = prof.toc("solve");

    std::cout
        << "iters: " << iters << std::endl
        << "error: " << error << std::endl
        << prof << std::endl;

#ifdef VEXCL_BACKEND_OPENCL
    log_times("amgcl-vexcl-opencl.txt", 1, rows, iters, tm_setup, tm_solve);
#else
    log_times("amgcl-vexcl-cuda.txt", 1, rows, iters, tm_setup, tm_solve);
#endif
}
