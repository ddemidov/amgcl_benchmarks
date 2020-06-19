#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/scope_exit.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#if defined(SOLVER_BACKEND_VEXCL)
#  include <amgcl/backend/vexcl.hpp>
   typedef amgcl::backend::vexcl<double> Backend;
#elif defined(SOLVER_BACKEND_CUDA)
#  include <amgcl/backend/cuda.hpp>
#  include <amgcl/relaxation/cusparse_ilu0.hpp>
   typedef amgcl::backend::cuda<double> Backend;
#else
#  ifndef SOLVER_BACKEND_BUILTIN
#    define SOLVER_BACKEND_BUILTIN
#  endif
#  include <amgcl/backend/builtin.hpp>
   typedef amgcl::backend::builtin<double> Backend;
#endif

#include <amgcl/mpi/direct_solver/runtime.hpp>
#include <amgcl/mpi/coarsening/runtime.hpp>
#include <amgcl/mpi/relaxation/runtime.hpp>
#include <amgcl/mpi/partition/runtime.hpp>
#include <amgcl/mpi/amg.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/solver/runtime.hpp>
#include <amgcl/profiler.hpp>

#include "argh.h"
#include "domain_partition.hpp"

namespace amgcl {
    profiler<> prof;
}

struct renumbering {
    const domain_partition<3> &part;
    const std::vector<ptrdiff_t> &dom;

    renumbering(
            const domain_partition<3> &p,
            const std::vector<ptrdiff_t> &d
            ) : part(p), dom(d)
    {}

    ptrdiff_t operator()(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) const {
        boost::array<ptrdiff_t, 3> p = {{i, j, k}};
        std::pair<int,ptrdiff_t> v = part.index(p);
        return dom[v.first] + v.second;
    }
};

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    if (world.rank == 0)
        std::cout << "World size: " << world.size << std::endl;

    // Read configuration from command line
    ptrdiff_t n = 128;

    bool symm_dirichlet = true;
    std::string parameter_file;

    argh::parser cmdl(argc, argv);

    cmdl({"n","size"}, 128) >> n;
    cmdl("symbc") >> symm_dirichlet;
    cmdl({"P", "params"}, "") >> parameter_file;

    boost::property_tree::ptree prm;
    if (!parameter_file.empty()) read_json(parameter_file, prm);

    for(size_t i = 1; i < cmdl.size(); ++i) {
        amgcl::put(prm, cmdl(i).str());
    }

    const ptrdiff_t n3 = n * n * n;

    boost::array<ptrdiff_t, 3> lo = { {0,   0,   0  } };
    boost::array<ptrdiff_t, 3> hi = { {n-1, n-1, n-1} };

    using amgcl::prof;

    prof.tic("partition");
    domain_partition<3> part(lo, hi, world.size);
    ptrdiff_t chunk = part.size( world.rank );

    std::vector<ptrdiff_t> domain(world.size + 1);
    MPI_Allgather(
            &chunk, 1, amgcl::mpi::datatype<ptrdiff_t>(),
            &domain[1], 1, amgcl::mpi::datatype<ptrdiff_t>(), world);
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    lo = part.domain(world.rank).min_corner();
    hi = part.domain(world.rank).max_corner();

    renumbering renum(part, domain);
    prof.toc("partition");

    prof.tic("assemble");
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 7);
    val.reserve(chunk * 7);
    rhs.reserve(chunk);

    ptr.push_back(0);

    const double h2i  = (n - 1) * (n - 1);

    for(ptrdiff_t k = lo[2]; k <= hi[2]; ++k) {
        for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
            for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {

                if (!symm_dirichlet && (i == 0 || j == 0 || k == 0 || i + 1 == n || j + 1 == n || k + 1 == n)) {
                    col.push_back(renum(i,j,k));
                    val.push_back(1);
                    rhs.push_back(0);
                } else {
                    if (k > 0)  {
                        col.push_back(renum(i,j,k-1));
                        val.push_back(-h2i);
                    }

                    if (j > 0)  {
                        col.push_back(renum(i,j-1,k));
                        val.push_back(-h2i);
                    }

                    if (i > 0) {
                        col.push_back(renum(i-1,j,k));
                        val.push_back(-h2i);
                    }

                    col.push_back(renum(i,j,k));
                    val.push_back(6 * h2i);

                    if (i + 1 < n) {
                        col.push_back(renum(i+1,j,k));
                        val.push_back(-h2i);
                    }

                    if (j + 1 < n) {
                        col.push_back(renum(i,j+1,k));
                        val.push_back(-h2i);
                    }

                    if (k + 1 < n) {
                        col.push_back(renum(i,j,k+1));
                        val.push_back(-h2i);
                    }

                    rhs.push_back(1);
                }
                ptr.push_back( col.size() );
            }
        }
    }
    prof.toc("assemble");

    Backend::params bprm;

#if defined(SOLVER_BACKEND_VEXCL)
    vex::Context ctx(vex::Filter::Env);
    if (world.rank == 0)
        std::cout << ctx << std::endl;
    bprm.q = ctx;
#elif defined(SOLVER_BACKEND_CUDA)
    cusparseCreate(&bprm.cusparse_handle);
#endif

    auto f = Backend::copy_vector(rhs, bprm);
    auto x = Backend::create_vector(chunk, bprm);

    amgcl::backend::clear(*x);

    size_t iters;
    double resid, tm_setup, tm_solve;

    try {
        prof.tic("setup");
        typedef
            amgcl::mpi::make_solver<
                amgcl::mpi::amg<
                    Backend,
                    amgcl::runtime::mpi::coarsening::wrapper<Backend>,
                    amgcl::runtime::mpi::relaxation::wrapper<Backend>,
                    amgcl::runtime::mpi::direct::solver<double>,
                    amgcl::runtime::mpi::partition::wrapper<Backend>
                    >,
                amgcl::runtime::mpi::solver::wrapper<Backend>
                >
            Solver;

        Solver solve(world, std::tie(chunk, ptr, col, val), prm, bprm);
        tm_setup = prof.toc("setup");

        if (world.rank == 0)
            std::cout << solve << std::endl;

        prof.tic("solve");
        std::tie(iters, resid) = solve(*f, *x);
        tm_solve = prof.toc("solve");
    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        throw e;
    }

    if (world.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << resid << std::endl
            << std::endl
            << prof << std::endl;

#ifdef _OPENMP
        int nt = omp_get_max_threads();
#else
        int nt = 1;
#endif
        std::ostringstream log_name;
        log_name << "amgcl_amg";
        log_name << ".txt";

        std::ofstream log(log_name.str().c_str(), std::ios::app);
        log << n3 << "\t" << nt << "\t" << world.size
            << "\t" << tm_setup << "\t" << tm_solve
            << "\t" << iters << "\t" << std::endl;
    }
}
