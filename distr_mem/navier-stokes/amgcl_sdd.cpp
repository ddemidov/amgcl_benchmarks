#include <iostream>
#include <iterator>
#include <iomanip>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/scope_exit.hpp>

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

#include <amgcl/io/binary.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/schur_pressure_correction.hpp>
#include <amgcl/mpi/block_preconditioner.hpp>
#include <amgcl/mpi/subdomain_deflation.hpp>
#include <amgcl/mpi/direct_solver/runtime.hpp>
#include <amgcl/mpi/solver/runtime.hpp>
#include <amgcl/profiler.hpp>

#include "argh.h"

namespace amgcl {
    profiler<> prof;
}

using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
std::vector<ptrdiff_t> read_problem(
        const amgcl::mpi::communicator &world,
        const std::string &A_file,
        const std::string &rhs_file,
        const std::string &part_file,
        std::vector<ptrdiff_t> &ptr,
        std::vector<ptrdiff_t> &col,
        std::vector<double>    &val,
        std::vector<double>    &rhs
        )
{
    // Read partition
    ptrdiff_t n, m;
    std::vector<ptrdiff_t> domain(world.size + 1, 0);
    std::vector<int> part;

    std::tie(n, m) = amgcl::io::mm_reader(part_file)(part);
    for(int p : part) {
        ++domain[p+1];
        precondition(p < world.size, "MPI world does not correspond to partition");
    }
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    ptrdiff_t chunk_beg = domain[world.rank];
    ptrdiff_t chunk_end = domain[world.rank + 1];
    ptrdiff_t chunk     = chunk_end - chunk_beg;

    // Reorder unknowns
    std::vector<ptrdiff_t> order(n);
    for(ptrdiff_t i = 0; i < n; ++i)
        order[i] = domain[part[i]]++;

    std::rotate(domain.begin(), domain.end()-1, domain.end());
    domain[0] = 0;

    // Read matrix chunk
    {
        using namespace amgcl::io;

        std::ifstream A(A_file.c_str(), std::ios::binary);
        precondition(A, "Failed to open matrix file (" + A_file + ")");

        std::ifstream b(rhs_file.c_str(), std::ios::binary);
        precondition(b, "Failed to open rhs file (" + rhs_file + ")");

        ptrdiff_t rows;
        precondition(read(A, rows), "File I/O error");
        precondition(rows == n, "Matrix and partition have incompatible sizes");

        ptr.clear(); ptr.reserve(chunk + 1); ptr.push_back(0);

        std::vector<ptrdiff_t> gptr(n + 1);
        precondition(read(A, gptr), "File I/O error");

        size_t col_beg = sizeof(rows) + sizeof(gptr[0]) * (n + 1);
        size_t val_beg = col_beg + sizeof(col[0]) * gptr.back();
        size_t rhs_beg = 2 * sizeof(ptrdiff_t);

        // Count local nonzeros
        for(ptrdiff_t i = 0; i < n; ++i)
            if (part[i] == world.rank)
                ptr.push_back(gptr[i+1] - gptr[i]);

        std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

        col.clear(); col.reserve(ptr.back());
        val.clear(); val.reserve(ptr.back());
        rhs.clear(); rhs.reserve(chunk);

        // Read local matrix and rhs stripes
        for(ptrdiff_t i = 0; i < n; ++i) {
            if (part[i] != world.rank) continue;

            ptrdiff_t c;
            A.seekg(col_beg + gptr[i] * sizeof(c));
            for(ptrdiff_t j = gptr[i], e = gptr[i+1]; j < e; ++j) {
                precondition(read(A, c), "File I/O error (1)");
                col.push_back(order[c]);
            }
        }

        for(ptrdiff_t i = 0; i < n; ++i) {
            if (part[i] != world.rank) continue;

            double v;
            A.seekg(val_beg + gptr[i] * sizeof(v));
            for(ptrdiff_t j = gptr[i], e = gptr[i+1]; j < e; ++j) {
                precondition(read(A, v), "File I/O error (2)");
                val.push_back(v);
            }
        }

        for(ptrdiff_t i = 0; i < n; ++i) {
            if (part[i] != world.rank) continue;

            double f;
            b.seekg(rhs_beg + i * sizeof(f));
            precondition(read(b, f), "File I/O error (3)");
            rhs.push_back(f);
        }
    }

    return domain;
}

//---------------------------------------------------------------------------
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
    argh::parser cmdl(argc, argv);

    boost::property_tree::ptree prm;

    std::string prm_file;
    if (cmdl({"P", "params"}) >> prm_file)
        read_json(prm_file, prm);

    for(size_t i = 1; i < cmdl.size(); ++i)
        amgcl::put(prm, cmdl(i).str());

    prof.tic("read problem");
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    std::vector<ptrdiff_t> domain = read_problem(
            world,
            cmdl({"A", "matrix"}).str(),
            cmdl({"f", "rhs"}).str(),
            cmdl({"s", "part"}).str(),
            ptr, col, val, rhs
            );

    ptrdiff_t chunk = domain[world.rank + 1] - domain[world.rank];
    prof.toc("read problem");

    std::vector<char> pm(chunk, 0);
    for(ptrdiff_t i = 0; i < chunk; i += 4) pm[i] = 1;
    prm.put("precond.pmask", static_cast<void*>(&pm[0]));
    prm.put("precond.pmask_size", chunk);

    std::function<double(ptrdiff_t,unsigned)> dv = amgcl::mpi::constant_deflation(1);
    prm.put("precond.psolver.num_def_vec", 1);
    prm.put("precond.psolver.def_vec", &dv);

    Backend::params bprm;

#if defined(SOLVER_BACKEND_VEXCL)
    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;
    bprm.q = ctx;
#elif defined(SOLVER_BACKEND_CUDA)
    cusparseCreate(&bprm.cusparse_handle);
#endif

    auto f = Backend::copy_vector(rhs, bprm);
    auto x = Backend::create_vector(chunk, bprm);

    amgcl::backend::clear(*x);

    MPI_Barrier(world);

    prof.tic("setup");
    typedef
        amgcl::mpi::make_solver<
            amgcl::mpi::schur_pressure_correction<
                amgcl::mpi::make_solver<
                    amgcl::mpi::block_preconditioner<
                        amgcl::relaxation::as_preconditioner<Backend, amgcl::runtime::relaxation::wrapper>
                        >,
                    amgcl::runtime::mpi::solver::wrapper<Backend>
                    >,
                amgcl::mpi::subdomain_deflation<
                    amgcl::amg<Backend, amgcl::runtime::coarsening::wrapper, amgcl::runtime::relaxation::wrapper>,
                    amgcl::runtime::mpi::solver::wrapper<Backend>,
                    amgcl::runtime::mpi::direct::solver<double>
                    >
                >,
            amgcl::runtime::mpi::solver::wrapper<Backend>
            > Solver;

    Solver solve(world, std::tie(chunk, ptr, col, val), prm, bprm);
    double tm_setup = prof.toc("setup");

    prof.tic("solve");
    size_t iters;
    double resid;
    std::tie(iters, resid) = solve(*f, *x);
    double tm_solve = prof.toc("solve");

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
        std::ofstream log("amgcl_spc_sdd.txt", std::ios::app);
        log << domain.back() << "\t" << nt << "\t" << world.size
            << "\t" << tm_setup << "\t" << tm_solve
            << "\t" << iters << "\t" << std::endl;
    }
}
