#include <iostream>
#include <vector>

#include <amgcl/backend/cuda.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/profiler.hpp>

#include "log_times.hpp"
#include "argh.h"

namespace amgcl { profiler<backend::cuda_clock> prof; }
using amgcl::prof;

//---------------------------------------------------------------------------
void assemble(
        int n,
        std::vector<int>    &ptr,
        std::vector<int>    &col,
        std::vector<double> &val
        )
{
    int n3 = n * n * n;

    ptr.clear(); ptr.reserve(n3 + 1);
    col.clear(); col.reserve(n3 * 7);
    val.clear(); val.reserve(n3 * 7);

    ptr.push_back(0);

    for(int k = 0, idx = 0; k < n; ++k) {
        for(int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i, ++idx) {
                if (k > 0) {
                    col.push_back(idx - n * n);
                    val.push_back(-1.0/6.0);
                }

                if (j > 0) {
                    col.push_back(idx - n);
                    val.push_back(-1.0/6.0);
                }

                if (i > 0) {
                    col.push_back(idx - 1);
                    val.push_back(-1.0/6.0);
                }

                col.push_back(idx);
                val.push_back(1.0);

                if (i + 1 < n) {
                    col.push_back(idx + 1);
                    val.push_back(-1.0/6.0);
                }

                if (j + 1 < n) {
                    col.push_back(idx + n);
                    val.push_back(-1.0/6.0);
                }

                if (k + 1 < n) {
                    col.push_back(idx + n * n);
                    val.push_back(-1.0/6.0);
                }

                ptr.push_back(col.size());
            }
        }
    }
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using namespace amgcl;

    typedef
        backend::cuda<double> Backend;

    typedef
        make_solver<
            amg<
                Backend,
                coarsening::smoothed_aggregation,
                relaxation::spai0
                >,
            solver::cg<Backend>
        > Solver;

    Backend::params bprm;
    cusparseCreate(&bprm.cusparse_handle);

    Solver::params prm;
    prm.precond.coarsening.relax = 0.75;

    int n;

    argh::parser cmdl(argc, argv);
    cmdl({"n", "size"}, "150") >> n;
    int n3 = n * n * n;

    std::vector<int> ptr, col;
    std::vector<double> val;

    prof.tic("assemble");
    assemble(n, ptr, col, val);
    prof.toc("assemble");

    thrust::device_vector<double> f(n3, 1.0);
    thrust::device_vector<double> x(n3, 0.0);

    prof.tic("setup");
    Solver solve(std::tie(n3, ptr, col, val), prm, bprm);
    double tm_setup = prof.toc("setup");

    std::cout << solve << std::endl;

    int iters;
    double error;

    prof.tic("solve");
    std::tie(iters, error) = solve(f, x);
    double tm_solve = prof.toc("solve");

    std::cout
        << "iters: " << iters << std::endl
        << "error: " << error << std::endl
        << prof << std::endl;

    log_times("amgcl-cuda.txt", 1, n, iters, tm_setup, tm_solve);
}
