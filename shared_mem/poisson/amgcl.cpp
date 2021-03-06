#include <iostream>
#include <vector>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/profiler.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif
#include "log_times.hpp"
#include "argh.h"

namespace amgcl { profiler<> prof; }
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
        make_solver<
            amg<
                backend::builtin<double>,
                coarsening::smoothed_aggregation,
                relaxation::spai0
                >,
            solver::cg<backend::builtin<double>>
        > Solver;

    Solver::params prm;

    argh::parser cmdl(argc, argv);
    int n;
    cmdl({"n", "size"}, "150") >> n;

    int n3 = n * n * n;

    std::vector<int> ptr, col;
    std::vector<double> val;

    backend::numa_vector<double> f(n3, false), x(n3, true);

#pragma omp parallel for
    for(int i = 0; i < n3; ++i) f[i] = 1.0;

    prof.tic("assemble");
    assemble(n, ptr, col, val);
    prof.toc("assemble");

    prof.tic("setup");
    Solver solve(std::tie(n3, ptr, col, val), prm);
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

#ifdef _OPENMP
        int nt = omp_get_max_threads();
#else
        int nt = 1;
#endif
    log_times("amgcl.txt", nt, n, iters, tm_setup, tm_solve);
}
