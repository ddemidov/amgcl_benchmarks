#include <iostream>

#include <cusp/csr_matrix.h>
#include <cusp/gallery/diffusion.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>
#include <cusp/precond/smoother/gauss_seidel_smoother.h>
#include <cusp/precond/smoother/polynomial_smoother.h>
#include <performance/timer.h>

#include "log_times.hpp"
#include "argh.h"

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

using namespace cusp::precond::aggregation;

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    argh::parser cmdl(argc, argv);
    typedef cusp::device_memory mem_space;

    int n;
    cmdl({"n", "size"}, "150") >> n;
    int n3 = n * n * n;

    std::vector<int> ptr, col;
    std::vector<double> val;
    assemble(n, ptr, col, val);

    timer t0;
    cusp::hyb_matrix<int, double, mem_space> A(
            cusp::make_csr_matrix_view(n3, n3, ptr.back(),
                cusp::array1d<int,    cusp::host_memory>(ptr),
                cusp::array1d<int,    cusp::host_memory>(col),
                cusp::array1d<double, cusp::host_memory>(val)));

    smoothed_aggregation<int, double, cusp::device_memory> M(A);
    double tm_setup = t0.seconds_elapsed();

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<double, mem_space> x(A.num_rows, 0);
    cusp::array1d<double, mem_space> b(A.num_rows, 1);

    cusp::monitor<double> monitor(b, 1000, 1e-8);

    // solve
    timer t1;
    cusp::krylov::cg(A, x, b, monitor, M);
    double tm_solve = t1.seconds_elapsed();

    std::cout
        << "iters: " << monitor.iteration_count()    << std::endl
        << "error: " << monitor.relative_tolerance() << std::endl
        << "setup: " << tm_setup << std::endl
        << "solve: " << tm_solve << std::endl
        ;

    log_times("cusp.txt", 1, n, monitor.iteration_count(), tm_setup, tm_solve);
}

