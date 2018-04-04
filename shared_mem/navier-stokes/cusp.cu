#include <iostream>
#include <fstream>

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
int read_problem(std::string A_file, std::string f_file,
        std::vector<int> &ptr, std::vector<int> &col, std::vector<double> &val,
        std::vector<double> &rhs)
{
    std::ifstream A(A_file.c_str(), std::ios::binary);
    std::ifstream F(f_file.c_str(), std::ios::binary);

    ptrdiff_t n;

    A.read((char*)&n, sizeof(n));

    ptr.clear(); ptr.reserve(n+1);

    for(int i = 0; i <= n; ++i) {
        ptrdiff_t p;
        A.read((char*)&p, sizeof(p));
        ptr.push_back(p);
    }

    col.clear(); col.reserve(ptr.back());

    for(int i = 0; i < ptr.back(); ++i) {
        ptrdiff_t c;
        A.read((char*)&c, sizeof(c));
        col.push_back(c);
    }

    val.resize(ptr.back());
    A.read((char*)val.data(), ptr.back() * sizeof(double));

    rhs.resize(n);

    ptrdiff_t shape[2];
    F.read((char*)shape, sizeof(shape));
    F.read((char*)rhs.data(), n * sizeof(double));

    return n;
}

using namespace cusp::precond::aggregation;

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    argh::parser cmdl(argc, argv);

    typedef cusp::device_memory mem_space;

    std::vector<int> ptr, col;
    std::vector<double> val, rhs;
    int n = read_problem(
            cmdl({"A", "matrix"}, "A.bin").str(),
            cmdl({"f", "rhs"}, "b.bin").str(),
            ptr, col, val, rhs);

    timer t0;
    cusp::hyb_matrix<int, double, mem_space> A(
            cusp::make_csr_matrix_view(n, n, ptr.back(),
                cusp::array1d<int,    cusp::host_memory>(ptr),
                cusp::array1d<int,    cusp::host_memory>(col),
                cusp::array1d<double, cusp::host_memory>(val)));

    smoothed_aggregation<int, double, cusp::device_memory> M(A);
    double tm_setup = t0.seconds_elapsed();

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<double, mem_space> x(n, 0);
    cusp::array1d<double, mem_space> b(rhs);

    double tol;
    cmdl({"e", "tol"}, "1e-4") >> tol;
    cusp::monitor<double> monitor(b, 1000, tol);

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

