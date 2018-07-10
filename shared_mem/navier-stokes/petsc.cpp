#include <iostream>
#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>
#include <cassert>

#include <petscksp.h>
#include <petsctime.h>

#include <amgcl/io/binary.hpp>
#include <amgcl/io/mm.hpp>

#include "log_times.hpp"

//---------------------------------------------------------------------------
void read_problem(std::string A_file, std::string f_file, std::string p_file,
                  Mat &A, Vec &f, Vec &x)
{
    namespace io = amgcl::io;

    int mpi_rank;
    int mpi_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Read partition
    int n,m;
    std::vector<int> part;
    std::tie(n, m) = io::mm_reader(p_file)(part);

    assert(m == 1);

    if (mpi_rank == 0) {
        std::cout << "global rows: " << n << std::endl;
    }

    // Compute domain sizes
    std::vector<int> domain(mpi_size + 1, 0);
    for(auto p : part) ++domain[p+1];
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    int chunk = domain[mpi_rank+1] - domain[mpi_rank];

    if (mpi_rank == 0) {
        std::cout << "local rows:";
        for(auto d : domain) std::cout << " " << d;
        std::cout << std::endl;
    }

    // Reorder unknowns
    std::vector<int> perm(n);
    for(int i = 0; i < n; ++i) perm[i] = domain[part[i]]++;
    std::rotate(domain.begin(), domain.end()-1, domain.end());
    domain[0] = 0;

    // Read our chunk of the matrix
    std::ifstream af(A_file, std::ios::binary);
    size_t rows;
    af.read((char*)&rows, sizeof(size_t));

    assert(rows == n);

    std::vector<int> ptr(chunk);
    std::vector<int> nnz(chunk);

    size_t ptr_pos = af.tellg();

    ptrdiff_t glob_nnz;
    {
        af.seekg(ptr_pos + n * sizeof(ptrdiff_t));
        af.read((char*)&glob_nnz, sizeof(ptrdiff_t));

        if (mpi_rank == 0) {
            std::cout << "global nnz: " << glob_nnz << std::endl;
        }
    }

    size_t col_pos = ptr_pos + (n + 1) * sizeof(ptrdiff_t);
    size_t val_pos = col_pos + glob_nnz * sizeof(ptrdiff_t);

    for(int i = 0, j = 0; i < n; ++i) {
        if (part[i] == mpi_rank) {
            assert(perm[i] - domain[mpi_rank] == j);

            af.seekg(ptr_pos + i * sizeof(ptrdiff_t));

            ptrdiff_t p[2];
            af.read((char*)p, sizeof(p));

            ptr[j] = p[0];
            nnz[j] = p[1] - p[0];

            ++j;
        }
    }

    MatCreate(MPI_COMM_WORLD, &A);
    MatSetSizes(A, chunk, chunk, n, n);
    MatSetFromOptions(A);
    MatMPIAIJSetPreallocation(A, 128, 0, 128, 0);
    MatSeqAIJSetPreallocation(A, 128, 0);

    std::vector<PetscInt>    col; col.reserve(128);
    std::vector<PetscScalar> val; val.reserve(128);

    for(int i = 0; i < chunk; ++i, col.clear(), val.clear()) {
        af.seekg(col_pos + ptr[i] * sizeof(ptrdiff_t));
        for(int j = 0; j < nnz[i]; ++j) {
            ptrdiff_t c;
            af.read((char*)&c, sizeof(ptrdiff_t));
            col.push_back(perm[c]);
        }

        af.seekg(val_pos + ptr[i] * sizeof(double));
        for(int j = 0; j < nnz[i]; ++j) {
            double v;
            af.read((char*)&v, sizeof(double));
            val.push_back(v);
        }

        PetscInt row = i + domain[mpi_rank];
        MatSetValues(A, 1, &row, col.size(), col.data(), val.data(), INSERT_VALUES);
    }

    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

    // Read our chunk of the RHS
    VecCreate(MPI_COMM_WORLD, &f);
    VecCreate(MPI_COMM_WORLD, &x);

    VecSetFromOptions(f);
    VecSetFromOptions(x);

    VecSetSizes(f, chunk, n);
    VecSetSizes(x, chunk, n);

    VecSet(x, 0.0);

    std::ifstream ff(f_file, std::ios::binary);
    {
        size_t shape[2];
        ff.read((char*)shape, sizeof(shape));

        assert(shape[0] == n);
        assert(shape[1] == 1);
    }
    size_t f_pos = ff.tellg();

    for(int i = 0, j = 0; i < n; ++i) {
        if (part[i] == mpi_rank) {
            ff.seekg(f_pos + i * sizeof(double));

            double v;
            ff.read((char*)&v, sizeof(double));

            VecSetValue(f, domain[mpi_rank] + j, v, INSERT_VALUES);
            ++j;
        }
    }

    VecAssemblyBegin(f);
    VecAssemblyEnd(f);
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    KSP                solver;
    PC                 prec;
    Mat                A;
    Vec                x,f;
    PetscScalar        v;
    KSPConvergedReason reason;
    PetscInt           n = 200, iters;
    PetscReal          error;
    PetscLogDouble     tic, toc;

    char A_file[256];
    char f_file[256];
    char p_file[256];

    PetscInitialize(&argc, &argv, 0, "poisson3d");
    PetscOptionsGetString(0, 0, "-A", A_file, 255, 0);
    PetscOptionsGetString(0, 0, "-f", f_file, 255, 0);
    PetscOptionsGetString(0, 0, "-p", p_file, 255, 0);

    read_problem(A_file, f_file, p_file, A, f, x);

    PetscTime(&tic);
    KSPCreate(MPI_COMM_WORLD, &solver);
    KSPSetOperators(solver,A,A);
    KSPSetTolerances(solver, 1e-4, PETSC_DEFAULT, PETSC_DEFAULT, 100);

    KSPGetPC(solver,&prec);
    PCSetType(prec, PCFIELDSPLIT);
    PCFieldSplitSetBlockSize(prec, 4);
    PetscInt pfields[] = {0};
    PetscInt ufields[] = {1,2,3};
    PCFieldSplitSetFields(prec, "p", 1, pfields, pfields);
    PCFieldSplitSetFields(prec, "u", 3, ufields, ufields);
    PCFieldSplitSetType(prec, PC_COMPOSITE_ADDITIVE);

    PetscInt nsplits;
    KSP *subksp;
    PCFieldSplitGetSubKSP(prec, &nsplits, &subksp);
    PC pc_p;
    KSPGetPC(subksp[0], &pc_p);
    PCSetType(pc_p, PCGAMG);
    PetscFree(subksp);

    KSPSetFromOptions(solver);
    PCSetFromOptions(prec);
    KSPSetUp(solver);
    PetscTime(&toc);
    double tm_setup = toc - tic;

    PetscTime(&tic);
    KSPSolve(solver,f,x);
    PetscTime(&toc);
    double tm_solve = toc - tic;

    KSPGetConvergedReason(solver,&reason);
    if (reason==KSP_DIVERGED_INDEFINITE_PC) {
        PetscPrintf(PETSC_COMM_WORLD,"\nDivergence because of indefinite preconditioner;\n");
        PetscPrintf(PETSC_COMM_WORLD,"Run the executable again but with '-pc_factor_shift_type POSITIVE_DEFINITE' option.\n");
    } else if (reason<0) {
        PetscPrintf(PETSC_COMM_WORLD,"\nOther kind of divergence: this should not happen.\n");
    } else {
        KSPGetIterationNumber(solver,&iters);
        KSPGetResidualNorm(solver,&error);
        PetscPrintf(PETSC_COMM_WORLD,"\niters: %d\nerror: %lf", iters, error);
        PetscPrintf(PETSC_COMM_WORLD,"\nsetup: %lf\nsolve: %lf\n", tm_setup, tm_solve);
    }

    VecDestroy(&x);
    VecDestroy(&f);
    MatDestroy(&A);
    KSPDestroy(&solver);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank == 0) {
        log_times("petsc.txt", size, n, iters, tm_setup, tm_solve);
    }

    PetscFinalize();
}
