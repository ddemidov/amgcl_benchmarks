#include <iostream>
#include <fstream>
#include <vector>
#include <petscksp.h>
#include <petsctime.h>

#include "log_times.hpp"

//---------------------------------------------------------------------------
void assemble(PetscInt n, Mat &A, Vec &f, Vec &x) {
    PetscInt n3 = n * n * n;


    VecCreate(MPI_COMM_WORLD, &f);
    VecCreate(MPI_COMM_WORLD, &x);

    VecSetFromOptions(f);
    VecSetFromOptions(x);

    VecSetSizes(f, PETSC_DECIDE, n3);
    VecSetSizes(x, PETSC_DECIDE, n3);

    VecSet(f, 1.0);
    VecSet(x, 0.0);

    MatCreate(MPI_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n3, n3);
    MatSetFromOptions(A);
    MatMPIAIJSetPreallocation(A, 7, 0, 7, 0);
    MatSeqAIJSetPreallocation(A, 7, 0);

    PetscInt istart, iend;

    MatGetOwnershipRange(A, &istart, &iend);

    std::vector<PetscInt>    col; col.reserve(7);
    std::vector<PetscScalar> val; val.reserve(7);

    for(PetscInt idx = istart; idx < iend; ++idx, col.clear(), val.clear()) {
        PetscInt i = idx % n;
        PetscInt j = (idx / n) % n;
        PetscInt k = idx / (n * n);

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

        MatSetValues(A, 1, &idx, col.size(), col.data(), val.data(), INSERT_VALUES);
    }

    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
}

//---------------------------------------------------------------------------
int main(int argc,char **argv) {
    KSP                solver;
    PC                 prec;
    Mat                A;
    Vec                x,f;
    PetscScalar        v;
    KSPConvergedReason reason;
    PetscInt           n = 150, iters;
    PetscReal          error;
    PetscLogDouble     tic, toc;

    PetscInitialize(&argc, &argv, 0, "poisson3d");
    PetscOptionsGetInt(0, 0, "-n", &n, 0);

    assemble(n, A, f, x);

    PetscTime(&tic);
    KSPCreate(MPI_COMM_WORLD, &solver);
    KSPSetOperators(solver,A,A);
    KSPSetTolerances(solver, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 100);
    KSPSetType(solver, KSPCG);

    KSPGetPC(solver,&prec);
    PCSetType(prec, PCGAMG);
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
