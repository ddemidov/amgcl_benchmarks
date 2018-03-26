#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <petscksp.h>
#include <petsctime.h>

#include "domain_partition.hpp"

//---------------------------------------------------------------------------
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

//---------------------------------------------------------------------------
void assemble(PetscInt n, Mat &A, Vec &f, Vec &x) {
    PetscInt n3 = n * n * n;

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    boost::array<ptrdiff_t, 3> lo = { {0,   0,   0  } };
    boost::array<ptrdiff_t, 3> hi = { {n-1, n-1, n-1} };

    domain_partition<3> part(lo, hi, mpi_size);
    ptrdiff_t chunk = part.size( mpi_rank );

    std::vector<ptrdiff_t> domain(mpi_size + 1);
    MPI_Allgather(&chunk, 1, MPI_LONG, &domain[1], 1, MPI_LONG, MPI_COMM_WORLD);
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    lo = part.domain(mpi_rank).min_corner();
    hi = part.domain(mpi_rank).max_corner();

    renumbering renum(part, domain);

    VecCreate(MPI_COMM_WORLD, &f);
    VecCreate(MPI_COMM_WORLD, &x);

    VecSetFromOptions(f);
    VecSetFromOptions(x);

    VecSetSizes(f, chunk, n3);
    VecSetSizes(x, chunk, n3);

    VecSet(f, 1.0);
    VecSet(x, 0.0);

    MatCreate(MPI_COMM_WORLD, &A);
    MatSetSizes(A, chunk, chunk, n3, n3);
    MatSetFromOptions(A);
    MatMPIAIJSetPreallocation(A, 7, 0, 7, 0);
    MatSeqAIJSetPreallocation(A, 7, 0);

    std::vector<PetscInt>    col; col.reserve(7);
    std::vector<PetscScalar> val; val.reserve(7);

    for(ptrdiff_t k = lo[2]; k <= hi[2]; ++k) {
        for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
            for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {
                col.clear();
                val.clear();

                PetscInt row = renum(i,j,k);

                if (k > 0) {
                    col.push_back(renum(i,j,k-1));
                    val.push_back(-1.0/6.0);
                }

                if (j > 0) {
                    col.push_back(renum(i,j-1,k));
                    val.push_back(-1.0/6.0);
                }

                if (i > 0) {
                    col.push_back(renum(i-1,j,k));
                    val.push_back(-1.0/6.0);
                }

                col.push_back(renum(i,j,k));
                val.push_back(1.0);

                if (i + 1 < n) {
                    col.push_back(renum(i+1,j,k));
                    val.push_back(-1.0/6.0);
                }

                if (j + 1 < n) {
                    col.push_back(renum(i,j+1,k));
                    val.push_back(-1.0/6.0);
                }

                if (k + 1 < n) {
                    col.push_back(renum(i,j,k+1));
                    val.push_back(-1.0/6.0);
                }

                MatSetValues(A, 1, &row, col.size(), col.data(), val.data(), INSERT_VALUES);
            }
        }
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
        std::cout
            << "Iterations: " << iters << std::endl
            << "setup:      " << tm_setup    << std::endl
            << "solve:      " << tm_solve    << std::endl
            ;

        std::ofstream log("petsc.txt", std::ios::app);
        log << size << " " << n << " " << iters << " "
            << std::scientific << tm_setup << " " << tm_solve << std::endl;
    }

    PetscFinalize();
}
