#include <iostream>

#include <Epetra_ConfigDefs.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#  include <Epetra_MpiComm.h>
#else
#  include <Epetra_SerialComm.h>
#endif
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Time.h>
#include <AztecOO.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <ml_epetra_preconditioner.h>

#include "log_times.hpp"

using namespace Teuchos;

//---------------------------------------------------------------------------
void assemble(int n, Epetra_CrsMatrix &A, Epetra_Vector &f, Epetra_Vector &x)
{
    std::vector<int>    col; col.reserve(7);
    std::vector<double> val; val.reserve(7);

    for(int row = 0; row < A.RowMap().NumMyElements(); ++row, col.clear(), val.clear()) {
        int idx = A.RowMap().GID(row);

        int i = idx % n;
        int j = (idx / n) % n;
        int k = idx / (n * n);

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

        f[row] = 1.0;
        x[row] = 0.0;

        A.InsertGlobalValues(idx, col.size(), val.data(), col.data());
    }

    A.FillComplete();
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[])
{

#ifdef EPETRA_MPI
    MPI_Init(&argc,&argv);
#endif

    {
#ifdef EPETRA_MPI
        Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
        Epetra_SerialComm Comm;
#endif

        int n = 150;

        CommandLineProcessor CLP;
        CLP.setOption("n", &n, "problem size");
        CLP.parse(argc, argv);

        Epetra_Time Time(Comm);

        // Assemble problem
        Time.ResetStartTime();
        Epetra_Map Map(n * n * n, 0, Comm);
        Epetra_Vector f(Map), x(Map);
        Epetra_CrsMatrix A(Copy, Map, 7);

        assemble(n, A, f, x);

        Epetra_LinearProblem Problem(&A, &x, &f);
        double tm_assemble = Time.ElapsedTime();

        // Construct a solver object for this problem
        Time.ResetStartTime();
        AztecOO Solver(Problem);

        Teuchos::ParameterList MLList;
        //set multigrid defaults based on problem type
        //  SA is appropriate for Laplace-like systems
        //  NSSA is appropriate for nonsymmetric problems such as convection-diffusion
        ML_Epetra::SetDefaults("SA",MLList);
        MLList.set("ML output", 10);

        // create the preconditioner object based on options in MLList and compute hierarchy
        ML_Epetra::MultiLevelPreconditioner MLPrec(A, MLList);

        // tell AztecOO to use this preconditioner, then solve
        Solver.SetPrecOperator(&MLPrec);

        Solver.SetAztecOption(AZ_solver, AZ_cg);
        Solver.SetAztecOption(AZ_output, 1);
        double tm_setup = Time.ElapsedTime();

        Time.ResetStartTime();
        Solver.Iterate(100, 1e-8);
        double tm_solve = Time.ElapsedTime();

        // print out some information about the preconditioner
        if( Comm.MyPID() == 0 ) {
            std::cout << "assemble: " << tm_assemble << std::endl;
            std::cout << "setup:    " << tm_setup    << std::endl;
            std::cout << "solve:    " << tm_solve    << std::endl;

            log_times("trilinos.txt", Comm.NumProc(), n, Solver.NumIters(), tm_setup, tm_solve);
        }
    }

#ifdef EPETRA_MPI
    MPI_Finalize() ;
#endif
}
