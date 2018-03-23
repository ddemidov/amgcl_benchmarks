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

using namespace Teuchos;

//---------------------------------------------------------------------------
void assemble(int n, Epetra_CrsMatrix &A, Epetra_Vector &f, Epetra_Vector &x,
        std::vector<double> &x_coo, std::vector<double> &y_coo, std::vector<double> &z_coo
        )
{
    std::vector<int>    col; col.reserve(7);
    std::vector<double> val; val.reserve(7);

    const int n_loc = A.RowMap().NumMyElements();

    x_coo.resize(n_loc);
    y_coo.resize(n_loc);
    z_coo.resize(n_loc);

    for(int row = 0; row < n_loc; ++row, col.clear(), val.clear()) {
        int idx = A.RowMap().GID(row);

        int i = idx % n;
        int j = (idx / n) % n;
        int k = idx / (n * n);

        x_coo[row] = i;
        y_coo[row] = j;
        z_coo[row] = k;

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
        int dd = 0;
        std::string rebalance;

        CommandLineProcessor CLP;
        CLP.setOption("n", &n, "problem size");
        CLP.setOption("r", &rebalance, "rebalance (Zoltan/ParMETIS)");
        CLP.setOption("dd", &dd, "Use DD-ML");
        CLP.parse(argc, argv);

        Epetra_Time Time(Comm);

        // Assemble problem
        Time.ResetStartTime();
        Epetra_Map Map(n * n * n, 0, Comm);
        Epetra_Vector f(Map), x(Map);
        Epetra_CrsMatrix A(Copy, Map, 7);

        std::vector<double> x_coo, y_coo, z_coo;

        assemble(n, A, f, x, x_coo, y_coo, z_coo);

        Epetra_LinearProblem Problem(&A, &x, &f);
        double tm_assemble = Time.ElapsedTime();

        // Construct a solver object for this problem
        Time.ResetStartTime();
        AztecOO Solver(Problem);

        Teuchos::ParameterList MLList;
        //set multigrid defaults based on problem type
        //  SA is appropriate for Laplace-like systems
        //  NSSA is appropriate for nonsymmetric problems such as convection-diffusion
        if (dd) {
            ML_Epetra::SetDefaults("DD-ML",MLList);
        } else {
            ML_Epetra::SetDefaults("SA",MLList);
        }
        MLList.set("ML output", 10);

        if (!rebalance.empty()) {
            MLList.set("repartition: enable", 1);
            MLList.set("repartition: partitioner", rebalance);
            MLList.set("repartition: max min ratio", 1.3);
            MLList.set("repartition: min per proc", 500);

            if (rebalance == "Zoltan") {
                MLList.set("repartition: Zoltan dimensions", 3);
                MLList.set("x-coordinates", x_coo.data());
                MLList.set("y-coordinates", y_coo.data());
                MLList.set("z-coordinates", z_coo.data());
            }
        }

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

            std::ofstream f(dd ? "trilinos_dd.txt" : "trilinos.txt", std::ios::app);
            f << Comm.NumProc() << " " << n << " " << Solver.NumIters() << " "
              << std::scientific << tm_setup << " " << tm_solve << std::endl;
        }
    }

#ifdef EPETRA_MPI
    MPI_Finalize() ;
#endif
}
