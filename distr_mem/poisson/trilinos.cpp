#include <iostream>
#include <numeric>

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
void assemble(const Epetra_Comm &comm, const domain_partition<3> &part,
        int n, Epetra_CrsMatrix &A, Epetra_Vector &f, Epetra_Vector &x,
        std::vector<double> &x_coo, std::vector<double> &y_coo, std::vector<double> &z_coo
        )
{
    std::vector<int>    col; col.reserve(7);
    std::vector<double> val; val.reserve(7);

    ptrdiff_t n_loc = part.size(comm.MyPID());

    std::vector<ptrdiff_t> domain(comm.NumProc() + 1, 0);
    comm.GatherAll(&n_loc, domain.data() + 1, 1);
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    renumbering renum(part, domain);

    x_coo.resize(n_loc);
    y_coo.resize(n_loc);
    z_coo.resize(n_loc);

    boost::array<ptrdiff_t, 3> lo = part.domain(comm.MyPID()).min_corner();
    boost::array<ptrdiff_t, 3> hi = part.domain(comm.MyPID()).max_corner();

    for(int k = lo[2], idx = 0; k <= hi[2]; ++k) {
        for(int j = lo[1]; j <= hi[1]; ++j) {
            for(int i = lo[0]; i <= hi[0]; ++i, ++idx) {
                col.clear();
                val.clear();

                x_coo[idx] = i;
                y_coo[idx] = j;
                z_coo[idx] = k;

                f[idx] = 1.0;
                x[idx] = 0.0;

                int row = renum(i,j,k);

                if (k > 0)  {
                    col.push_back(renum(i,j,k-1));
                    val.push_back(-1.0/6.0);
                }

                if (j > 0)  {
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

                A.InsertGlobalValues(row, col.size(), val.data(), col.data());
            }
        }
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

        Teuchos::CommandLineProcessor CLP;
        CLP.setOption("n", &n, "problem size");
        CLP.setOption("r", &rebalance, "rebalance (Zoltan/ParMETIS)");
        CLP.setOption("dd", &dd, "Use DD/DD-ML");
        CLP.parse(argc, argv);

        // Partitioning
        boost::array<ptrdiff_t, 3> lo = { {0,   0,   0  } };
        boost::array<ptrdiff_t, 3> hi = { {n-1, n-1, n-1} };

        domain_partition<3> part(lo, hi, Comm.NumProc());
        ptrdiff_t chunk = part.size( Comm.MyPID() );

        Epetra_Time Time(Comm);

        // Assemble problem
        Time.ResetStartTime();
        Epetra_Map Map(n * n * n, chunk, 0, Comm);
        Epetra_Vector f(Map), x(Map);
        Epetra_CrsMatrix A(Copy, Map, 7);

        std::vector<double> x_coo, y_coo, z_coo;

        assemble(Comm, part, n, A, f, x, x_coo, y_coo, z_coo);

        Epetra_LinearProblem Problem(&A, &x, &f);
        double tm_assemble = Time.ElapsedTime();

        // Construct a solver object for this problem
        Time.ResetStartTime();
        AztecOO Solver(Problem);

        Teuchos::ParameterList MLList;
        //set multigrid defaults based on problem type
        //  SA is appropriate for Laplace-like systems
        //  NSSA is appropriate for nonsymmetric problems such as convection-diffusion
        if (dd == 1) {
            ML_Epetra::SetDefaults("DD",MLList);
        } else if (dd == 2) {
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
        Solver.Iterate(500, 1e-8);
        double tm_solve = Time.ElapsedTime();

        // print out some information about the preconditioner
        if( Comm.MyPID() == 0 ) {
            std::cout << "assemble: " << tm_assemble << std::endl;
            std::cout << "setup:    " << tm_setup    << std::endl;
            std::cout << "solve:    " << tm_solve    << std::endl;

            std::ostringstream fname;
            fname << "trilinos";
            if (dd == 1)
                fname << "_dd";
            else if (dd == 2)
                fname << "_ddml";
            fname << ".txt";

            std::ofstream f(fname.str(), std::ios::app);
            f << Comm.NumProc() << " " << n << " " << Solver.NumIters() << " "
              << std::scientific << tm_setup << " " << tm_solve << std::endl;
        }
    }

#ifdef EPETRA_MPI
    MPI_Finalize() ;
#endif
}
