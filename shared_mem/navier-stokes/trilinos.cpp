#include <iostream>
#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>

#include <boost/range.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>

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
#include <Epetra_VbrMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Time.h>
#include <AztecOO.h>
#include <AztecOO_string_maps.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <ml_epetra_preconditioner.h>

#include "log_times.hpp"

using namespace Teuchos;
using namespace amgcl;

//---------------------------------------------------------------------------
std::tuple<
    std::shared_ptr<Epetra_Map>,
    std::shared_ptr<Epetra_CrsMatrix>,
    std::shared_ptr<Epetra_Vector>
    >
read_problem(const Epetra_MpiComm &Comm,
        std::string A_file, std::string f_file, std::string p_file)
{
    typedef std::ifstream::off_type off_type;
    namespace io = amgcl::io;

    const int mpi_rank = Comm.MyPID();
    const int mpi_size = Comm.NumProc();

    // Read partition
    int n,m;
    std::vector<int> part;
    std::tie(n, m) = io::mm_reader(p_file)(part);

    TEUCHOS_ASSERT_EQUALITY(m, 1);

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

    auto M = std::make_shared<Epetra_Map>(n, chunk, 0, Comm);

    // Read our chunk of the matrix
    std::ifstream af(A_file, std::ios::binary);
    size_t rows;
    af.read((char*)&rows, sizeof(size_t));

    TEUCHOS_ASSERT_EQUALITY(rows, n);

    std::vector<int> ptr(chunk);
    std::vector<int> nnz(chunk);

    auto ptr_pos = af.tellg();

    ptrdiff_t glob_nnz;
    {
        af.seekg(ptr_pos + off_type(n * sizeof(ptrdiff_t)));
        af.read((char*)&glob_nnz, sizeof(ptrdiff_t));

        if (mpi_rank == 0) {
            std::cout << "global nnz: " << glob_nnz << std::endl;
        }
    }

    auto col_pos = ptr_pos + off_type((n + 1) * sizeof(ptrdiff_t));
    auto val_pos = col_pos + off_type(glob_nnz * sizeof(ptrdiff_t));

    for(int i = 0, j = 0; i < n; ++i) {
        if (part[i] == mpi_rank) {
            TEUCHOS_ASSERT_EQUALITY(perm[i] - domain[mpi_rank], j);

            af.seekg(ptr_pos + off_type(i * sizeof(ptrdiff_t)));

            ptrdiff_t p[2];
            af.read((char*)p, sizeof(p));

            ptr[j] = p[0];
            nnz[j] = p[1] - p[0];

            ++j;
        }
    }

    auto A = std::make_shared<Epetra_CrsMatrix>(Copy, *M, nnz.data(), true);

    std::vector<int>    col; col.reserve(128);
    std::vector<double> val; val.reserve(128);

    for(int i = 0; i < chunk; ++i, col.clear(), val.clear()) {
        af.seekg(col_pos + off_type(ptr[i] * sizeof(ptrdiff_t)));
        for(int j = 0; j < nnz[i]; ++j) {
            ptrdiff_t c;
            af.read((char*)&c, sizeof(ptrdiff_t));
            col.push_back(perm[c]);
        }

        af.seekg(val_pos + off_type(ptr[i] * sizeof(double)));
        val.resize(nnz[i]);
        af.read((char*)val.data(), nnz[i] * sizeof(double));

        A->InsertGlobalValues(i + domain[mpi_rank], nnz[i], val.data(), col.data());
    }

    A->FillComplete();

    // Read our chunk of the RHS
    auto F = std::make_shared<Epetra_Vector>(*M);
    std::ifstream ff(f_file, std::ios::binary);
    {
        size_t shape[2];
        ff.read((char*)shape, sizeof(shape));

        TEUCHOS_ASSERT_EQUALITY(shape[0], n);
        TEUCHOS_ASSERT_EQUALITY(shape[1], 1);
    }
    auto f_pos = ff.tellg();

    for(int i = 0, j = 0; i < n; ++i) {
        if (part[i] == mpi_rank) {
            ff.seekg(f_pos + off_type(i * sizeof(double)));

            double v;
            ff.read((char*)&v, sizeof(double));

            (*F)[j++] = v;
        }
    }

    return std::make_tuple(M, A, F);
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
#ifdef EPETRA_MPI
    MPI_Init(&argc,&argv);
#endif

    {
#ifdef EPETRA_MPI
        Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
        Epetra_SerialComm Comm;
#endif

        std::string A_file = "A.bin";
        std::string f_file = "b.bin";
        std::string p_file = "part-" + std::to_string(Comm.NumProc()) + ".mtx";
        double tol = 1e-4;

        CommandLineProcessor CLP;
        CLP.setOption("matrix", &A_file, "matrix file name");
        CLP.setOption("RHS", &f_file, "RHS file name");
        CLP.setOption("part", &p_file, "part file name");
        CLP.setOption("tol", &tol, "tolerance");
        CLP.parse(argc, argv);

        Epetra_Time Time(Comm);
        Time.ResetStartTime();

        std::shared_ptr<Epetra_Map>       M;
        std::shared_ptr<Epetra_CrsMatrix> A;
        std::shared_ptr<Epetra_Vector>    f;

        std::tie(M, A, f) = read_problem(Comm, A_file, f_file, p_file);
        double tm_read = Time.ElapsedTime();

        Epetra_Vector x(*M);
        Epetra_LinearProblem Problem(A.get(), &x, f.get());

        // Construct a solver object for this problem
        Time.ResetStartTime();
        AztecOO Solver(Problem);

        Teuchos::ParameterList MLList;
        //set multigrid defaults based on problem type
        //  SA is appropriate for Laplace-like systems
        //  NSSA is appropriate for nonsymmetric problems such as convection-diffusion
        ML_Epetra::SetDefaults("NSSA", MLList);
        MLList.set("ML output", 1);
        MLList.set("max levels", 3);
        MLList.set("PDE equations", 4);
        MLList.set("aggregation: type", "Uncoupled");

        // create the preconditioner object based on options in MLList and compute hierarchy
        ML_Epetra::MultiLevelPreconditioner MLPrec(*A, MLList);

        // tell AztecOO to use this preconditioner, then solve
        Solver.SetPrecOperator(&MLPrec);

        Solver.SetAztecOption(AZ_solver, AZ_gmres);
        Solver.SetAztecOption(AZ_output, 1);
        Solver.SetAztecOption(AZ_kspace, 200);
        double tm_setup = Time.ElapsedTime();

        Time.ResetStartTime();
        Solver.Iterate(1000, tol);
        double tm_solve = Time.ElapsedTime();

        if( Comm.MyPID() == 0 ) {
            std::cout << "read: "  << tm_read  << std::endl;
            std::cout << "setup: " << tm_setup << std::endl;
            std::cout << "solve: " << tm_solve << std::endl;

            log_times("trilinos.txt", Comm.NumProc(), M->NumGlobalPoints(), Solver.NumIters(), tm_setup, tm_solve);
        }
    }

#ifdef EPETRA_MPI
    MPI_Finalize() ;
#endif
}
