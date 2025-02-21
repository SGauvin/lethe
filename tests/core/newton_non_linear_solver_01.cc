#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <core/newton_non_linear_solver.h>
#include <core/physics_solver.h>
#include <core/simulation_control.h>

#include <iostream>
#include <memory>

#include "../tests.h"
#include "non_linear_test_system_01.h"

/**
 * @brief The TestClass tests the non-linear solvers using a simple systme of two
 * equations, only one of which is non-linear
 */

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, numbers::invalid_unsigned_int);
  initlog();

  Parameters::NonLinearSolver params{
    Parameters::Verbosity::quiet,
    Parameters::NonLinearSolver::SolverType::newton,
    1e-8, // tolerance
    10,   // maxIter
    4,    // display precision
    1     // skip iterations
  };

  deallog << "Creating solver" << std::endl;

  // Create an instantiation of the Test Class
  std::unique_ptr<TestClass> solver = std::make_unique<TestClass>(params);


  deallog << "Solving non-linear system " << std::endl;
  // Solve the non-linear system of equation
  solver->solve_non_linear_system(
    Parameters::SimulationControl::TimeSteppingMethod::steady, true, true);


  deallog << "The final solution is : " << solver->get_present_solution()
          << std::endl;
}
