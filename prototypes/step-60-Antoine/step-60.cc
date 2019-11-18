/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2018 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Luca Heltai, Giovanni Alzetta,
 * International School for Advanced Studies, Trieste, 2018
 */

// @sect3{Include files}

#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/non_matching/coupling.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <iostream>
#include <fstream>


namespace Step60
{
  using namespace dealii;

  // @sect3{DistributedLagrangeProblem}
  //
  // In the DistributedLagrangeProblem, we need two parameters describing the
  // dimensions of the domain $\Gamma$ (`dim`) and of the domain $\Omega$
  // (`spacedim`).
  //
  // These will be used to initialize a Triangulation<dim,spacedim> (for
  // $\Gamma$) and a Triangulation<spacedim,spacedim> (for $\Omega$).
  //
  //
  // Here we assume that upon construction, the classes that build up our
  // problem are not usable yet. Parsing the parameter file is what ensures we
  // have all ingredients to build up our classes, and we design them so that if
  // parsing fails, or is not executed, the run is aborted.
  

  template <int dim, int spacedim = dim>
  class DistributedLagrangeProblem
  {
  public:
    // The `Parameters` class is derived from ParameterAcceptor. This allows us
    // to use the ParameterAcceptor::add_parameter() method in its constructor.
    class Parameters : public ParameterAcceptor


    {
    public:
      Parameters();
      // Initial refinement for the embedding grid, corresponding to the domain
      // $\Omega$.
      unsigned int initial_refinement = 4;

      // For this reason we define `delta_refinement`: if it is greater
      // than zero, then we mark each cell of the space grid that contains
      // a vertex of the embedded grid and its neighbors, execute the
      // refinement, and repeat this process `delta_refinement` times.
      unsigned int delta_refinement = 3;

      // Starting refinement of the embedded grid, corresponding to the domain
      // $\Gamma$.
      unsigned int initial_embedded_refinement = 8;

      // The list of boundary ids where we impose homogeneous Dirichlet boundary
      // conditions. On the remaining boundary ids (if any), we impose
      // homogeneous Neumann boundary conditions.
      // As a default problem we have zero Dirichlet boundary conditions on
      // $\partial \Omega$
      std::list<types::boundary_id> homogeneous_dirichlet_ids{0, 1, 2, 3};

      // FiniteElement degree of the embedding space: $V_h(\Omega)$
      unsigned int embedding_space_finite_element_degree = 1;

      // FiniteElement degree of the embedded space: $Q_h(\Gamma)$
      unsigned int embedded_space_finite_element_degree = 1;

      // FiniteElement degree of the space used to describe the deformation
      // of the embedded domain
      unsigned int embedded_configuration_finite_element_degree = 1;

      // Order of the quadrature formula used to integrate the coupling
      unsigned int coupling_quadrature_order = 3;

      // If set to true, then the embedded configuration function is
      // interpreted as a displacement function
      bool use_displacement = false;

      // Level of verbosity to use in the output
      unsigned int verbosity_level = 10;

      // A flag to keep track if we were initialized or not
      bool initialized = false;


    };

    DistributedLagrangeProblem(const Parameters &parameters);

    // Entry point for the DistributedLagrangeProblem
    void run();

  private:
    // Object containing the actual parameters
    const Parameters &parameters;

    void setup_grids_and_dofs();
    void setup_embedding_dofs();
    void setup_embedded_dofs();

    void setup_coupling();
    void assemble_system();
    void solve();

    void L2_error();

    void output_results();


    // first we gather all the objects related to the embedding space geometry

    std::unique_ptr<Triangulation<spacedim>> space_grid;
    std::unique_ptr<GridTools::Cache<spacedim, spacedim>>
                                             space_grid_tools_cache;
    std::unique_ptr<FiniteElement<spacedim>> space_fe;
    std::unique_ptr<DoFHandler<spacedim>>    space_dh;

    // Then the ones related to the embedded grid, with the DoFHandler
    // associated to the Lagrange multiplier `lambda`
    //std::unique_ptr<Triangulation<dim, spacedim>> embedded_grid;

    //std::unique_ptr<Triangulation<dim, spacedim>> embedded_grid1;
    //std::unique_ptr<Triangulation<dim, spacedim>> embedded_gridn;

    Triangulation<dim, spacedim> embedded_grid;


    std::unique_ptr<FiniteElement<dim, spacedim>> embedded_fe;
    std::unique_ptr<DoFHandler<dim, spacedim>>    embedded_dh;

    // And finally, everything that is needed to *deform* the embedded
    // triangulation
    std::unique_ptr<FiniteElement<dim, spacedim>> embedded_configuration_fe;
    std::unique_ptr<DoFHandler<dim, spacedim>>    embedded_configuration_dh;
    Vector<double>                                embedded_configuration;

    // The ParameterAcceptorProxy class is a "transparent" wrapper derived
    // from both ParameterAcceptor and the type passed as its template
    // parameter.
    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_configuration_function;

    std::unique_ptr<Mapping<dim, spacedim>> embedded_mapping;

    // We do the same thing to specify the value of the function $g$
    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_value_function;

    // Specify all possible stopping criteria for the Schur complement
    // iterative solver we'll use later on.
    ParameterAcceptorProxy<ReductionControl> schur_solver_control;

    // Next we gather all SparsityPattern, SparseMatrix, and Vector objects
    // we'll need
    SparsityPattern stiffness_sparsity;
    SparsityPattern coupling_sparsity;

    SparseMatrix<double> stiffness_matrix;
    SparseMatrix<double> coupling_matrix;

    AffineConstraints<double> constraints;

    Vector<double> solution;
    Vector<double> rhs;

    Vector<double> lambda;
    Vector<double> embedded_rhs;
    Vector<double> embedded_value;

    // The TimerOutput class is used to provide some statistics on
    // the performance of our program.
    TimerOutput monitor;
  };

  // @sect3{DistributedLagrangeProblem::Parameters}
  //
  // At construction time, we initialize also the ParameterAcceptor class, with
  // the section name we want our problem to use when parsing the parameter
  // file.
  template <int dim, int spacedim>
  DistributedLagrangeProblem<dim, spacedim>::Parameters::Parameters()
    : ParameterAcceptor("/Distributed Lagrange<" +
                        Utilities::int_to_string(dim) + "," +
                        Utilities::int_to_string(spacedim) + ">/")
  {
    // The ParameterAcceptor::add_parameter() function :
    add_parameter("Initial embedding space refinement", initial_refinement);
    add_parameter("Initial embedded space refinement",
                  initial_embedded_refinement);
    add_parameter("Local refinements steps near embedded domain",
                  delta_refinement);
    add_parameter("Homogeneous Dirichlet boundary ids",
                  homogeneous_dirichlet_ids);
    add_parameter("Use displacement in embedded interface", use_displacement);
    add_parameter("Embedding space finite element degree",
                  embedding_space_finite_element_degree);
    add_parameter("Embedded space finite element degree",
                  embedded_space_finite_element_degree);
    add_parameter("Embedded configuration finite element degree",
                  embedded_configuration_finite_element_degree);
    add_parameter("Coupling quadrature order", coupling_quadrature_order);
    add_parameter("Verbosity level", verbosity_level);

    // Once the parameter file has been parsed, then the parameters are good to
    // go. Set the internal variable `initialized` to true.
    parse_parameters_call_back.connect([&]() -> void { initialized = true; });
  }

  // The constructor is pretty standard, with the exception of the
  // `ParameterAcceptorProxy` objects, as explained earlier.
  template <int dim, int spacedim>
  DistributedLagrangeProblem<dim, spacedim>::DistributedLagrangeProblem(
    const Parameters &parameters)
    : parameters(parameters)
    , embedded_configuration_function("Embedded configuration", spacedim)
    , embedded_value_function("Embedded value")
    , schur_solver_control("Schur solver control")
    , monitor(std::cout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  {
    // Here is a way to set default values for a ParameterAcceptor class
    // that was constructed using ParameterAcceptorProxy.
    embedded_configuration_function.declare_parameters_call_back.connect(
      []() -> void {
        ParameterAcceptor::prm.set("Function constants", "R=.3, Cx=.4, Cy=.4");
        ParameterAcceptor::prm.set("Function expression",
                                   "R*cos(2*pi*x)+Cx; R*sin(2*pi*x)+Cy");
      });
    embedded_value_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "1"); });
    schur_solver_control.declare_parameters_call_back.connect([]() -> void {
      ParameterAcceptor::prm.set("Max steps", "1000");
      ParameterAcceptor::prm.set("Reduction", "1.e-12");
      ParameterAcceptor::prm.set("Tolerance", "1.e-12");
    });
  }

  // @sect3{Set up}
  //
  // The function `DistributedLagrangeProblem::setup_grids_and_dofs()` is used
  // to set up the finite element spaces.
  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::setup_grids_and_dofs()
  {
    TimerOutput::Scope timer_section(monitor, "Setup grids and dofs");

    // Initializing $\Omega$: constructing the Triangulation and wrapping it
    // into a `std::unique_ptr` object
    space_grid = std_cxx14::make_unique<Triangulation<spacedim>>();
    const Point<spacedim> centerpoint_embedding(0.0, 0.0);
    GridGenerator::hyper_ball(*space_grid, centerpoint_embedding, 1, false);


    

   
    // Once we constructed a Triangulation, we refine it globally according to
    // the specifications in the parameter file, and construct a
    // GridTools::Cache with it.
    space_grid->refine_global(parameters.initial_refinement);
    space_grid_tools_cache =
      std_cxx14::make_unique<GridTools::Cache<spacedim, spacedim>>(*space_grid);

    // The same is done with the embedded grid. Since the embedded grid is
    // deformed, we first need to setup the deformation mapping. We do so in the
    // following few lines:
    const unsigned int nbelem = 2;

    Triangulation<dim, spacedim> embedded_grid1;
    const Point<spacedim> centerpoint_elem1(0.35, 0.35);
    GridGenerator::hyper_sphere(embedded_grid1, centerpoint_elem1, 0.2);

    Triangulation<dim, spacedim> embedded_gridn;
    const Point<spacedim> centerpoint_elemn(-0.35, -0.35);
    GridGenerator::hyper_sphere(embedded_gridn, centerpoint_elemn, 0.2);


    GridGenerator::merge_triangulations(embedded_grid1, embedded_gridn, embedded_grid);



    

    if (nbelem ==1) {
      
    }
    else {
      

    }

    embedded_grid.refine_global(parameters.initial_embedded_refinement);

    



    

    

    embedded_configuration_fe = std_cxx14::make_unique<FESystem<dim, spacedim>>(
      FE_Q<dim, spacedim>(
        parameters.embedded_configuration_finite_element_degree),
      spacedim);

    embedded_configuration_dh =
      std_cxx14::make_unique<DoFHandler<dim, spacedim>>(embedded_grid);

    embedded_configuration_dh->distribute_dofs(*embedded_configuration_fe);
    embedded_configuration.reinit(embedded_configuration_dh->n_dofs());

    // Once we have defined a finite dimensional space for the deformation, we
    // interpolate the `embedded_configuration_function` defined in the
    // parameter file:
    VectorTools::interpolate(*embedded_configuration_dh,
                             embedded_configuration_function,
                             embedded_configuration);

    // Now we can interpret it according to what the user has specified in the
    // parameter file: as a displacement, in which case we construct a mapping
    // that *displaces* the position of each support point of our configuration
    // finite element space by the specified amount on the corresponding
    // configuration vector, or as an absolution position.
    if (parameters.use_displacement == true) {
      embedded_mapping =
        std_cxx14::make_unique<MappingQEulerian<dim, Vector<double>, spacedim>>(
          parameters.embedded_configuration_finite_element_degree,
          *embedded_configuration_dh,
          embedded_configuration);
    }
    else {
      embedded_mapping =
        std_cxx14::make_unique<MappingFEField<dim,
                                              spacedim,
                                              Vector<double>,
                                              DoFHandler<dim, spacedim>>>(
          *embedded_configuration_dh, embedded_configuration);
    }
    setup_embedded_dofs();

    // In this tutorial program we not only refine $\Omega$ globally,
    // but also allow a local refinement depending on the position of $\Gamma$,
    // according to the value of `parameters.delta_refinement`, that we use to
    // decide how many rounds of local refinement we should do on $\Omega$,
    // corresponding to the position of $\Gamma$.
    //
    // With the mapping in place, it is now possible to query what is the
    // location of all support points associated with the `embedded_dh`, by
    // calling the method DoFTools::map_dofs_to_support_points.
    //
    // This method has two variants. One that does *not* take a Mapping, and
    // one that takes a Mapping. If you use the second type, like we are doing
    // in this case, the support points are computed through the specified
    // mapping, which can manipulate them accordingly.
    //
    // This is precisely what the `embedded_mapping` is there for.
    std::vector<Point<spacedim>> support_points(embedded_dh->n_dofs());
    if (parameters.delta_refinement != 0)  // && spacedim == 2
      DoFTools::map_dofs_to_support_points(*embedded_mapping,
                                           *embedded_dh,
                                           support_points);

    // Once we have the support points of the embedded finite element space, we
    // would like to identify what cells of the embedding space contain what
    // support point, to get a chance at refining the embedding grid where it is
    // necessary, i.e., where the embedded grid is. 

    // This is only possible if we ensure that the smallest cell size of the
    // embedding grid is nonetheless bigger than the largest cell size of the
    // embedded grid. 
    for (unsigned int i = 0; i < parameters.delta_refinement; ++i)
      {
        const auto point_locations =
          GridTools::compute_point_locations(*space_grid_tools_cache,
                                             support_points);
        // Commenter ces lignes enlève le maillage raffiné autour du "embedded", si on ne commente pas aussi const auto &cells, on a un warning et un ralentissement du script.
        int localrefinnement = 0;
        if (localrefinnement == 1) {
          const auto &cells = std::get<0>(point_locations);

          for (auto &cell : cells)
            {
              cell->set_refine_flag();
              for (unsigned int face_no = 0;
                   face_no < GeometryInfo<spacedim>::faces_per_cell;
                   ++face_no)
                if (!cell->at_boundary(face_no))
                  {
                    auto neighbor = cell->neighbor(face_no);
                    neighbor->set_refine_flag();
                  }
            }
        }
        space_grid->execute_coarsening_and_refinement();
      }

    // In order to avoid issues, in this tutorial we will throw an exception if
    // the parameters chosen by the user are such that the maximal diameter of
    // the embedded grid is greater than the minimal diameter of the embedding
    // grid.
    //
    // This choice guarantees that almost every cell of the embedded grid spans
    // no more than two cells of the embedding grid, with some rare exceptions,
    // that are negligible in terms of the resulting inf-sup.
    
    const double embedded_space_maximal_diameter =
      GridTools::maximal_cell_diameter(embedded_grid, *embedded_mapping);
    
    double embedding_space_minimal_diameter =
      GridTools::minimal_cell_diameter(*space_grid);

    deallog << "Embedding minimal diameter: "
            << embedding_space_minimal_diameter
            << ", embedded maximal diameter: "
            << embedded_space_maximal_diameter << ", ratio: "
            << embedded_space_maximal_diameter /
                 embedding_space_minimal_diameter
            << std::endl;

    AssertThrow(embedded_space_maximal_diameter <
                  embedding_space_minimal_diameter,
                ExcMessage(
                  "The embedding grid is too refined (or the embedded grid "
                  "is too coarse). Adjust the parameters so that the minimal "
                  "grid size of the embedding grid is larger "
                  "than the maximal grid size of the embedded grid."));

    // $\Omega$ has been refined and we can now set up its DoFs
    setup_embedding_dofs();
  }

  // We now set up the DoFs of $\Omega$ and $\Gamma$
  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::setup_embedding_dofs()
  {
    space_dh = std_cxx14::make_unique<DoFHandler<spacedim>>(*space_grid);
    space_fe = std_cxx14::make_unique<FE_Q<spacedim>>(
      parameters.embedding_space_finite_element_degree);
    space_dh->distribute_dofs(*space_fe);

    DoFTools::make_hanging_node_constraints(*space_dh, constraints);
    for (auto id : parameters.homogeneous_dirichlet_ids)
      {
        VectorTools::interpolate_boundary_values(
          *space_dh, id, Functions::ZeroFunction<spacedim>(), constraints);
      }
    constraints.close();

    // By definition the stiffness matrix involves only $\Omega$'s DoFs
    DynamicSparsityPattern dsp(space_dh->n_dofs(), space_dh->n_dofs());
    DoFTools::make_sparsity_pattern(*space_dh, dsp, constraints);
    stiffness_sparsity.copy_from(dsp);
    stiffness_matrix.reinit(stiffness_sparsity);
    solution.reinit(space_dh->n_dofs());
    rhs.reinit(space_dh->n_dofs());

    deallog << "Embedding dofs: " << space_dh->n_dofs() << std::endl;
  }

  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::setup_embedded_dofs()
  {
    
    embedded_dh =
      std_cxx14::make_unique<DoFHandler<dim, spacedim>>(embedded_grid);
    
    embedded_fe = std_cxx14::make_unique<FE_Q<dim, spacedim>>(
      parameters.embedded_space_finite_element_degree);
    embedded_dh->distribute_dofs(*embedded_fe);

    // By definition the rhs of the system we're solving involves only a zero
    // vector and $G$, which is computed using only $\Gamma$'s DoFs
    lambda.reinit(embedded_dh->n_dofs());
    embedded_rhs.reinit(embedded_dh->n_dofs());
    embedded_value.reinit(embedded_dh->n_dofs());

    deallog << "Embedded dofs: " << embedded_dh->n_dofs() << std::endl;
  }

  // Creating the coupling sparsity pattern is a complex operation,
  // but it can be easily done using the
  // NonMatching::create_coupling_sparsity_pattern
  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::setup_coupling()
  {
    TimerOutput::Scope timer_section(monitor, "Setup coupling");

    QGauss<dim> quad(parameters.coupling_quadrature_order);

    DynamicSparsityPattern dsp(space_dh->n_dofs(), embedded_dh->n_dofs());

    NonMatching::create_coupling_sparsity_pattern(*space_grid_tools_cache,
                                                  *space_dh,
                                                  *embedded_dh,
                                                  quad,
                                                  dsp,
                                                  AffineConstraints<double>(),
                                                  ComponentMask(),
                                                  ComponentMask(),
                                                  *embedded_mapping);
    coupling_sparsity.copy_from(dsp);
    coupling_matrix.reinit(coupling_sparsity);
  }

  // @sect3{Assembly}
  //
  // The following function creates the matrices: as noted before computing the
  // stiffness matrix and the rhs is a standard procedure.
  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::assemble_system()
  {
    {
      TimerOutput::Scope timer_section(monitor, "Assemble system");

      // Embedding stiffness matrix $K$, and the right hand side $G$.
      MatrixTools::create_laplace_matrix(
        *space_dh,
        QGauss<spacedim>(2 * space_fe->degree + 1),
        stiffness_matrix,
        static_cast<const Function<spacedim> *>(nullptr),
        constraints);

      VectorTools::create_right_hand_side(*embedded_mapping,
                                          *embedded_dh,
                                          QGauss<dim>(2 * embedded_fe->degree +
                                                      1),
                                          embedded_value_function,
                                          embedded_rhs);
    }
    {
      TimerOutput::Scope timer_section(monitor, "Assemble coupling system");

      // To compute the coupling matrix we use the
      // NonMatching::create_coupling_mass_matrix tool, which works similarly to
      // NonMatching::create_coupling_sparsity_pattern.
      QGauss<dim> quad(parameters.coupling_quadrature_order);
      NonMatching::create_coupling_mass_matrix(*space_grid_tools_cache,
                                               *space_dh,
                                               *embedded_dh,
                                               quad,
                                               coupling_matrix,
                                               AffineConstraints<double>(),
                                               ComponentMask(),
                                               ComponentMask(),
                                               *embedded_mapping);

      VectorTools::interpolate(*embedded_mapping,
                               *embedded_dh,
                               embedded_value_function,
                               embedded_value);
    }
  }

  // @sect3{Solve}
  //
  // All parts have been assembled: we solve the system
  // using the Schur complement method
  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::solve()
  {
    TimerOutput::Scope timer_section(monitor, "Solve system");

    // Start by creating the inverse stiffness matrix
    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    // Initializing the operators, as described in the introduction
    auto K  = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C  = transpose_operator(Ct);

    auto K_inv = linear_operator(K, K_inv_umfpack);

    // Using the Schur complement method
    auto                     S = C * K_inv * Ct;
    SolverCG<Vector<double>> solver_cg(schur_solver_control);
    auto S_inv = inverse_operator(S, solver_cg, PreconditionIdentity());

    lambda = S_inv * embedded_rhs;

    solution = K_inv * Ct * lambda;

    constraints.distribute(solution);
  }

  // Classe de la solution analytique, son calcul et son gradient
  // Setup de la solution comme une classe unique
  template <int spacedim>
  class AnalyticSolution : public Function<spacedim>
  {
  public:
    AnalyticSolution()
      : Function<spacedim>()
    {}

    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override;

    //virtual Tensor<1, spacedim>
    //gradient(const Point<spacedim> & p,
    //         const unsigned int component = 0) const override;
  };    

  // Calcul numérique de la solution telle qu'elle est réellement (calcul purement mathématique)
  template <int spacedim>
  double AnalyticSolution<spacedim>::value(const Point<spacedim> &p, const unsigned int) const
  {
    double R1 = 0.2; // rayon interne
    double R2 = 1.0; // rayon externe
    double r = 0;
    double T = 0;

    // Calcul du rayon, il est ici assumé que le domaine est centré autour de 0
    if (spacedim==2) {
      r = std::sqrt((p[0]*p[0]+p[1]*p[1]));
    }
    else if (spacedim==3) {
       r = std::sqrt((p[0]*p[0]+p[1]*p[1]+p[2]*p[2]));
    }

    if (r <= R1) {
      T = 1;
    }
    else {
      if (spacedim == 2) {
        T = std::log(r/R2)/std::log(R1/R2);  
      }
      else if (spacedim == 3) {
        T = (R1/(R1-R2))*(1-(R2/r));
      }
    }  

   
    return T;
  }

  double l2error = 0;
  // Calcul de l'erreur de la norme L2
  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::L2_error()
  {
    // setup points gauss et FE values
    QGauss<spacedim> quadrature_formula(5);
    const MappingQ<spacedim> mapping(5, true);
    FEValues<spacedim> fe_values(mapping, *space_fe, quadrature_formula, update_values |update_quadrature_points | update_JxW_values);

    // obtenir les dofs/cell
    const unsigned int dofs_per_cell = this->space_fe->dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // nombre de points de quadrature
    const unsigned int n_q_points = quadrature_formula.size();
    
    
    // setup la solution analytique
    std::vector<double> local_temperature_values(n_q_points);
  
    double l2errorU = 0.;
    // Loop au travers des éléments pour tester le L2 error
    typename DoFHandler<spacedim>::active_cell_iterator cell = this->space_dh->begin_active(), endc = this->space_dh->end();
  
    AnalyticSolution<spacedim> AS;
    for (;cell != endc; ++cell) {
      fe_values.reinit(cell);
      fe_values.get_function_values(solution, local_temperature_values);
  
        cell->get_dof_indices(local_dof_indices);
  
      // loop au travers des vraies valeurs (local_temperature_values placeholder) et comparant à celles de la solution analytique
      for (unsigned int q = 0; q < n_q_points; q++) {
        double u_approx = local_temperature_values[q];
        Point<spacedim> current_point = fe_values.quadrature_point(q);
        double u_exact = AS.value(current_point,q);
  
        l2errorU += pow((u_approx - u_exact), 2) * fe_values.JxW(q);
      }
    }
    l2error = sqrt(l2errorU);
  }

  // The following function simply generates standard result output on two
  // separate files, one for each mesh.
  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::output_results()
  {
    TimerOutput::Scope timer_section(monitor, "Output results");

    DataOut<spacedim> embedding_out;

    std::ofstream embedding_out_file("embedding.vtu");

    embedding_out.attach_dof_handler(*space_dh);
    embedding_out.add_data_vector(solution, "solution");
    embedding_out.build_patches(
      parameters.embedding_space_finite_element_degree);
    embedding_out.write_vtu(embedding_out_file);

    // The only difference between the two output routines is that in the
    // second case, we want to output the data on the current configuration, and
    // not on the reference one. This is possible by passing the actual
    // embedded_mapping to the DataOut::build_patches function. The mapping will
    // take care of outputting the result on the actual deformed configuration.

    DataOut<dim, DoFHandler<dim, spacedim>> embedded_out;

    std::ofstream embedded_out_file("embedded.vtu");

    embedded_out.attach_dof_handler(*embedded_dh);
    embedded_out.add_data_vector(lambda, "lambda");
    embedded_out.add_data_vector(embedded_value, "g");
    embedded_out.build_patches(*embedded_mapping,
                               parameters.embedded_space_finite_element_degree);
    embedded_out.write_vtu(embedded_out_file);

    std::cout<< "L2 error = " << l2error << std::endl;
  }

  // Similar to all other tutorial programs, the `run()` function simply calls
  // all other methods in the correct order. Nothing special to note, except
  // that we check if parsing was done before we actually attempt to run our
  // program.
  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::run()
  {
    AssertThrow(parameters.initialized, ExcNotInitialized());
    deallog.depth_console(parameters.verbosity_level);

    setup_grids_and_dofs();
    setup_coupling();
    assemble_system();
    solve();
    L2_error();
    output_results();
  }
} // namespace Step60



int main(int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace Step60;

      const unsigned int dim = 1, spacedim = 2;
      

      // Differently to what happens in other tutorial programs, here we use
      // ParameterAcceptor style of initialization, i.e., all objects are first
      // constructed, and then a single call to the static method
      // ParameterAcceptor::initialize is issued to fill all parameters of the
      // classes that are derived from ParameterAcceptor.

      DistributedLagrangeProblem<dim, spacedim>::Parameters parameters;
      DistributedLagrangeProblem<dim, spacedim>             problem(parameters);

      std::string parameter_file;
      if (argc > 1)
        parameter_file = argv[1];
      else
        parameter_file = "parameters.prm";

      ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
