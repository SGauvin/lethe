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


#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/coupling.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

namespace Step60
{
	using namespace dealii;

	template <int dim0, int dim1, int spacedim>
	std::pair<std::vector<Point<spacedim>>, std::vector<unsigned int>>
		qpoints_over_locally_owned_cells(
			const GridTools::Cache<dim0, spacedim> &cache,
			const DoFHandler<dim1, spacedim> &      immersed_dh,
			const Quadrature<dim1> &                quad,
			const Mapping<dim1, spacedim> &         immersed_mapping,
			const bool                              tria_is_parallel)
	{
		const auto &                 immersed_fe = immersed_dh.get_fe();
		std::vector<Point<spacedim>> points_over_local_cells;
		// Keep track of which cells we actually used
		std::vector<unsigned int> used_cells_ids;
		{
			FEValues<dim1, spacedim> fe_v(immersed_mapping,
				immersed_fe,
				quad,
				update_quadrature_points);
			unsigned int             cell_id = 0;
			for (const auto &cell : immersed_dh.active_cell_iterators())
			{
				bool use_cell = false;
				if (tria_is_parallel)
				{
					const auto bbox = cell->bounding_box();
					std::vector<std::pair<
						BoundingBox<spacedim>,
						typename Triangulation<dim0, spacedim>::active_cell_iterator>>
						out_vals;
					cache.get_cell_bounding_boxes_rtree().query(
						boost::geometry::index::intersects(bbox),
						std::back_inserter(out_vals));
					// Each bounding box corresponds to an active cell
					// of the embedding triangulation: we now check if
					// the current cell, of the embedded triangulation,
					// overlaps a locally owned cell of the embedding one
					for (const auto &bbox_it : out_vals)
						if (bbox_it.second->is_locally_owned())
						{
							use_cell = true;
							used_cells_ids.emplace_back(cell_id);
							break;
						}
				}
				else
					// for sequential triangulations, simply use all cells
					use_cell = true;

				if (use_cell)
				{
					// Reinitialize the cell and the fe_values
					fe_v.reinit(cell);
					const std::vector<Point<spacedim>> &x_points =
						fe_v.get_quadrature_points();

					// Insert the points to the vector
					points_over_local_cells.insert(points_over_local_cells.end(),
						x_points.begin(),
						x_points.end());
				}
				++cell_id;
			}
		}
		return { std::move(points_over_local_cells), std::move(used_cells_ids) };
	}

	template <int dim, int spacedim = dim>
	class DistributedLagrangeProblem
	{
	public:
		class Parameters : public ParameterAcceptor
		{
		public:
			Parameters();


			// Initial refinement for the embedding grid, corresponding to the domain
			// $\Omega$.
			unsigned int initial_refinement = 4;

			// The interaction between the embedded grid $\Omega$ and the embedding
			// grid $\Gamma$ is handled through the computation of $C$, which
			// involves all cells of $\Omega$ overlapping with parts of $\Gamma$:
			// a higher refinement of such cells might improve quality of our
			// computations.
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
			std::list<types::boundary_id> homogeneous_dirichlet_ids{ 0, 1, 2, 3 };

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

			// Beta for the Nitsche method
			double beta = 10.;

      // Paramètres en plus
      unsigned int nbelem = 1;
      unsigned int timesteps = 1;
      double displacement_x = 0.1;
      double displacement_y = 0.1;
      double displacement_z = 0.1;
      double rayon_embedding = 1;
      double rayon_embedded = 0.2;
      int l2test = 0;

		};

		DistributedLagrangeProblem(const Parameters &parameters);

		// Entry point for the DistributedLagrangeProblem
		void
			run();

	private:
		// Object containing the actual parameters
		const Parameters &parameters;

		void
			setup_grids_and_dofs();

		void
			setup_embedding_dofs();

		void
			setup_embedded_dofs();

		void
			assemble_nitsche();


		void
			assemble_system();

		void
			solve();

		void
			output_results();

    void L2_error();

    void MoveEmbedded();
    void setup_mapping();


		// first we gather all the objects related to the embedding space geometry

		std::unique_ptr<Triangulation<spacedim>> space_grid;
		std::unique_ptr<GridTools::Cache<spacedim, spacedim>>
			space_grid_tools_cache;
		std::unique_ptr<FiniteElement<spacedim>> space_fe;
		std::unique_ptr<DoFHandler<spacedim>>    space_dh;

		// Then the ones related to the embedded grid, with the DoFHandler
		// associated to the Lagrange multiplier `lambda`

		//std::unique_ptr<Triangulation<dim, spacedim>> embedded_grid;
    Triangulation<dim, spacedim> embedded_grid;
		std::unique_ptr<FiniteElement<dim, spacedim>> embedded_fe;
		std::unique_ptr<DoFHandler<dim, spacedim>>    immersed_dh;

		// And finally, everything that is needed to *deform* the embedded
		// triangulation
		std::unique_ptr<FiniteElement<dim, spacedim>> embedded_configuration_fe;
		std::unique_ptr<DoFHandler<dim, spacedim>>    embedded_configuration_dh;
		Vector<double>                                embedded_configuration;


		ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
			embedded_configuration_function;

		std::unique_ptr<Mapping<dim, spacedim>> embedded_mapping;

		// We do the same thing to specify the value of the function $g$,
		// which is what we want our solution to be in the embedded space.
		// In this case the Function is a scalar one.
		ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
			embedded_value_function;

		// Similarly to what we have done with the Functions::ParsedFunction class,
		// we repeat the same for the ReductionControl class, allowing us to
		// specify all possible stopping criteria for the Schur complement
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

		Vector<double> embedded_value;

		// The TimerOutput class is used to provide some statistics on
		// the performance of our program.
		TimerOutput monitor;
	};

	template <int dim, int spacedim>
	DistributedLagrangeProblem<dim, spacedim>::Parameters::Parameters()
		: ParameterAcceptor("/Distributed Lagrange<" +
			Utilities::int_to_string(dim) + "," +
			Utilities::int_to_string(spacedim) + ">/")
	{
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

		add_parameter("Beta", beta);

    // Paramètres en plus
    add_parameter("Number of elements to embed into embedding", nbelem);
    add_parameter("Number of solves to the stationnary problem", timesteps);
    add_parameter("Size of displacement in x", displacement_x);
    add_parameter("Size of displacement in y", displacement_y);
    add_parameter("Size of displacement in z", displacement_z);
    add_parameter("Radius of embedding domain", rayon_embedding);
    add_parameter("Radius of embedded domain", rayon_embedded);
    add_parameter("Do you want to test L2 error (0/1)", l2test);


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
  void DistributedLagrangeProblem<dim, spacedim>::setup_mapping()
  {
    TimerOutput::Scope timer_section(monitor, "Setup grids and dofs");

    // Initializing $\Omega$: constructing the Triangulation and wrapping it
    // into a `std::unique_ptr` object
    space_grid = std_cxx14::make_unique<Triangulation<spacedim>>();
    const Point<spacedim> centerpoint_embedding(0.0, 0.0);
    GridGenerator::hyper_ball(*space_grid, centerpoint_embedding, parameters.rayon_embedding, false);

   
    // Once we constructed a Triangulation, we refine it globally according to
    // the specifications in the parameter file, and construct a
    // GridTools::Cache with it.
    space_grid->refine_global(parameters.initial_refinement);
    space_grid_tools_cache =
      std_cxx14::make_unique<GridTools::Cache<spacedim, spacedim>>(*space_grid);

    // The same is done with the embedded grid. Since the embedded grid is
    // deformed, we first need to setup the deformation mapping. We do so in the
    // following few lines:
    double radius = parameters.rayon_embedded;
    if (parameters.l2test == 0) {
      unsigned int nb_elem = parameters.nbelem;

      double xcoord;
      double ycoord;
      double zcoord;
      double rand_radius;

      double upper_bound = (parameters.rayon_embedding-radius);
      double lower_bound = -upper_bound;
      double rand1;
      double rand2;
      double rand3;
      std::srand(std::time(NULL));


      Triangulation<dim, spacedim> embedded_gridn;
      Triangulation<dim, spacedim> embedded_grid_flatten;
      Point<spacedim> centerpoint_elemn(0.0, 0.0);
      GridGenerator::hyper_sphere(embedded_grid_flatten, centerpoint_elemn, radius);

      embedded_grid_flatten.refine_global(parameters.initial_embedded_refinement);
      GridGenerator::flatten_triangulation(embedded_grid_flatten, embedded_gridn);



      for (unsigned int elem =0; elem < nb_elem; elem++) {

        // Point aléatoire dans cercle total
        while (true) {

		      rand1 = (double)rand() / RAND_MAX;
		      xcoord = lower_bound + rand1 * (upper_bound - lower_bound);

		      rand2 = (double)rand() / RAND_MAX;
		      ycoord = lower_bound + rand2 * (upper_bound - lower_bound);

          rand_radius = sqrt((xcoord*xcoord)+(ycoord*ycoord));

          if (spacedim == 3) {
            rand3 = (double)rand() / RAND_MAX;
            zcoord = lower_bound + rand3 * (upper_bound - lower_bound);

            rand_radius = sqrt((xcoord*xcoord)+(ycoord*ycoord) + (zcoord*zcoord));
          }

          if (rand_radius < (parameters.rayon_embedding-radius)) {
            break;
          }
        }

        // Pour voir les coordonnées obtenues
        //std::cout<< "   " << elem << "   " << xcoord << "   " << ycoord << std::endl;

        // Placer objet aux coordonnées obtenues
        GridTools::shift(Point<spacedim>(xcoord, ycoord), embedded_gridn);
        GridGenerator::merge_triangulations(embedded_grid, embedded_gridn, embedded_grid, 1.0e-12, true);

        // Reset position objet à imbriquer
        GridTools::shift(Point<spacedim>(-xcoord, -ycoord), embedded_gridn);


      }

      //embedded_grid.refine_global(parameters.initial_embedded_refinement);
    }
  
    else if (parameters.l2test == 1) {
      Point<spacedim> centerpoint_elem(0.0, 0.0);
      GridGenerator::hyper_sphere(embedded_grid, centerpoint_elem, radius);

      embedded_grid.refine_global(parameters.initial_embedded_refinement);
    }
  }

	template <int dim, int spacedim>
	void
		DistributedLagrangeProblem<dim, spacedim>::setup_grids_and_dofs()
	{
		embedded_configuration_fe = std_cxx14::make_unique<FESystem<dim, spacedim>>(
			FE_Q<dim, spacedim>(
				parameters.embedded_configuration_finite_element_degree),
			spacedim);

		embedded_configuration_dh =
			std_cxx14::make_unique<DoFHandler<dim, spacedim>>(embedded_grid);

		embedded_configuration_dh->distribute_dofs(*embedded_configuration_fe);
		embedded_configuration.reinit(embedded_configuration_dh->n_dofs());

		VectorTools::interpolate(*embedded_configuration_dh,
			embedded_configuration_function,
			embedded_configuration);

		if (parameters.use_displacement == true)
			embedded_mapping =
			std_cxx14::make_unique<MappingQEulerian<dim, Vector<double>, spacedim>>(
				parameters.embedded_configuration_finite_element_degree,
				*embedded_configuration_dh,
				embedded_configuration);
		else
			embedded_mapping =
			std_cxx14::make_unique<MappingFEField<dim,
			spacedim,
			Vector<double>,
			DoFHandler<dim, spacedim>>>(
				*embedded_configuration_dh, embedded_configuration);

		setup_embedded_dofs();

		std::vector<Point<spacedim>> support_points(immersed_dh->n_dofs());
		if (parameters.delta_refinement != 0)
			DoFTools::map_dofs_to_support_points(*embedded_mapping,
				*immersed_dh,
				support_points);

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

		// $\Omega$ has been refined and we can now set up its DoFs
		setup_embedding_dofs();
	}

	// We now set up the DoFs of $\Omega$ and $\Gamma$: since they are
	// fundamentally independent (except for the fact that $\Omega$'s mesh is more
	// refined "around"
	// $\Gamma$) the procedure is standard.
	template <int dim, int spacedim>
	void
		DistributedLagrangeProblem<dim, spacedim>::setup_embedding_dofs()
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
	void
		DistributedLagrangeProblem<dim, spacedim>::setup_embedded_dofs()
	{
		immersed_dh =
			std_cxx14::make_unique<DoFHandler<dim, spacedim>>(embedded_grid);
		embedded_fe = std_cxx14::make_unique<FE_Q<dim, spacedim>>(
			parameters.embedded_space_finite_element_degree);
		immersed_dh->distribute_dofs(*embedded_fe);

		// By definition the rhs of the system we're solving involves only a zero
		// vector and $G$, which is computed using only $\Gamma$'s DoFs
		embedded_value.reinit(immersed_dh->n_dofs());

		deallog << "Embedded dofs: " << immersed_dh->n_dofs() << std::endl;
	}

	template <int dim, int spacedim>
	void
		DistributedLagrangeProblem<dim, spacedim>::assemble_nitsche()
	{
		//    MappingQ1<dim, spacedim>      immersed_mapping= *embedded_mapping;
		MappingQ1<spacedim, spacedim> space_mapping;
		QGauss<dim>                   quad(2 * embedded_fe->degree + 1);

		const auto &space_fe = space_dh->get_fe();
		const auto &immersed_fe = immersed_dh->get_fe();

		FEValues<dim, spacedim> fe_v(*embedded_mapping,
			immersed_fe,
			quad,
			update_JxW_values | update_quadrature_points |
			update_normal_vectors | update_values);

		const unsigned int n_q_points = quad.size();
		const unsigned int n_active_c =
			immersed_dh->get_triangulation().n_active_cells();

		const auto used_cells_data = qpoints_over_locally_owned_cells(
			*space_grid_tools_cache, *immersed_dh, quad, *embedded_mapping, false);

		const auto &points_over_local_cells = std::get<0>(used_cells_data);
		const auto &used_cells_ids = std::get<1>(used_cells_data);

		//    // Get a list of outer cells, qpoints and maps.
		const auto cpm =
			GridTools::compute_point_locations(*space_grid_tools_cache,
				points_over_local_cells);
		const auto &all_cells = std::get<0>(cpm);
		const auto &all_qpoints = std::get<1>(cpm);
		const auto &all_maps = std::get<2>(cpm);

		std::vector<std::vector<
			typename Triangulation<spacedim, spacedim>::active_cell_iterator>>
			cell_container(n_active_c);
		std::vector<std::vector<std::vector<Point<spacedim>>>> qpoints_container(
			n_active_c);
		std::vector<std::vector<std::vector<unsigned int>>> maps_container(
			n_active_c);

		// Cycle over all cells of underling mesh found
		// call it omesh, elaborating the output
		for (unsigned int o = 0; o < all_cells.size(); ++o)
		{
			for (unsigned int j = 0; j < all_maps[o].size(); ++j)
			{
				// Find the index of the "owner" cell and qpoint
				// with regard to the immersed mesh
				// Find in which cell of immersed triangulation the point lies
				unsigned int cell_id;
				cell_id = all_maps[o][j] / n_q_points;

				const unsigned int n_pt = all_maps[o][j] % n_q_points;

				// If there are no cells, we just add our data
				if (cell_container[cell_id].empty())
				{
					cell_container[cell_id].emplace_back(all_cells[o]);
					qpoints_container[cell_id].emplace_back(
						std::vector<Point<spacedim>>{all_qpoints[o][j]});
					maps_container[cell_id].emplace_back(
						std::vector<unsigned int>{n_pt});
				}
				// If there are already cells, we begin by looking
				// at the last inserted cell, which is more likely:
				else if (cell_container[cell_id].back() == all_cells[o])
				{
					qpoints_container[cell_id].back().emplace_back(
						all_qpoints[o][j]);
					maps_container[cell_id].back().emplace_back(n_pt);
				}
				else
				{
					// We don't need to check the last element
					const auto cell_p = std::find(cell_container[cell_id].begin(),
						cell_container[cell_id].end() - 1,
						all_cells[o]);

					if (cell_p == cell_container[cell_id].end() - 1)
					{
						cell_container[cell_id].emplace_back(all_cells[o]);
						qpoints_container[cell_id].emplace_back(
							std::vector<Point<spacedim>>{all_qpoints[o][j]});
						maps_container[cell_id].emplace_back(
							std::vector<unsigned int>{n_pt});
					}
					else
					{
						const unsigned int pos =
							cell_p - cell_container[cell_id].begin();
						qpoints_container[cell_id][pos].emplace_back(
							all_qpoints[o][j]);
						maps_container[cell_id][pos].emplace_back(n_pt);
					}
				}
			}
		}

		std::vector<types::global_dof_index> dofs(immersed_fe.dofs_per_cell);
		std::vector<types::global_dof_index> odofs(space_fe.dofs_per_cell);


		FullMatrix<double> cell_matrix(space_fe.dofs_per_cell,
			space_fe.dofs_per_cell);

		Vector<double> local_rhs(space_fe.dofs_per_cell);


		typename DoFHandler<dim, spacedim>::active_cell_iterator
			cell = immersed_dh->begin_active(),
			endc = immersed_dh->end();

		for (unsigned int c_j = 0; cell != endc; ++cell, ++c_j)
		{
			// Reinitialize the cell and the fe_values
			fe_v.reinit(cell);
			cell->get_dof_indices(dofs);

			// Get a list of outer cells, qpoints and maps.
			const auto &cells = cell_container[c_j];
			const auto &qpoints = qpoints_container[c_j];
			const auto &maps = maps_container[c_j];

			auto immersed_quad_points = fe_v.get_quadrature_points();
			for (unsigned int c = 0; c < cells.size(); ++c)
			{
				cell_matrix = 0;
				local_rhs = 0;
				// Get the ones in the current outer cell
				typename DoFHandler<spacedim, spacedim>::active_cell_iterator ocell(
					*cells[c], &*space_dh);
				//            Make sure we act only on locally_owned cells
				if (ocell->is_locally_owned())
				{
					const std::vector<Point<spacedim>> &qps = qpoints[c];
					const std::vector<unsigned int> &   ids = maps[c];

					FEValues<spacedim, spacedim> o_fe_v(
						space_grid_tools_cache->get_mapping(),
						space_dh->get_fe(),
						qps,
						update_values | update_gradients);
					o_fe_v.reinit(ocell);
					ocell->get_dof_indices(odofs);

					const double h = std::pow(ocell->measure(), 1. / dim);

					for (unsigned int i = 0; i < space_fe.dofs_per_cell; ++i)
					{
						const auto comp_i =
							space_fe.system_to_component_index(i).first;
						for (unsigned int oq = 0; oq < o_fe_v.n_quadrature_points;
							++oq)
						{
							// Get the corresponding q point
							const unsigned int q = ids[oq];
							double dirichlet_value = embedded_value_function.value(
								immersed_quad_points[oq]);
							auto normal_vector = -fe_v.normal_vector(oq);
							//                        std::cout
							//                          << " Normal vecgtor : " <<
							//                          normal_vector
							//                          << " Quadrature point : " <<
							//                          immersed_quad_points[oq]
							//                          << std::endl;

							for (unsigned int j = 0; j < space_fe.dofs_per_cell;
								++j)

							{
								cell_matrix(i, j) +=
									parameters.beta / h *
									(o_fe_v.shape_value(j, oq) *
										o_fe_v.shape_value(i, oq) * fe_v.JxW(q));

								// Gradient term
								cell_matrix(i, j) += o_fe_v.shape_value(j, oq) *
									o_fe_v.shape_grad(i, oq) *
									normal_vector * fe_v.JxW(q);

								// Gradient term
								cell_matrix(i, j) -=
									o_fe_v.shape_grad(j, oq) * normal_vector *
									o_fe_v.shape_value(i, oq) * fe_v.JxW(q);
							}

							local_rhs(i) += parameters.beta / h * dirichlet_value *
								o_fe_v.shape_value(i, oq) * fe_v.JxW(q);

							local_rhs(i) += dirichlet_value *
								o_fe_v.shape_grad(i, oq) *
								normal_vector * fe_v.JxW(q);
						}
					}
				}
				constraints.distribute_local_to_global(
					cell_matrix, local_rhs, odofs, stiffness_matrix, rhs);
			} // namespace Step60
		}
	}
	// @sect3{Assembly}
	//
	// The following function creates the matrices: as noted before computing
	// the stiffness matrix and the rhs is a standard procedure.
	template <int dim, int spacedim>
	void
		DistributedLagrangeProblem<dim, spacedim>::assemble_system()
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

			//      VectorTools::create_right_hand_side(*embedded_mapping,
			//                                          *immersed_dh,
			//                                          QGauss<dim>(2 *
			//                                          embedded_fe->degree +
			//                                                      1),
			//                                          embedded_value_function,
			//                                          embedded_rhs);
		}
		{
			TimerOutput::Scope timer_section(monitor, "Assemble Nitsche BC");
			assemble_nitsche();
		}

		//      VectorTools::interpolate(*embedded_mapping,
		//                               *embedded_dh,
		//                               embedded_value_function,
		//                               embedded_value);
	} // namespace Step60

	// @sect3{Solve}
	//
	// All parts have been assembled: we solve the system
	// using the Schur complement method
	template <int dim, int spacedim>
	void
		DistributedLagrangeProblem<dim, spacedim>::solve()
	{
		TimerOutput::Scope timer_section(monitor, "Solve system");

		// Start by creating the inverse stiffness matrix
		SparseDirectUMFPACK K_inv_umfpack;
		K_inv_umfpack.initialize(stiffness_matrix);

		// Initializing the operators, as described in the introduction
		auto K = linear_operator(stiffness_matrix);
		auto K_inv = linear_operator(K, K_inv_umfpack);

		solution = K_inv * rhs;
  
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

  };    

  // Calcul numérique de la solution telle qu'elle est réellement (calcul purement mathématique)
  template <int spacedim>
  double AnalyticSolution<spacedim>::value(const Point<spacedim> &p, const unsigned int) const
  {
    double R1 = 0.2; // rayon interne
    double R2 = 1; // rayon externe
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

   // Déplacement dans le temps du maillage
  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::MoveEmbedded()
  {

    double Deplacement_x = parameters.displacement_x;
    double Deplacement_y = parameters.displacement_y;
    double Deplacement_z = parameters.displacement_z;

    if (spacedim == 2) {
      GridTools::shift(Point<spacedim>(Deplacement_x, Deplacement_y), embedded_grid);
    }
    else if (spacedim == 3) {
      GridTools::shift(Point<spacedim>(Deplacement_x, Deplacement_y, Deplacement_z), embedded_grid);
    }


  }

	// The following function simply generates standard result output on two
	// separate files, one for each mesh.
  int vtu_output = 1;
	template <int dim, int spacedim>
	void
		DistributedLagrangeProblem<dim, spacedim>::output_results()
	{
		TimerOutput::Scope timer_section(monitor, "Output results");

		DataOut<spacedim> embedding_out;

		auto embedding_name = std::string("embedding_nitsche_" + std::to_string(vtu_output) + ".vtu");
    
    std::ofstream embedding_out_file(embedding_name);

		embedding_out.attach_dof_handler(*space_dh);
		embedding_out.add_data_vector(solution, "solution");
		embedding_out.build_patches(
			parameters.embedding_space_finite_element_degree);
		embedding_out.write_vtu(embedding_out_file);

		DataOut<dim, DoFHandler<dim, spacedim>> embedded_out;

		auto embedded_name = std::string("embedded_nitsche_" + std::to_string(vtu_output) + ".vtu");
    
    std::ofstream embedded_out_file(embedded_name);

		embedded_out.attach_dof_handler(*immersed_dh);
		embedded_out.add_data_vector(embedded_value, "g");
		embedded_out.build_patches(*embedded_mapping,
			parameters.embedded_space_finite_element_degree);
		embedded_out.write_vtu(embedded_out_file);

    std::cout<< "L2 error = " << l2error << std::endl;
    vtu_output++;
	}

	// Similar to all other tutorial programs, the `run()` function simply calls
	// all other methods in the correct order. Nothing special to note, except
	// that we check if parsing was done before we actually attempt to run our
	// program.
	template <int dim, int spacedim>
	void
		DistributedLagrangeProblem<dim, spacedim>::run()
	{
		AssertThrow(parameters.initialized, ExcNotInitialized());
		deallog.depth_console(parameters.verbosity_level);

    unsigned int Deplacements =  parameters.timesteps; // mettre 1 pour aucun déplacements

    setup_mapping();
    
    for (unsigned int move = 0; move < Deplacements; move++) {
      std::cout << "Situation #" << (move+1) << std::endl;
    
      setup_grids_and_dofs();
      assemble_system();
      solve();
      L2_error();
      output_results();

      MoveEmbedded();
    }

		/*setup_grids_and_dofs();
		assemble_system();
		solve();
		output_results();*/
	}
} // namespace Step60



int
main(int argc, char **argv)
{
	try
	{
		using namespace dealii;
		using namespace Step60;

		const unsigned int dim = 1, spacedim = 2;

		// Differently to what happens in other tutorial programs, here we use
		// ParameterAcceptor style of initialization, i.e., all objects are
		// first constructed, and then a single call to the static method
		// ParameterAcceptor::initialize is issued to fill all parameters of the
		// classes that are derived from ParameterAcceptor.
		//
		// We check if the user has specified a parameter file name to use when
		// the program was launched. If so, try to read that parameter file,
		// otherwise, try to read the file "parameters.prm".
		//
		// If the parameter file that was specified (implicitly or explicitly)
		// does not exist, ParameterAcceptor::initialize will create one for
		// you, and exit the program.

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