#ifndef RT_PROJECTION_HH_
#define RT_PROJECTION_HH_


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/base/parameter_handler.h>

#include "AverageGradientOperators.hh"

#include "aux_primary.hh"
#include "FE_RTProj.hh"

#include <iostream>
#include <fstream>
#include <algorithm>

namespace RT_Projection
{
using namespace dealii;
//using namespace FE_RTProj;

template<int dim>
PETScWrappers::MPI::Vector compute_RT0_projection(Triangulation<dim, dim> &triangulation, const unsigned int degree, double theta_pl, double time,
		double time_step, double penalty_pl, double penalty_pl_bdry, std::vector<unsigned int> dirichlet_id_pl, bool use_exact_pl_in_RT0,
		bool use_exact_Sa_in_RT0, bool use_exact_Sv_in_RT0, bool second_order_extrapolation, bool incompressible,
		PETScWrappers::MPI::Vector pl_solution, PETScWrappers::MPI::Vector Sa_solution_n, PETScWrappers::MPI::Vector Sa_solution_nminus1,
		PETScWrappers::MPI::Vector Sv_solution_n, PETScWrappers::MPI::Vector Sv_solution_nminus1, PETScWrappers::MPI::Vector kappa_abs_vec,
		bool project_only_kappa, MPI_Comm mpi_communicator, const unsigned int n_mpi_processes, const unsigned int this_mpi_process)
{
	ConditionalOStream pcout(std::cout, (this_mpi_process == 0));

	const MappingQ1<dim> mapping;

	FE_DGQ<dim>     fe(degree);
	DoFHandler<dim> dof_handler(triangulation);

	dof_handler.distribute_dofs(fe);

	IndexSet locally_owned_dofs_DG;
	IndexSet locally_relevant_dofs_DG;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_DG =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
	locally_owned_dofs_DG = locally_owned_dofs_per_proc_DG[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs_DG);

	// Kappa stuff
	FE_DGQ<dim> fe_dg0(0);
	DoFHandler<dim> dof_handler_dg0(triangulation);
	IndexSet locally_owned_dofs_dg0;
	IndexSet locally_relevant_dofs_dg0;

	dof_handler_dg0.distribute_dofs(fe_dg0);
	const std::vector<IndexSet> locally_owned_dofs_per_proc_dg0 =
			DoFTools::locally_owned_dofs_per_subdomain(dof_handler_dg0);
	locally_owned_dofs_dg0 = locally_owned_dofs_per_proc_dg0[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_dg0, locally_relevant_dofs_dg0);

	// RT Projection vector
	PETScWrappers::MPI::Vector totalDarcyvelocity_RT0;

	// RT Projection space
	FE_RaviartThomas<dim> fe_RT0(0);
	DoFHandler<dim> dof_handler_RT0(triangulation);

	dof_handler_RT0.distribute_dofs(fe_RT0);

	IndexSet locally_owned_dofs_RT0;
	IndexSet locally_relevant_dofs_RT0;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_RT0 =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_RT0);
	locally_owned_dofs_RT0 = locally_owned_dofs_per_proc_RT0[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_RT0, locally_relevant_dofs_RT0);

	// DG space on faces for RT Projection
	FE_FaceP<dim> fe_test_scalar(degree-1);
	DoFHandler<dim> dof_handler_test_scalar(triangulation);

	dof_handler_test_scalar.distribute_dofs(fe_test_scalar);

	IndexSet locally_owned_dofs_test_scalar;
	IndexSet locally_relevant_dofs_test_scalar;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_test_scalar =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_test_scalar);
	locally_owned_dofs_test_scalar = locally_owned_dofs_per_proc_test_scalar[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_test_scalar, locally_relevant_dofs_test_scalar);

	totalDarcyvelocity_RT0.reinit(locally_owned_dofs_RT0, mpi_communicator);

	const QGauss<dim>     quadrature_formula(fe_RT0.degree + 1);
	const QGauss<dim - 1> face_quadrature_formula(fe_RT0.degree + 1);
	BoundaryValuesLiquidPressure<dim> boundary_function;
	boundary_function.set_time(time);

    // Densities
    rho_l<dim> rho_l_fcn;
    rho_v<dim> rho_v_fcn;
    rho_a<dim> rho_a_fcn;

    // Mobilities
    lambda_l<dim> lambda_l_fcn;
    lambda_v<dim> lambda_v_fcn;
    lambda_a<dim> lambda_a_fcn;

    ExactLiquidPressure<dim> pl_fcn;
    ExactAqueousSaturation<dim> Sa_fcn;
	ExactVaporSaturation<dim> Sv_fcn;

	FEValues<dim> fe_values_DG(fe,
							   quadrature_formula,
							   update_values | update_gradients |
							   update_quadrature_points |
							   update_JxW_values);

	FEFaceValues<dim> fe_face_values_DG(fe,
										face_quadrature_formula,
										update_values | update_gradients |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_DG_neighbor(fe,
										face_quadrature_formula,
										update_values | update_gradients |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEValues<dim> fe_values_RT0(fe_RT0,
	                               quadrature_formula,
	                               update_values | update_gradients |
	                               update_quadrature_points |
	                               update_JxW_values);

	FEFaceValues<dim> fe_face_values_RT0(fe_RT0,
										face_quadrature_formula,
										update_values |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_test_scalar(fe_test_scalar,
										  face_quadrature_formula,
										  update_values |
										  update_normal_vectors |
										  update_quadrature_points |
										  update_JxW_values);

	const unsigned int dofs_per_cell_DG = fe.dofs_per_cell;
	const unsigned int dofs_per_cell_RT0 = fe_RT0.dofs_per_cell;
	const unsigned int dofs_per_face_RT0 = fe_RT0.n_dofs_per_face();
	const unsigned int dofs_per_face_test_scalar = fe_test_scalar.n_dofs_per_face();

	std::vector<std::vector<unsigned int>> fe_support_on_face_scalar(GeometryInfo<dim>::faces_per_cell);

	for (unsigned int face_no : GeometryInfo<dim>::face_indices())
	{
		for (unsigned int i = 0; i < fe_test_scalar.dofs_per_cell; ++i)
		{
			if (fe_test_scalar.has_support_on_face(i, face_no))
				fe_support_on_face_scalar[face_no].push_back(i);
		}
	}

	const FEValuesExtractors::Vector velocities(0);

	std::vector<double> basis_fcns_faces_scalar(fe_test_scalar.dofs_per_cell);

	FullMatrix<double> cell_matrix_RT0(GeometryInfo<dim>::faces_per_cell*dofs_per_face_test_scalar, dofs_per_cell_RT0);
	Vector<double> cell_solution_RT0(dofs_per_cell_RT0);
	Vector<double> cell_RHS_RT0(GeometryInfo<dim>::faces_per_cell*dofs_per_face_test_scalar);

	FullMatrix<double> face_matrix_RT0(dofs_per_face_RT0, dofs_per_face_test_scalar);
	Vector<double> face_RHS_RT0(dofs_per_face_test_scalar);

	std::vector<types::global_dof_index> local_dof_indices_DG(dofs_per_cell_DG);
	std::vector<types::global_dof_index> local_dof_indices_RT0(dofs_per_cell_RT0);

	typename DoFHandler<dim>::active_cell_iterator
		  cell_DG = dof_handler.begin_active(),
		  endc = dof_handler.end(), cell_RT0 = dof_handler_RT0.begin_active(),
		  cell_scalar = dof_handler_test_scalar.begin_active();

    const unsigned int n_q_points_DG = fe_values_DG.get_quadrature().size();
    const unsigned int n_q_points_RT0 = fe_values_RT0.get_quadrature().size();
	const unsigned int n_q_points_face_DG = fe_face_values_DG.get_quadrature().size();
	const unsigned int n_q_points_face_test_scalar = fe_face_values_test_scalar.get_quadrature().size();

	PETScWrappers::MPI::Vector temp_pl_solution;

	PETScWrappers::MPI::Vector temp_Sa_solution_n;
	PETScWrappers::MPI::Vector temp_Sa_solution_nminus1;

	PETScWrappers::MPI::Vector temp_Sv_solution_n;
	PETScWrappers::MPI::Vector temp_Sv_solution_nminus1;

	PETScWrappers::MPI::Vector temp_kappa;

	temp_pl_solution.reinit(locally_owned_dofs_DG,
							locally_relevant_dofs_DG,
							mpi_communicator);

	temp_Sa_solution_n.reinit(locally_owned_dofs_DG,
							  locally_relevant_dofs_DG,
							  mpi_communicator);

	temp_Sa_solution_nminus1.reinit(locally_owned_dofs_DG,
									locally_relevant_dofs_DG,
									mpi_communicator);

	temp_Sv_solution_n.reinit(locally_owned_dofs_DG,
							  locally_relevant_dofs_DG,
							  mpi_communicator);

	temp_Sv_solution_nminus1.reinit(locally_owned_dofs_DG,
									locally_relevant_dofs_DG,
									mpi_communicator);

	temp_kappa.reinit(locally_owned_dofs_dg0,
					  locally_relevant_dofs_dg0,
					  mpi_communicator);

	temp_pl_solution = pl_solution;

	temp_Sa_solution_n = Sa_solution_n;
	temp_Sa_solution_nminus1 = Sa_solution_nminus1;

	temp_Sv_solution_n = Sv_solution_n;
	temp_Sv_solution_nminus1 = Sv_solution_nminus1;

	temp_kappa = kappa_abs_vec;

	// Loop over cells
	for (; cell_DG != endc; ++cell_DG, ++cell_RT0, ++cell_scalar)
	{
		if (cell_DG->subdomain_id() == this_mpi_process)
		{
			fe_values_DG.reinit(cell_DG);
			fe_values_RT0.reinit(cell_RT0);

			cell_solution_RT0 = 0.0;

			// Loop over faces
			for (const auto &face : cell_RT0->face_iterators())
			{
				face_matrix_RT0 = 0.0;
				face_RHS_RT0 = 0.0;

				unsigned int face_num = cell_scalar->face_iterator_to_index(face);

				fe_face_values_DG.reinit(cell_DG, face);
				fe_face_values_RT0.reinit(cell_RT0, face);
				fe_face_values_test_scalar.reinit(cell_scalar, face);

				const auto &q_points = fe_face_values_test_scalar.get_quadrature_points();

				std::vector<double> g(n_q_points_face_test_scalar);
				boundary_function.value_list(fe_face_values_test_scalar.get_quadrature_points(), g);

				for (unsigned int q = 0; q < n_q_points_face_test_scalar; ++q)
				{
					const Tensor<1, dim> normal = fe_face_values_test_scalar.normal_vector(q);

					for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
						basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

					for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
					{
						for (unsigned int j = 0; j < dofs_per_cell_RT0; ++j)
						{
							const unsigned int ii = fe_support_on_face_scalar[face_num][i];

							face_matrix_RT0(i, 0) +=
									fe_face_values_RT0[velocities].value(j, q)
									* normal
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);
						}
					}
				}

				if(face->at_boundary())
				{
					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					std::vector<double> Sa_vals_face_n(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_face_n);

					std::vector<double> Sa_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1);

					std::vector<double> Sv_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_face);

					std::vector<double> Sv_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1);

					std::vector<Tensor<1, dim>> grad_pl_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_gradients(temp_pl_solution, grad_pl_face);

					double kappa = temp_kappa[cell_DG->global_active_cell_index()];

					for (unsigned int q = 0; q < n_q_points_face_test_scalar; ++q)
					{
						const Tensor<1, dim> normal = fe_face_values_test_scalar.normal_vector(q);
						Tensor<1,dim> neg_pl_grad = -grad_pl_face[q];

						double pl_value = pl_vals_face[q];
						double Sa_value_n = Sa_vals_face_n[q];
						double Sa_value_nminus1 = Sa_vals_face_nminus1[q];
						double Sv_value_n = Sv_vals_face[q];
						double Sv_value_nminus1 = Sv_vals_face_nminus1[q];

						if(use_exact_pl_in_RT0)
						{
							pl_fcn.set_time(time);
							pl_value = pl_fcn.value(q_points[q]);

							neg_pl_grad = pl_fcn.gradient(q_points[q]);
							neg_pl_grad *= -1.0;
						}

						if(use_exact_Sa_in_RT0)
						{
							Sa_fcn.set_time(time - time_step);
							Sa_value_n = Sa_fcn.value(q_points[q]);

							Sa_fcn.set_time(time - 2.0*time_step);
							Sa_value_nminus1 = Sa_fcn.value(q_points[q]);
						}

						if(use_exact_Sv_in_RT0)
						{
							Sv_fcn.set_time(time - time_step);
							Sv_value_n = Sv_fcn.value(q_points[q]);

							Sv_fcn.set_time(time - 2.0*time_step);
							Sv_value_nminus1 = Sv_fcn.value(q_points[q]);
						}

						double Sa_nplus1_extrapolation = Sa_value_n;
						double Sv_nplus1_extrapolation = Sv_value_n;

						if(second_order_extrapolation)
						{
							Sa_nplus1_extrapolation *= 2.0;
							Sa_nplus1_extrapolation -= Sa_value_nminus1;

							Sv_nplus1_extrapolation *= 2.0;
							Sv_nplus1_extrapolation -= Sv_value_nminus1;

						}

						double rho_l = rho_l_fcn.value(pl_value);
						double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double rho_a = rho_a_fcn.value(pl_value);

						double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

						double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

						if(incompressible)
							rholambda_t = lambda_l + lambda_v + lambda_a;

						if(project_only_kappa)
							rholambda_t = 1.0;

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						double gamma_ch_e = rholambda_t*kappa;
			            double h_e = cell_DG->face(face_num)->measure();
			            double penalty_factor = (penalty_pl_bdry/h_e) * gamma_ch_e * degree*(degree + dim - 1);

			            // Figure out if this face is Dirichlet or Neumann
			            bool dirichlet = false;

			            for(unsigned int i = 0; i < dirichlet_id_pl.size(); i++)
			            {
			            	if(face->boundary_id() == dirichlet_id_pl[i])
			            	{
			            		dirichlet = true;
			            		break;
			            	}
			            }

						for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
						{
							const unsigned int ii = fe_support_on_face_scalar[face_num][i];

							if(dirichlet)
							{
								face_RHS_RT0[i] +=
										rholambda_t
										* kappa
										* neg_pl_grad
										* normal
										* basis_fcns_faces_scalar[i]
										* fe_face_values_test_scalar.JxW(q);

								face_RHS_RT0[i] +=
										penalty_factor
										* (pl_value - g[q])
										* basis_fcns_faces_scalar[i]
										* fe_face_values_test_scalar.JxW(q);
								//std::cout << "pl_value = " << pl_value << " g[q] = " << g[q] << std::endl;
							}

						}
					}
				}
				else
				{
					typename DoFHandler<dim>::active_cell_iterator cell_DG_neighbor = cell_DG->neighbor(face_num);
					unsigned int neighbor_index = cell_DG->neighbor_index(face_num);

					fe_face_values_DG_neighbor.reinit(cell_DG_neighbor, face);

					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					std::vector<double> pl_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_pl_solution, pl_vals_face_neighbor);

					std::vector<double> Sa_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_face);

					std::vector<double> Sa_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sa_solution_n, Sa_vals_face_neighbor);

					std::vector<double> Sa_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1);

					std::vector<double> Sa_vals_face_nminus1_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1_neighbor);

					std::vector<double> Sv_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_face);

					std::vector<double> Sv_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sv_solution_n, Sv_vals_face_neighbor);

					std::vector<double> Sv_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1);

					std::vector<double> Sv_vals_face_nminus1_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1_neighbor);

					std::vector<Tensor<1, dim>> grad_pl_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_gradients(temp_pl_solution, grad_pl_face);

					std::vector<Tensor<1, dim>> grad_pl_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_gradients(temp_pl_solution, grad_pl_face_neighbor);

					double kappa0 = temp_kappa[cell_DG->global_active_cell_index()];
					double kappa1 = temp_kappa[cell_DG_neighbor->global_active_cell_index()];

					for (unsigned int q = 0; q < n_q_points_face_test_scalar; ++q)
					{
						const Tensor<1, dim> normal = fe_face_values_test_scalar.normal_vector(q);

						double pl_value0 = pl_vals_face[q];
						double pl_value1 = pl_vals_face_neighbor[q];

						double Sa_value0_n = Sa_vals_face[q];
						double Sa_value1_n = Sa_vals_face_neighbor[q];
						double Sa_value0_nminus1 = Sa_vals_face_nminus1[q];
						double Sa_value1_nminus1 = Sa_vals_face_nminus1_neighbor[q];

						double Sv_value0_n = Sv_vals_face[q];
						double Sv_value1_n = Sv_vals_face_neighbor[q];
						double Sv_value0_nminus1 = Sv_vals_face_nminus1[q];
						double Sv_value1_nminus1 = Sv_vals_face_nminus1_neighbor[q];

						Tensor<1,dim> neg_pl_grad = -grad_pl_face[q];
						Tensor<1,dim> neg_pl_grad_neighbor = -grad_pl_face_neighbor[q];

						if(use_exact_pl_in_RT0)
						{
							pl_fcn.set_time(time);
							pl_value0 = pl_fcn.value(q_points[q]);
							pl_value1 = pl_value0;

							neg_pl_grad = pl_fcn.gradient(q_points[q]);
							neg_pl_grad *= -1.0;

							neg_pl_grad_neighbor = neg_pl_grad;
						}

						if(use_exact_Sa_in_RT0)
						{
							Sa_fcn.set_time(time - time_step);
							Sa_value0_n = Sa_fcn.value(q_points[q]);
							Sa_value1_n = Sa_value0_n;

							Sa_fcn.set_time(time - 2.0*time_step);
							Sa_value0_nminus1 = Sa_fcn.value(q_points[q]);

							Sa_value1_nminus1 = Sa_value0_nminus1;
						}

						if(use_exact_Sv_in_RT0)
						{
							Sv_fcn.set_time(time - time_step);
							Sv_value0_n = Sv_fcn.value(q_points[q]);
							Sv_value1_n = Sv_value0_n;

							Sv_fcn.set_time(time - 2.0*time_step);
							Sv_value0_nminus1 = Sv_fcn.value(q_points[q]);
							Sv_value1_nminus1 = Sv_value0_nminus1;
						}

						double Sa_nplus1_extrapolation0 = Sa_value0_n;
						double Sa_nplus1_extrapolation1 = Sa_value1_n;
						double Sv_nplus1_extrapolation0 = Sv_value0_n;
						double Sv_nplus1_extrapolation1 = Sv_value1_n;

						if(second_order_extrapolation)
						{
							Sa_nplus1_extrapolation0 *= 2.0;
							Sa_nplus1_extrapolation0 -= Sa_value0_nminus1;

							Sa_nplus1_extrapolation1 *= 2.0;
							Sa_nplus1_extrapolation1 -= Sa_value1_nminus1;

							Sv_nplus1_extrapolation0 *= 2.0;
							Sv_nplus1_extrapolation0 -= Sv_value0_nminus1;

							Sv_nplus1_extrapolation1 *= 2.0;
							Sv_nplus1_extrapolation1 -= Sv_value1_nminus1;
						}

						double rho_l0 = rho_l_fcn.value(pl_value0);
						double rho_l1 = rho_l_fcn.value(pl_value1);

						double rho_v0 = rho_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double rho_v1 = rho_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

						double rho_a0 = rho_a_fcn.value(pl_value0);
						double rho_a1 = rho_a_fcn.value(pl_value1);

						double lambda_l0 = lambda_l_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double lambda_v0 = lambda_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double lambda_a0 = lambda_a_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);

						double lambda_l1 = lambda_l_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
						double lambda_v1 = lambda_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
						double lambda_a1 = lambda_a_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

						double rholambda_t0 = rho_l0*lambda_l0 + rho_v0*lambda_v0 + rho_a0*lambda_a0;
						double rholambda_t1 = rho_l1*lambda_l1 + rho_v1*lambda_v1 + rho_a1*lambda_a1;

						if(incompressible)
						{
							rholambda_t0 = lambda_l0 + lambda_v0 + lambda_a0;
							rholambda_t1 = lambda_l1 + lambda_v1 + lambda_a1;
						}

						if(project_only_kappa)
						{
							rholambda_t0 = 1.0;
							rholambda_t1 = 1.0;
						}

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						double coef0 = rholambda_t0*kappa0;
						double coef1 = rholambda_t1*kappa1;

						double weight0 = 0.0;
						double weight1 = 0.0;

//						if(fabs(coef0) > 1.e-14 || fabs(coef1) > 1.e-14)
//						{
							weight0 = coef1/(coef0 + coef1 + 1.e-20);
							weight1 = coef0/(coef0 + coef1 + 1.e-20);
//						}

						double weighted_aver_rhs = AverageGradOperators::weighted_average_rhs<dim>(normal,
									neg_pl_grad, neg_pl_grad_neighbor,
									coef0, coef1,
									weight0, weight1);

						for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
						{
							const unsigned int ii = fe_support_on_face_scalar[face_num][i];

							face_RHS_RT0[i] +=
									weighted_aver_rhs
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);

							double gamma_ch_e = 2.0*coef0*coef1/(coef0 + coef1);
				            double h_e = cell_DG->face(face_num)->measure();
				            double penalty_factor = (penalty_pl/h_e) * gamma_ch_e * degree*(degree + dim - 1);

							face_RHS_RT0[i] +=
									penalty_factor
									* (pl_value0 - pl_value1)
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);
						}
					}
				}

				FullMatrix<double> inv_face_matrix_RT0(dofs_per_face_test_scalar, dofs_per_face_RT0);
				inv_face_matrix_RT0.invert(face_matrix_RT0);
				Vector<double> sol_local_face_RT0(face_RHS_RT0.size());

				inv_face_matrix_RT0.vmult(sol_local_face_RT0, face_RHS_RT0);

				std::vector<unsigned int> dof_ind(1);
				face->get_dof_indices(dof_ind);

				totalDarcyvelocity_RT0[dof_ind[0]] = sol_local_face_RT0[0];

			}
		}

		totalDarcyvelocity_RT0.compress(VectorOperation::insert);

	}

	return totalDarcyvelocity_RT0;
}


template<int dim>
PETScWrappers::MPI::Vector compute_RT0_projection_with_gravity(Triangulation<dim, dim> &triangulation,
		const unsigned int degree, double theta_pl, double time, double time_step, double penalty_pl, double penalty_pl_bdry,
		std::vector<unsigned int> dirichlet_id_pl, bool use_exact_pl_in_RT0,
		bool use_exact_Sa_in_RT0, bool use_exact_Sv_in_RT0, bool second_order_extrapolation, bool incompressible,
		PETScWrappers::MPI::Vector pl_solution, PETScWrappers::MPI::Vector Sa_solution_n, PETScWrappers::MPI::Vector Sa_solution_nminus1,
		PETScWrappers::MPI::Vector Sv_solution_n, PETScWrappers::MPI::Vector Sv_solution_nminus1, PETScWrappers::MPI::Vector kappa_abs_vec,
		bool use_Sa, bool project_only_kappa, MPI_Comm mpi_communicator, const unsigned int n_mpi_processes, const unsigned int this_mpi_process)
{
	const MappingQ1<dim> mapping;

	// Furthermore we want to use DG elements.
	FE_DGQ<dim>     fe(degree);
	DoFHandler<dim> dof_handler(triangulation);

	dof_handler.distribute_dofs(fe);

	IndexSet locally_owned_dofs_DG;
	IndexSet locally_relevant_dofs_DG;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_DG =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
	locally_owned_dofs_DG = locally_owned_dofs_per_proc_DG[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs_DG);

	// Kappa stuff
	FE_DGQ<dim> fe_dg0(0);
	DoFHandler<dim> dof_handler_dg0(triangulation);
	IndexSet locally_owned_dofs_dg0;
	IndexSet locally_relevant_dofs_dg0;

	dof_handler_dg0.distribute_dofs(fe_dg0);
	const std::vector<IndexSet> locally_owned_dofs_per_proc_dg0 =
			DoFTools::locally_owned_dofs_per_subdomain(dof_handler_dg0);
	locally_owned_dofs_dg0 = locally_owned_dofs_per_proc_dg0[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_dg0, locally_relevant_dofs_dg0);

	// RT Projection vector
	PETScWrappers::MPI::Vector totalDarcyvelocity_RT0;

	// RT Projection space
	FE_RaviartThomas<dim> fe_RT0(0);
	DoFHandler<dim> dof_handler_RT0(triangulation);

	dof_handler_RT0.distribute_dofs(fe_RT0);

	IndexSet locally_owned_dofs_RT0;
	IndexSet locally_relevant_dofs_RT0;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_RT0 =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_RT0);
	locally_owned_dofs_RT0 = locally_owned_dofs_per_proc_RT0[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_RT0, locally_relevant_dofs_RT0);

	// DG space on faces for RT Projection
	FE_FaceP<dim> fe_test_scalar(degree-1);
	DoFHandler<dim> dof_handler_test_scalar(triangulation);

	dof_handler_test_scalar.distribute_dofs(fe_test_scalar);

	IndexSet locally_owned_dofs_test_scalar;
	IndexSet locally_relevant_dofs_test_scalar;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_test_scalar =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_test_scalar);
	locally_owned_dofs_test_scalar = locally_owned_dofs_per_proc_test_scalar[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_test_scalar, locally_relevant_dofs_test_scalar);

	totalDarcyvelocity_RT0.reinit(locally_owned_dofs_RT0, mpi_communicator);

	const QGauss<dim>     quadrature_formula(fe_RT0.degree + 1);
	const QGauss<dim - 1> face_quadrature_formula(fe_RT0.degree + 1);
	BoundaryValuesLiquidPressure<dim> boundary_function;
	boundary_function.set_time(time);

    // Densities
    rho_l<dim> rho_l_fcn;
    rho_v<dim> rho_v_fcn;
    rho_a<dim> rho_a_fcn;

    // Mobilities
    lambda_l<dim> lambda_l_fcn;
    lambda_v<dim> lambda_v_fcn;
    lambda_a<dim> lambda_a_fcn;

    // Gravity
    GravitySourceTerm<dim> gravity_fcn;

    gravity_fcn.set_time(time);

    ExactLiquidPressure<dim> pl_fcn;
    ExactAqueousSaturation<dim> Sa_fcn;
	ExactVaporSaturation<dim> Sv_fcn;

	FEValues<dim> fe_values_DG(fe,
							   quadrature_formula,
							   update_values | update_gradients |
							   update_quadrature_points |
							   update_JxW_values);

	FEFaceValues<dim> fe_face_values_DG(fe,
										face_quadrature_formula,
										update_values | update_gradients |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_DG_neighbor(fe,
										face_quadrature_formula,
										update_values | update_gradients |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEValues<dim> fe_values_RT0(fe_RT0,
	                               quadrature_formula,
	                               update_values | update_gradients |
	                               update_quadrature_points |
	                               update_JxW_values);

	FEFaceValues<dim> fe_face_values_RT0(fe_RT0,
										face_quadrature_formula,
										update_values |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_test_scalar(fe_test_scalar,
										  face_quadrature_formula,
										  update_values |
										  update_normal_vectors |
										  update_quadrature_points |
										  update_JxW_values);

	const unsigned int dofs_per_cell_DG = fe.dofs_per_cell;
	const unsigned int dofs_per_cell_RT0 = fe_RT0.dofs_per_cell;
	const unsigned int dofs_per_face_RT0 = fe_RT0.n_dofs_per_face();
	const unsigned int dofs_per_face_test_scalar = fe_test_scalar.n_dofs_per_face();

	std::vector<std::vector<unsigned int>> fe_support_on_face_scalar(GeometryInfo<dim>::faces_per_cell);

	for (unsigned int face_no : GeometryInfo<dim>::face_indices())
	{
		for (unsigned int i = 0; i < fe_test_scalar.dofs_per_cell; ++i)
		{
			if (fe_test_scalar.has_support_on_face(i, face_no))
				fe_support_on_face_scalar[face_no].push_back(i);
		}
	}

	const FEValuesExtractors::Vector velocities(0);

	std::vector<double> basis_fcns_faces_scalar(fe_test_scalar.dofs_per_cell);

	FullMatrix<double> cell_matrix_RT0(GeometryInfo<dim>::faces_per_cell*dofs_per_face_test_scalar, dofs_per_cell_RT0);
	Vector<double> cell_solution_RT0(dofs_per_cell_RT0);
	Vector<double> cell_RHS_RT0(GeometryInfo<dim>::faces_per_cell*dofs_per_face_test_scalar);

	FullMatrix<double> face_matrix_RT0(dofs_per_face_RT0, dofs_per_face_test_scalar);
	Vector<double> face_RHS_RT0(dofs_per_face_test_scalar);

	std::vector<types::global_dof_index> local_dof_indices_DG(dofs_per_cell_DG);
	std::vector<types::global_dof_index> local_dof_indices_RT0(dofs_per_cell_RT0);

	typename DoFHandler<dim>::active_cell_iterator
		  cell_DG = dof_handler.begin_active(),
		  endc = dof_handler.end(), cell_RT0 = dof_handler_RT0.begin_active(),
		  cell_scalar = dof_handler_test_scalar.begin_active();

    const unsigned int n_q_points_DG = fe_values_DG.get_quadrature().size();
    const unsigned int n_q_points_RT0 = fe_values_RT0.get_quadrature().size();
	const unsigned int n_q_points_face_DG = fe_face_values_DG.get_quadrature().size();
	const unsigned int n_q_points_face_test_scalar = fe_face_values_test_scalar.get_quadrature().size();

	PETScWrappers::MPI::Vector temp_pl_solution;

	PETScWrappers::MPI::Vector temp_Sa_solution_n;
	PETScWrappers::MPI::Vector temp_Sa_solution_nminus1;

	PETScWrappers::MPI::Vector temp_Sv_solution_n;
	PETScWrappers::MPI::Vector temp_Sv_solution_nminus1;

	PETScWrappers::MPI::Vector temp_kappa;

	temp_pl_solution.reinit(locally_owned_dofs_DG,
							locally_relevant_dofs_DG,
							mpi_communicator);

	temp_Sa_solution_n.reinit(locally_owned_dofs_DG,
							  locally_relevant_dofs_DG,
							  mpi_communicator);

	temp_Sa_solution_nminus1.reinit(locally_owned_dofs_DG,
									locally_relevant_dofs_DG,
									mpi_communicator);

	temp_Sv_solution_n.reinit(locally_owned_dofs_DG,
							  locally_relevant_dofs_DG,
							  mpi_communicator);

	temp_Sv_solution_nminus1.reinit(locally_owned_dofs_DG,
									locally_relevant_dofs_DG,
									mpi_communicator);

	temp_kappa.reinit(locally_owned_dofs_dg0,
					  locally_relevant_dofs_dg0,
					  mpi_communicator);

	temp_pl_solution = pl_solution;

	temp_Sa_solution_n = Sa_solution_n;
	temp_Sa_solution_nminus1 = Sa_solution_nminus1;

	temp_Sv_solution_n = Sv_solution_n;
	temp_Sv_solution_nminus1 = Sv_solution_nminus1;

	temp_kappa = kappa_abs_vec;

//	std::vecor<int> face_done(GeometryInfo<dim>::)
	for (; cell_DG != endc; ++cell_DG, ++cell_RT0, ++cell_scalar)
	{
		if (cell_DG->subdomain_id() == this_mpi_process)
		{
			fe_values_DG.reinit(cell_DG);
			fe_values_RT0.reinit(cell_RT0);

			cell_solution_RT0 = 0.0;

			for (const auto &face : cell_RT0->face_iterators())
			{
	//			const auto face = cell_scalar->face(2);
				face_matrix_RT0 = 0.0;
				face_RHS_RT0 = 0.0;

				unsigned int face_num = cell_scalar->face_iterator_to_index(face);

				fe_face_values_DG.reinit(cell_DG, face);
				fe_face_values_RT0.reinit(cell_RT0, face);
				fe_face_values_test_scalar.reinit(cell_scalar, face);

				const auto &q_points = fe_face_values_test_scalar.get_quadrature_points();

				std::vector<double> g(n_q_points_face_test_scalar);
				boundary_function.value_list(fe_face_values_test_scalar.get_quadrature_points(), g);

				for (unsigned int q = 0; q < n_q_points_face_test_scalar; ++q)
				{
					const Tensor<1, dim> normal = fe_face_values_test_scalar.normal_vector(q);

					for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
						basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

					for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
					{
						for (unsigned int j = 0; j < dofs_per_cell_RT0; ++j)
						{
							const unsigned int ii = fe_support_on_face_scalar[face_num][i];

	//						std::cout << "fe_face_values_RT[velocities].value(j, q) = " << fe_face_values_RT0[velocities].value(j, q) << std::endl;
							face_matrix_RT0(i, 0) +=
									fe_face_values_RT0[velocities].value(j, q)
									* normal
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);
						}
					}
				}

				if(face->at_boundary())
				{
					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					std::vector<double> Sa_vals_face_n(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_face_n);

					std::vector<double> Sa_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1);

					std::vector<double> Sv_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_face);

					std::vector<double> Sv_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1);

					std::vector<Tensor<1, dim>> grad_pl_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_gradients(temp_pl_solution, grad_pl_face);

					double kappa = temp_kappa[cell_DG->global_active_cell_index()];

					for (unsigned int q = 0; q < n_q_points_face_test_scalar; ++q)
					{
						const Tensor<1, dim> normal = fe_face_values_test_scalar.normal_vector(q);
						Tensor<1,dim> neg_pl_grad = -grad_pl_face[q];

						double pl_value = pl_vals_face[q];
						double Sa_value_n = Sa_vals_face_n[q];
						double Sa_value_nminus1 = Sa_vals_face_nminus1[q];
						double Sv_value_n = Sv_vals_face[q];
						double Sv_value_nminus1 = Sv_vals_face_nminus1[q];

						if(use_exact_pl_in_RT0)
						{
							pl_fcn.set_time(time);
							pl_value = pl_fcn.value(q_points[q]);

							neg_pl_grad = pl_fcn.gradient(q_points[q]);
							neg_pl_grad *= -1.0;
						}

						if(use_exact_Sa_in_RT0)
						{
							Sa_fcn.set_time(time - time_step);
							Sa_value_n = Sa_fcn.value(q_points[q]);

							Sa_fcn.set_time(time - 2.0*time_step);
							Sa_value_nminus1 = Sa_fcn.value(q_points[q]);
						}

						if(use_exact_Sv_in_RT0)
						{
							Sv_fcn.set_time(time - time_step);
							Sv_value_n = Sv_fcn.value(q_points[q]);

							Sv_fcn.set_time(time - 2.0*time_step);
							Sv_value_nminus1 = Sv_fcn.value(q_points[q]);
						}

						double Sa_nplus1_extrapolation = Sa_value_n;
						double Sv_nplus1_extrapolation = Sv_value_n;

						if(second_order_extrapolation)
						{
							Sa_nplus1_extrapolation *= 2.0;
							Sa_nplus1_extrapolation -= Sa_value_nminus1;

							Sv_nplus1_extrapolation *= 2.0;
							Sv_nplus1_extrapolation -= Sv_value_nminus1;

						}

						double rho_l = rho_l_fcn.value(pl_value);
						double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double rho_a = rho_a_fcn.value(pl_value);

						double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

						double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

						if(incompressible)
							rholambda_t = lambda_l + lambda_v + lambda_a;

						if(project_only_kappa)
							rholambda_t = 1.0;

						Tensor<1,dim> g_val = gravity_fcn.vector_value(q_points[q]);
						double density_g;

						if(use_Sa)
							density_g = rho_a;
						else
							density_g = rho_v;

						g_val *= density_g;

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						double gamma_ch_e = rholambda_t*kappa;
						double h_e = cell_DG->face(face_num)->measure();
						double penalty_factor = (penalty_pl_bdry/h_e) * gamma_ch_e * degree*(degree + dim - 1);

			            // Figure out if this face is Dirichlet or Neumann
			            bool dirichlet = false;

			            for(unsigned int i = 0; i < dirichlet_id_pl.size(); i++)
			            {
			            	if(face->boundary_id() == dirichlet_id_pl[i])
			            	{
			            		dirichlet = true;
			            		break;
			            	}
			            }

						for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
						{
							const unsigned int ii = fe_support_on_face_scalar[face_num][i];

							if(dirichlet)
							{
								face_RHS_RT0[i] +=
										rholambda_t
										* kappa
										* neg_pl_grad
										* normal
										* basis_fcns_faces_scalar[i]
										* fe_face_values_test_scalar.JxW(q);

								face_RHS_RT0[i] +=
										rholambda_t
										* kappa
										* g_val
										* normal
										* basis_fcns_faces_scalar[i]
										* fe_face_values_test_scalar.JxW(q);

								face_RHS_RT0[i] +=
										penalty_factor
										* (pl_value - g[q])
										* basis_fcns_faces_scalar[i]
										* fe_face_values_test_scalar.JxW(q);
							}

						}
					}
				}
				else
				{
					typename DoFHandler<dim>::active_cell_iterator cell_DG_neighbor = cell_DG->neighbor(face_num);
					unsigned int neighbor_index = cell_DG->neighbor_index(face_num);

					fe_face_values_DG_neighbor.reinit(cell_DG_neighbor, face);

					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					std::vector<double> pl_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_pl_solution, pl_vals_face_neighbor);

					std::vector<double> Sa_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_face);

					std::vector<double> Sa_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sa_solution_n, Sa_vals_face_neighbor);

					std::vector<double> Sa_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1);

					std::vector<double> Sa_vals_face_nminus1_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1_neighbor);

					std::vector<double> Sv_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_face);

					std::vector<double> Sv_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sv_solution_n, Sv_vals_face_neighbor);

					std::vector<double> Sv_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1);

					std::vector<double> Sv_vals_face_nminus1_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1_neighbor);

					std::vector<Tensor<1, dim>> grad_pl_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_gradients(temp_pl_solution, grad_pl_face);

					std::vector<Tensor<1, dim>> grad_pl_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_gradients(temp_pl_solution, grad_pl_face_neighbor);

					double kappa0 = temp_kappa[cell_DG->global_active_cell_index()];
					double kappa1 = temp_kappa[cell_DG_neighbor->global_active_cell_index()];

					for (unsigned int q = 0; q < n_q_points_face_test_scalar; ++q)
					{
						const Tensor<1, dim> normal = fe_face_values_test_scalar.normal_vector(q);

						double pl_value0 = pl_vals_face[q];
						double pl_value1 = pl_vals_face_neighbor[q];

						double Sa_value0_n = Sa_vals_face[q];
						double Sa_value1_n = Sa_vals_face_neighbor[q];
						double Sa_value0_nminus1 = Sa_vals_face_nminus1[q];
						double Sa_value1_nminus1 = Sa_vals_face_nminus1_neighbor[q];

						double Sv_value0_n = Sv_vals_face[q];
						double Sv_value1_n = Sv_vals_face_neighbor[q];
						double Sv_value0_nminus1 = Sv_vals_face_nminus1[q];
						double Sv_value1_nminus1 = Sv_vals_face_nminus1_neighbor[q];

						Tensor<1,dim> neg_pl_grad = -grad_pl_face[q];
						Tensor<1,dim> neg_pl_grad_neighbor = -grad_pl_face_neighbor[q];

						if(use_exact_pl_in_RT0)
						{
							pl_fcn.set_time(time);
							pl_value0 = pl_fcn.value(q_points[q]);
							pl_value1 = pl_value0;

							neg_pl_grad = pl_fcn.gradient(q_points[q]);
							neg_pl_grad *= -1.0;

							neg_pl_grad_neighbor = neg_pl_grad;
						}

						if(use_exact_Sa_in_RT0)
						{
							Sa_fcn.set_time(time - time_step);
							Sa_value0_n = Sa_fcn.value(q_points[q]);
							Sa_value1_n = Sa_value0_n;

							Sa_fcn.set_time(time - 2.0*time_step);
							Sa_value0_nminus1 = Sa_fcn.value(q_points[q]);

							Sa_value1_nminus1 = Sa_value0_nminus1;
						}

						if(use_exact_Sv_in_RT0)
						{
							Sv_fcn.set_time(time - time_step);
							Sv_value0_n = Sv_fcn.value(q_points[q]);
							Sv_value1_n = Sv_value0_n;

							Sv_fcn.set_time(time - 2.0*time_step);
							Sv_value0_nminus1 = Sv_fcn.value(q_points[q]);
							Sv_value1_nminus1 = Sv_value0_nminus1;
						}

						double Sa_nplus1_extrapolation0 = Sa_value0_n;
						double Sa_nplus1_extrapolation1 = Sa_value1_n;
						double Sv_nplus1_extrapolation0 = Sv_value0_n;
						double Sv_nplus1_extrapolation1 = Sv_value1_n;

						if(second_order_extrapolation)
						{
							Sa_nplus1_extrapolation0 *= 2.0;
							Sa_nplus1_extrapolation0 -= Sa_value0_nminus1;

							Sa_nplus1_extrapolation1 *= 2.0;
							Sa_nplus1_extrapolation1 -= Sa_value1_nminus1;

							Sv_nplus1_extrapolation0 *= 2.0;
							Sv_nplus1_extrapolation0 -= Sv_value0_nminus1;

							Sv_nplus1_extrapolation1 *= 2.0;
							Sv_nplus1_extrapolation1 -= Sv_value1_nminus1;
						}

						double rho_l0 = rho_l_fcn.value(pl_value0);
						double rho_l1 = rho_l_fcn.value(pl_value1);

						double rho_v0 = rho_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double rho_v1 = rho_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

						double rho_a0 = rho_a_fcn.value(pl_value0);
						double rho_a1 = rho_a_fcn.value(pl_value1);

						double lambda_l0 = lambda_l_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double lambda_v0 = lambda_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double lambda_a0 = lambda_a_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);

						double lambda_l1 = lambda_l_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
						double lambda_v1 = lambda_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
						double lambda_a1 = lambda_a_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

						double rholambda_t0 = rho_l0*lambda_l0 + rho_v0*lambda_v0 + rho_a0*lambda_a0;
						double rholambda_t1 = rho_l1*lambda_l1 + rho_v1*lambda_v1 + rho_a1*lambda_a1;

						if(incompressible)
						{
							rholambda_t0 = lambda_l0 + lambda_v0 + lambda_a0;
							rholambda_t1 = lambda_l1 + lambda_v1 + lambda_a1;
						}

						if(project_only_kappa)
						{
							rholambda_t0 = 1.0;
							rholambda_t1 = 1.0;
						}

						Tensor<1,dim> g_val = gravity_fcn.vector_value(q_points[q]);

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						double coef0 = rholambda_t0*kappa0;
						double coef1 = rholambda_t1*kappa1;

						double weight0 = 0.0;
						double weight1 = 0.0;

//						if(fabs(coef0) > 1.e-14 || fabs(coef1) > 1.e-14)
//						{
							weight0 = coef1/(coef0 + coef1 + 1.e-20);
							weight1 = coef0/(coef0 + coef1 + 1.e-20);
//						}

						double weighted_aver_rhs = AverageGradOperators::weighted_average_rhs<dim>(normal,
									neg_pl_grad, neg_pl_grad_neighbor,
									coef0, coef1,
									weight0, weight1);

						double density_g0;
						double density_g1;

						if(use_Sa)
						{
							density_g0 = rho_a0;
							density_g1 = rho_a1;
						}
						else
						{
							density_g0 = rho_v0;
							density_g1 = rho_v1;
						}

						double coef0_g = rholambda_t0*kappa0*density_g0;
						double coef1_g = rholambda_t1*kappa1*density_g1;

						weight0 = 0.0;
						weight1 = 0.0;

//						if(fabs(coef0_g) > 1.e-14 || fabs(coef1_g) > 1.e-14)
//						{
							weight0 = coef1_g/(coef0_g + coef1_g + 1.e-20);
							weight1 = coef0_g/(coef0_g + coef1_g + 1.e-20);
//						}

						double weighted_aver_rhs_gravity = AverageGradOperators::weighted_average_rhs<dim>(normal,
									g_val, g_val,
									coef0_g, coef1_g,
									weight0, weight1);

						for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
						{
							const unsigned int ii = fe_support_on_face_scalar[face_num][i];

							face_RHS_RT0[i] +=
									(weighted_aver_rhs + weighted_aver_rhs_gravity)
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);

							double gamma_ch_e = 2.0*coef0*coef1/(coef0 + coef1);
							double h_e = cell_DG->face(face_num)->measure();
							double penalty_factor = (penalty_pl/h_e) * gamma_ch_e * degree*(degree + dim - 1);

							face_RHS_RT0[i] +=
									penalty_factor
									* (pl_value0 - pl_value1)
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);
						}
					}
				}

				FullMatrix<double> inv_face_matrix_RT0(dofs_per_face_test_scalar, dofs_per_face_RT0);
				inv_face_matrix_RT0.invert(face_matrix_RT0);
				Vector<double> sol_local_face_RT0(face_RHS_RT0.size());

				inv_face_matrix_RT0.vmult(sol_local_face_RT0, face_RHS_RT0);

				std::vector<unsigned int> dof_ind(1);
				face->get_dof_indices(dof_ind);

				totalDarcyvelocity_RT0[dof_ind[0]] = sol_local_face_RT0[0];

			}
		}

		totalDarcyvelocity_RT0.compress(VectorOperation::insert);
	}

	return totalDarcyvelocity_RT0;
}


template<int dim>
PETScWrappers::MPI::Vector compute_RTk_projection(Triangulation<dim, dim> &triangulation, const unsigned int degree, double theta_pl, double time,
		double time_step, double penalty_pl, double penalty_pl_bdry, std::vector<unsigned int> dirichlet_id_pl, bool use_exact_pl_in_RT0,
		bool use_exact_Sa_in_RT0, bool use_exact_Sv_in_RT0, bool second_order_extrapolation, bool incompressible,
		PETScWrappers::MPI::Vector pl_solution, PETScWrappers::MPI::Vector Sa_solution_n, PETScWrappers::MPI::Vector Sa_solution_nminus1,
		PETScWrappers::MPI::Vector Sv_solution_n, PETScWrappers::MPI::Vector Sv_solution_nminus1, PETScWrappers::MPI::Vector kappa_abs_vec,
		bool project_only_kappa, MPI_Comm mpi_communicator, const unsigned int n_mpi_processes, const unsigned int this_mpi_process)
{
	// Testing
//	FE_DGVector<LocalPolynomialsRT<dim>(degree), dim> fe_test(degree, mapping_raviart_thomas);
	FE_RTProj<dim> fe_testing(degree);
	// End testing

	FE_DGQ<dim>     fe(degree);
	DoFHandler<dim> dof_handler(triangulation);

	dof_handler.distribute_dofs(fe);

	IndexSet locally_owned_dofs_DG;
	IndexSet locally_relevant_dofs_DG;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_DG =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
	locally_owned_dofs_DG = locally_owned_dofs_per_proc_DG[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs_DG);

	// Kappa stuff
	FE_DGQ<dim> fe_dg0(0);
	DoFHandler<dim> dof_handler_dg0(triangulation);
	IndexSet locally_owned_dofs_dg0;
	IndexSet locally_relevant_dofs_dg0;

	dof_handler_dg0.distribute_dofs(fe_dg0);
	const std::vector<IndexSet> locally_owned_dofs_per_proc_dg0 =
			DoFTools::locally_owned_dofs_per_subdomain(dof_handler_dg0);
	locally_owned_dofs_dg0 = locally_owned_dofs_per_proc_dg0[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_dg0, locally_relevant_dofs_dg0);

	// RT Projection vector
	PETScWrappers::MPI::Vector totalDarcyvelocity_RT;

	// RT Projection space
	FE_RaviartThomas<dim> fe_RT(degree);
	DoFHandler<dim> dof_handler_RT(triangulation);

	dof_handler_RT.distribute_dofs(fe_RT);

	IndexSet locally_owned_dofs_RT;
	IndexSet locally_relevant_dofs_RT;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_RT =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_RT);
	locally_owned_dofs_RT = locally_owned_dofs_per_proc_RT[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_RT, locally_relevant_dofs_RT);

	// DG space on faces for RT Projection
	FE_FaceQ<dim> fe_test_scalar(degree);
	DoFHandler<dim> dof_handler_test_scalar(triangulation);

	dof_handler_test_scalar.distribute_dofs(fe_test_scalar);

	IndexSet locally_owned_dofs_test_scalar;
	IndexSet locally_relevant_dofs_test_scalar;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_test_scalar =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_test_scalar);
	locally_owned_dofs_test_scalar = locally_owned_dofs_per_proc_test_scalar[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_test_scalar, locally_relevant_dofs_test_scalar);

	// RT space on elements for RT projection
	// old:
//	FE_DGRaviartThomas<dim> fe_test_vector(degree-1);
	// new:
	FE_RTProj<dim> fe_test_vector(degree);

	DoFHandler<dim> dof_handler_test_vector(triangulation);

	dof_handler_test_vector.distribute_dofs(fe_test_vector);

	IndexSet locally_owned_dofs_test_vector;
	IndexSet locally_relevant_dofs_test_vector;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_test_vector =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_test_vector);
	locally_owned_dofs_test_vector = locally_owned_dofs_per_proc_test_vector[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_test_vector, locally_relevant_dofs_test_vector);

	totalDarcyvelocity_RT.reinit(locally_owned_dofs_RT, mpi_communicator);

	const QGauss<dim>     quadrature_formula(fe_RT.degree + 1);
	const QGauss<dim - 1> face_quadrature_formula(fe_RT.degree + 1);

	BoundaryValuesLiquidPressure<dim> boundary_function;
	boundary_function.set_time(time);

    // Densities
    rho_l<dim> rho_l_fcn;
    rho_v<dim> rho_v_fcn;
    rho_a<dim> rho_a_fcn;

    // Mobilities
    lambda_l<dim> lambda_l_fcn;
    lambda_v<dim> lambda_v_fcn;
    lambda_a<dim> lambda_a_fcn;

    ExactLiquidPressure<dim> pl_fcn;
    ExactAqueousSaturation<dim> Sa_fcn;
	ExactVaporSaturation<dim> Sv_fcn;

	FEValues<dim> fe_values_DG(fe,
							   quadrature_formula,
							   update_values | update_gradients |
							   update_quadrature_points |
							   update_JxW_values);

	FEFaceValues<dim> fe_face_values_DG(fe,
										face_quadrature_formula,
										update_values | update_gradients |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_DG_neighbor(fe,
										face_quadrature_formula,
										update_values | update_gradients |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEValues<dim> fe_values_RT(fe_RT,
							   quadrature_formula,
							   update_values | update_gradients |
							   update_quadrature_points |
							   update_JxW_values);

	FEFaceValues<dim> fe_face_values_RT(fe_RT,
										face_quadrature_formula,
										update_values |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_test_scalar(fe_test_scalar,
										  face_quadrature_formula,
										  update_values |
										  update_normal_vectors |
										  update_quadrature_points |
										  update_JxW_values);

	FEValues<dim> fe_values_test_vector(fe_test_vector,
							   	   	    quadrature_formula,
										update_values | update_gradients |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_test_vector(fe_test_vector,
												 face_quadrature_formula,
												 update_values |
												 update_normal_vectors |
												 update_quadrature_points |
												 update_JxW_values);

	const unsigned int dofs_per_cell_DG = fe.dofs_per_cell;
	const unsigned int dofs_per_cell_RT = fe_RT.dofs_per_cell;
	const unsigned int dofs_per_face_RT = fe_RT.n_dofs_per_face();
	const unsigned int dofs_per_face_test_scalar = fe_test_scalar.n_dofs_per_face();
	const unsigned int dofs_per_face_test_vector = fe_test_vector.n_dofs_per_face();
	const unsigned int dofs_per_cell_test_vector = fe_test_vector.dofs_per_cell;

//	std::cout << "dofs_per_cell_RT= " << dofs_per_cell_RT << std::endl;
//	std::cout << "dofs_per_cell_test_vector= " << dofs_per_cell_test_vector << std::endl;
//	std::cout << "dofs_per_face_test_vector= " << dofs_per_face_test_vector << std::endl;
//	std::cout << "dofs_per_face_test_scalar= " << dofs_per_face_test_scalar << std::endl;

	std::vector<std::vector<unsigned int>> fe_support_on_face_scalar(GeometryInfo<dim>::faces_per_cell);

	for (unsigned int face_no : GeometryInfo<dim>::face_indices())
	{
		for (unsigned int i = 0; i < fe_test_scalar.dofs_per_cell; ++i)
		{
			if (fe_test_scalar.has_support_on_face(i, face_no))
				fe_support_on_face_scalar[face_no].push_back(i);
		}
	}

	const FEValuesExtractors::Vector velocities(0);

	std::vector<double> basis_fcns_faces_scalar(fe_test_scalar.dofs_per_cell);

	unsigned int nrows_full = dofs_per_cell_test_vector + GeometryInfo<dim>::faces_per_cell*dofs_per_face_test_scalar;
	unsigned int ncols_full = nrows_full;

//	std::cout << "nrows_full = " << nrows_full << std::endl;

	FullMatrix<double> local_matrix_RT(nrows_full, ncols_full);
	Vector<double> local_solution_RT(ncols_full);
	Vector<double> local_RHS_RT(nrows_full);

//	Vector<double> sol_all_faces_one_cell(GeometryInfo<dim>::faces_per_cell*dofs_per_face_test_scalar);

	std::vector<types::global_dof_index> local_dof_indices_DG(dofs_per_cell_DG);
	std::vector<types::global_dof_index> local_dof_indices_RT(dofs_per_cell_RT);

	typename DoFHandler<dim>::active_cell_iterator
		  cell_DG = dof_handler.begin_active(),
		  endc = dof_handler.end(), cell_RT = dof_handler_RT.begin_active(),
		  cell_scalar = dof_handler_test_scalar.begin_active(),
		  cell_vector = dof_handler_test_vector.begin_active();

    const unsigned int n_q_points_DG = fe_values_DG.get_quadrature().size();
    const unsigned int n_q_points_RT = fe_values_RT.get_quadrature().size();
	const unsigned int n_q_points_face_DG = fe_face_values_DG.get_quadrature().size();
	const unsigned int n_q_points_face_test_scalar = fe_face_values_test_scalar.get_quadrature().size();
	const unsigned int n_q_points_cell_test_vector = fe_values_test_vector.get_quadrature().size();
	const unsigned int n_q_points_face_test_vector = fe_face_values_test_vector.get_quadrature().size();

	PETScWrappers::MPI::Vector temp_pl_solution;

	PETScWrappers::MPI::Vector temp_Sa_solution_n;
	PETScWrappers::MPI::Vector temp_Sa_solution_nminus1;

	PETScWrappers::MPI::Vector temp_Sv_solution_n;
	PETScWrappers::MPI::Vector temp_Sv_solution_nminus1;

	PETScWrappers::MPI::Vector temp_kappa;

	temp_pl_solution.reinit(locally_owned_dofs_DG,
							locally_relevant_dofs_DG,
							mpi_communicator);

	temp_Sa_solution_n.reinit(locally_owned_dofs_DG,
							  locally_relevant_dofs_DG,
							  mpi_communicator);

	temp_Sa_solution_nminus1.reinit(locally_owned_dofs_DG,
									locally_relevant_dofs_DG,
									mpi_communicator);

	temp_Sv_solution_n.reinit(locally_owned_dofs_DG,
							  locally_relevant_dofs_DG,
							  mpi_communicator);

	temp_Sv_solution_nminus1.reinit(locally_owned_dofs_DG,
									locally_relevant_dofs_DG,
									mpi_communicator);

	temp_kappa.reinit(locally_owned_dofs_dg0,
					  locally_relevant_dofs_dg0,
					  mpi_communicator);

	temp_pl_solution = pl_solution;

	temp_Sa_solution_n = Sa_solution_n;
	temp_Sa_solution_nminus1 = Sa_solution_nminus1;

	temp_Sv_solution_n = Sv_solution_n;
	temp_Sv_solution_nminus1 = Sv_solution_nminus1;

	temp_kappa = kappa_abs_vec;

//	std::vecor<int> face_done(GeometryInfo<dim>::)
	for (; cell_DG != endc; ++cell_DG, ++cell_RT, ++cell_scalar, ++cell_vector)
	{
//		if (cell_DG->subdomain_id() == this_mpi_process)
		if(cell_DG->is_locally_owned())
		{
			fe_values_DG.reinit(cell_DG);
			fe_values_RT.reinit(cell_RT);
			fe_values_test_vector.reinit(cell_vector);

			const auto &q_points = fe_values_test_vector.get_quadrature_points();

	//		face_solution_RT = 0.0;
	//		sol_all_faces_one_cell = 0.0;

			local_matrix_RT = 0.0;
			local_RHS_RT = 0.0;
			local_solution_RT = 0.0;

			std::vector<double> pl_vals_cell(n_q_points_DG);
			fe_values_DG.get_function_values(temp_pl_solution, pl_vals_cell);

			std::vector<double> Sa_vals_cell_n(n_q_points_DG);
			fe_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_cell_n);

			std::vector<double> Sa_vals_cell_nminus1(n_q_points_DG);
			fe_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_cell_nminus1);

			std::vector<double> Sv_vals_cell_n(n_q_points_DG);
			fe_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_cell_n);

			std::vector<double> Sv_vals_cell_nminus1(n_q_points_DG);
			fe_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_cell_nminus1);

			std::vector<Tensor<1, dim>> grad_pl_cell(n_q_points_DG);
			fe_values_DG.get_function_gradients(temp_pl_solution, grad_pl_cell);

			double kappa = temp_kappa[cell_DG->global_active_cell_index()];

			for (unsigned int q = 0; q < n_q_points_cell_test_vector; ++q)
			{
				Tensor<1,dim> neg_pl_grad = -grad_pl_cell[q];

				double pl_value = pl_vals_cell[q];
				double Sa_value_n = Sa_vals_cell_n[q];
				double Sa_value_nminus1 = Sa_vals_cell_nminus1[q];
				double Sv_value_n = Sv_vals_cell_n[q];
				double Sv_value_nminus1 = Sv_vals_cell_nminus1[q];

				if(use_exact_pl_in_RT0)
				{
					pl_fcn.set_time(time);
					pl_value = pl_fcn.value(q_points[q]);

					neg_pl_grad = pl_fcn.gradient(q_points[q]);
					neg_pl_grad *= -1.0;
				}

				if(use_exact_Sa_in_RT0)
				{
					Sa_fcn.set_time(time - time_step);
					Sa_value_n = Sa_fcn.value(q_points[q]);

					Sa_fcn.set_time(time - 2.0*time_step);
					Sa_value_nminus1 = Sa_fcn.value(q_points[q]);
				}

				if(use_exact_Sv_in_RT0)
				{
					Sv_fcn.set_time(time - time_step);
					Sv_value_n = Sv_fcn.value(q_points[q]);

					Sv_fcn.set_time(time - 2.0*time_step);
					Sv_value_nminus1 = Sv_fcn.value(q_points[q]);
				}

				double Sa_nplus1_extrapolation = Sa_value_n;
				double Sv_nplus1_extrapolation = Sv_value_n;

				if(second_order_extrapolation)
				{
					Sa_nplus1_extrapolation *= 2.0;
					Sa_nplus1_extrapolation -= Sa_value_nminus1;

					Sv_nplus1_extrapolation *= 2.0;
					Sv_nplus1_extrapolation -= Sv_value_nminus1;

				}

				double rho_l = rho_l_fcn.value(pl_value);
				double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double rho_a = rho_a_fcn.value(pl_value);

				double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

				double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

				if(incompressible)
					rholambda_t = lambda_l + lambda_v + lambda_a;

				if(project_only_kappa)
					rholambda_t = 1.0;

				for(unsigned int i = 0; i < dofs_per_cell_test_vector; ++i)
				{
					// old:
//					for(unsigned int j = 0; j < dofs_per_cell_RT; ++j)
//						local_matrix_RT(i, j) +=
//								(fe_values_test_vector.shape_value_component(i, q, 1)
//								 * fe_values_RT.shape_value_component(j, q, 0)
//								 + fe_values_test_vector.shape_value_component(i, q, 0)
//								 * fe_values_RT.shape_value_component(j, q, 1))
//								* fe_values_test_vector.JxW(q);
					// new:
					for(unsigned int j = 0; j < dofs_per_cell_RT; ++j)
						local_matrix_RT(i, j) +=
								fe_values_test_vector[velocities].value(i, q)
								 * fe_values_RT[velocities].value(j, q)
								* fe_values_test_vector.JxW(q);
//					std::cout << "cell 1" << std::endl;
					// old:
//					local_RHS_RT[i] += kappa
//									   *rholambda_t
//									   * (neg_pl_grad[0]
//											* fe_values_test_vector.shape_value_component(i, q, 1)
//											+ neg_pl_grad[1]
//											* fe_values_test_vector.shape_value_component(i, q, 0))
//									   * fe_values_test_vector.JxW(q);

					// new:
					local_RHS_RT[i] += kappa
									   * rholambda_t
									   * neg_pl_grad
									   * fe_values_test_vector[velocities].value(i, q)
									   * fe_values_test_vector.JxW(q);
//					std::cout << "cell 2" << std::endl;
				}
			}

			for (const auto &face : cell_vector->face_iterators())
			{
				unsigned int face_num = cell_scalar->face_iterator_to_index(face);

				fe_face_values_DG.reinit(cell_DG, face);
				fe_face_values_RT.reinit(cell_RT, face);
				fe_face_values_test_scalar.reinit(cell_scalar, face);
				fe_face_values_test_vector.reinit(cell_vector, face);

				const auto &q_points_face = fe_face_values_test_scalar.get_quadrature_points();

				std::vector<double> g(n_q_points_face_test_scalar);
				boundary_function.value_list(fe_face_values_test_scalar.get_quadrature_points(), g);

				for (unsigned int q = 0; q < n_q_points_face_test_scalar; ++q)
				{
					const Tensor<1, dim> normal = fe_face_values_test_scalar.normal_vector(q);

					for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
						basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

					for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
					{
						const unsigned int ii = fe_support_on_face_scalar[face_num][i];

						for (unsigned int j = 0; j < dofs_per_cell_RT; ++j)
						{
							local_matrix_RT(dofs_per_cell_test_vector + ii, j) +=
									fe_face_values_RT[velocities].value(j, q)
									* normal
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);
						}
					}
				}

				if(face->at_boundary())
				{
					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					std::vector<double> Sa_vals_face_n(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_face_n);

					std::vector<double> Sa_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1);

					std::vector<double> Sv_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_face);

					std::vector<double> Sv_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1);

					std::vector<Tensor<1, dim>> grad_pl_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_gradients(temp_pl_solution, grad_pl_face);

					double kappa = temp_kappa[cell_DG->global_active_cell_index()];

					for (unsigned int q = 0; q < n_q_points_face_test_vector; ++q)
					{
						double pl_value = pl_vals_face[q];
						Tensor<1,dim> neg_pl_grad = grad_pl_face[q];
						neg_pl_grad *= -1.0;

						double Sa_value_n = Sa_vals_face_n[q];
						double Sa_value_nminus1 = Sa_vals_face_nminus1[q];
						double Sv_value_n = Sv_vals_face[q];
						double Sv_value_nminus1 = Sv_vals_face_nminus1[q];

						if(use_exact_pl_in_RT0)
						{
							pl_fcn.set_time(time);
							pl_value = pl_fcn.value(q_points_face[q]);
							neg_pl_grad = pl_fcn.gradient(q_points_face[q]);
							neg_pl_grad *= -1.0;
						}

						if(use_exact_Sa_in_RT0)
						{
							Sa_fcn.set_time(time - time_step);
							Sa_value_n = Sa_fcn.value(q_points_face[q]);

							Sa_fcn.set_time(time - 2.0*time_step);
							Sa_value_nminus1 = Sa_fcn.value(q_points_face[q]);
						}

						if(use_exact_Sv_in_RT0)
						{
							Sv_fcn.set_time(time - time_step);
							Sv_value_n = Sv_fcn.value(q_points_face[q]);

							Sv_fcn.set_time(time - 2.0*time_step);
							Sv_value_nminus1 = Sv_fcn.value(q_points_face[q]);
						}

						double Sa_nplus1_extrapolation = Sa_value_n;
						double Sv_nplus1_extrapolation = Sv_value_n;

						if(second_order_extrapolation)
						{
							Sa_nplus1_extrapolation *= 2.0;
							Sa_nplus1_extrapolation -= Sa_value_nminus1;

							Sv_nplus1_extrapolation *= 2.0;
							Sv_nplus1_extrapolation -= Sv_value_nminus1;

						}

						double rho_l = rho_l_fcn.value(pl_value);
						double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double rho_a = rho_a_fcn.value(pl_value);

						double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

						double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

						if(incompressible)
							rholambda_t = lambda_l + lambda_v + lambda_a;

						if(project_only_kappa)
							rholambda_t = 1.0;

						const Tensor<1, dim> normal = fe_face_values_test_vector.normal_vector(q);

						// Figure out if this face is Dirichlet or Neumann
						bool dirichlet = false;

						for(unsigned int kk = 0; kk < dirichlet_id_pl.size(); kk++)
						{
							if(face->boundary_id() == dirichlet_id_pl[kk])
							{
								dirichlet = true;
								break;
							}
						}

						if(dirichlet)
						{
							// old:
//							for(unsigned int i = 0; i < dofs_per_cell_test_vector; ++i)
//								local_RHS_RT[i] -= theta_pl
//												  * rholambda_t
//												  * kappa
//												  * (pl_value - g[q])
//												  * (fe_face_values_test_vector.shape_value_component(i, q, 1)
//													 * normal[0]
//													 + fe_face_values_test_vector.shape_value_component(i, q, 0)
//													 * normal[1])
//												  * fe_face_values_test_vector.JxW(q);
							// new:
							for(unsigned int i = 0; i < dofs_per_cell_test_vector; ++i)
								local_RHS_RT[i] -= theta_pl
												  * rholambda_t
												  * kappa
												  * (pl_value - g[q])
												  * fe_face_values_test_vector[velocities].value(i, q)
												  * normal
												  * fe_face_values_test_vector.JxW(q);
//							std::cout << "bdr 1" << std::endl;
						}

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						double gamma_ch_e = rholambda_t*kappa;
						double h_e = cell_DG->face(face_num)->measure();
						double penalty_factor = (penalty_pl_bdry/h_e) * gamma_ch_e * degree*(degree + dim - 1);

						if(dirichlet)
						{
							for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
							{
								const unsigned int ii = fe_support_on_face_scalar[face_num][i];

								local_RHS_RT[dofs_per_cell_test_vector + ii] +=
											rholambda_t
											* kappa
											* neg_pl_grad
											* normal
											* basis_fcns_faces_scalar[i]
											* fe_face_values_test_scalar.JxW(q);

								local_RHS_RT[dofs_per_cell_test_vector + ii] +=
											penalty_factor
											* (pl_value - g[q])
											* basis_fcns_faces_scalar[i]
											* fe_face_values_test_scalar.JxW(q);


							}
						}
					}
				}
				else // interior faces
				{
					typename DoFHandler<dim>::active_cell_iterator cell_DG_neighbor = cell_DG->neighbor(face_num);
					unsigned int neighbor_index = cell_DG->neighbor_index(face_num);

					fe_face_values_DG_neighbor.reinit(cell_DG_neighbor, face);

					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					std::vector<double> pl_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_pl_solution, pl_vals_face_neighbor);

					std::vector<double> Sa_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_face);

					std::vector<double> Sa_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sa_solution_n, Sa_vals_face_neighbor);

					std::vector<double> Sa_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1);

					std::vector<double> Sa_vals_face_nminus1_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1_neighbor);

					std::vector<double> Sv_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_face);

					std::vector<double> Sv_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sv_solution_n, Sv_vals_face_neighbor);

					std::vector<double> Sv_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1);

					std::vector<double> Sv_vals_face_nminus1_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1_neighbor);

					std::vector<Tensor<1, dim>> grad_pl_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_gradients(temp_pl_solution, grad_pl_face);

					std::vector<Tensor<1, dim>> grad_pl_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_gradients(temp_pl_solution, grad_pl_face_neighbor);

					double kappa0 = temp_kappa[cell_DG->global_active_cell_index()];
					double kappa1 = temp_kappa[cell_DG_neighbor->global_active_cell_index()];

					for (unsigned int q = 0; q < n_q_points_face_test_vector; ++q)
					{
						double pl_value0 = pl_vals_face[q];
						double pl_value1 = pl_vals_face_neighbor[q];

						double Sa_value0_n = Sa_vals_face[q];
						double Sa_value1_n = Sa_vals_face_neighbor[q];
						double Sa_value0_nminus1 = Sa_vals_face_nminus1[q];
						double Sa_value1_nminus1 = Sa_vals_face_nminus1_neighbor[q];

						double Sv_value0_n = Sv_vals_face[q];
						double Sv_value1_n = Sv_vals_face_neighbor[q];
						double Sv_value0_nminus1 = Sv_vals_face_nminus1[q];
						double Sv_value1_nminus1 = Sv_vals_face_nminus1_neighbor[q];

						Tensor<1,dim> neg_pl_grad = -grad_pl_face[q];
						Tensor<1,dim> neg_pl_grad_neighbor = -grad_pl_face_neighbor[q];

						if(use_exact_pl_in_RT0)
						{
							pl_fcn.set_time(time);
							pl_value0 = pl_fcn.value(q_points_face[q]);
							pl_value1 = pl_value0;

							neg_pl_grad = pl_fcn.gradient(q_points_face[q]);
							neg_pl_grad *= -1.0;

							neg_pl_grad_neighbor = neg_pl_grad;
						}

						if(use_exact_Sa_in_RT0)
						{
							Sa_fcn.set_time(time - time_step);
							Sa_value0_n = Sa_fcn.value(q_points_face[q]);
							Sa_value1_n = Sa_value0_n;

							Sa_fcn.set_time(time - 2.0*time_step);
							Sa_value0_nminus1 = Sa_fcn.value(q_points_face[q]);

							Sa_value1_nminus1 = Sa_value0_nminus1;
						}

						if(use_exact_Sv_in_RT0)
						{
							Sv_fcn.set_time(time - time_step);
							Sv_value0_n = Sv_fcn.value(q_points_face[q]);
							Sv_value1_n = Sv_value0_n;

							Sv_fcn.set_time(time - 2.0*time_step);
							Sv_value0_nminus1 = Sv_fcn.value(q_points_face[q]);
							Sv_value1_nminus1 = Sv_value0_nminus1;
						}

						double Sa_nplus1_extrapolation0 = Sa_value0_n;
						double Sa_nplus1_extrapolation1 = Sa_value1_n;
						double Sv_nplus1_extrapolation0 = Sv_value0_n;
						double Sv_nplus1_extrapolation1 = Sv_value1_n;

						if(second_order_extrapolation)
						{
							Sa_nplus1_extrapolation0 *= 2.0;
							Sa_nplus1_extrapolation0 -= Sa_value0_nminus1;

							Sa_nplus1_extrapolation1 *= 2.0;
							Sa_nplus1_extrapolation1 -= Sa_value1_nminus1;

							Sv_nplus1_extrapolation0 *= 2.0;
							Sv_nplus1_extrapolation0 -= Sv_value0_nminus1;

							Sv_nplus1_extrapolation1 *= 2.0;
							Sv_nplus1_extrapolation1 -= Sv_value1_nminus1;
						}

						double rho_l0 = rho_l_fcn.value(pl_value0);
						double rho_l1 = rho_l_fcn.value(pl_value1);

						double rho_v0 = rho_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double rho_v1 = rho_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

						double rho_a0 = rho_a_fcn.value(pl_value0);
						double rho_a1 = rho_a_fcn.value(pl_value1);

						double lambda_l0 = lambda_l_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double lambda_v0 = lambda_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double lambda_a0 = lambda_a_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);

						double lambda_l1 = lambda_l_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
						double lambda_v1 = lambda_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
						double lambda_a1 = lambda_a_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

						double rholambda_t0 = rho_l0*lambda_l0 + rho_v0*lambda_v0 + rho_a0*lambda_a0;
						double rholambda_t1 = rho_l1*lambda_l1 + rho_v1*lambda_v1 + rho_a1*lambda_a1;

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						if(incompressible)
						{
							rholambda_t0 = lambda_l0 + lambda_v0 + lambda_a0;
							rholambda_t1 = lambda_l1 + lambda_v1 + lambda_a1;
						}

						if(project_only_kappa)
						{
							rholambda_t0 = 1.0;
							rholambda_t1 = 1.0;
						}

						double coef0 = rholambda_t0*kappa0;
						double coef1 = rholambda_t1*kappa1;

						double weight0 = 0.0;
						double weight1 = 0.0;

//						if(fabs(coef0) > 1.e-14 || fabs(coef1) > 1.e-14)
//						{
							weight0 = coef1/(coef0 + coef1 + 1.e-20);
							weight1 = coef0/(coef0 + coef1 + 1.e-20);
//						}

						const Tensor<1, dim> normal = fe_face_values_test_vector.normal_vector(q);

						// old:
//						for(unsigned int i = 0; i < dofs_per_cell_test_vector; ++i)
//							local_RHS_RT[i] -= theta_pl
//											  * weight0
//											  * coef0
//											  * (pl_value0 - pl_value1)
//											  * (fe_face_values_test_vector.shape_value_component(i, q, 1)
//												 * normal[0]
//												 + fe_face_values_test_vector.shape_value_component(i, q, 0)
//												 * normal[1])
//											  * fe_face_values_test_vector.JxW(q);

						// new:
						for(unsigned int i = 0; i < dofs_per_cell_test_vector; ++i)
							local_RHS_RT[i] -= theta_pl
											  * weight0
											  * coef0
											  * (pl_value0 - pl_value1)
											  * fe_face_values_test_vector[velocities].value(i, q)
											  * normal
											  * fe_face_values_test_vector.JxW(q);

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						double weighted_aver_rhs = AverageGradOperators::weighted_average_rhs<dim>(normal,
									neg_pl_grad, neg_pl_grad_neighbor,
									coef0, coef1,
									weight0, weight1);

						for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
						{
							const unsigned int ii = fe_support_on_face_scalar[face_num][i];

							local_RHS_RT[dofs_per_cell_test_vector + ii] +=
									weighted_aver_rhs
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);

							double gamma_ch_e = 2.0*coef0*coef1/(coef0 + coef1 + 1.e-20);
				            double h_e = cell_DG->face(face_num)->measure();
				            double penalty_factor = (penalty_pl/h_e) * gamma_ch_e * degree*(degree + dim - 1);

							local_RHS_RT[dofs_per_cell_test_vector + ii] +=
									penalty_factor
									* (pl_value0 - pl_value1)
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);
						}
					}
				}

			}
			FullMatrix<double> inv_local_matrix_RT(nrows_full, ncols_full);
			inv_local_matrix_RT.invert(local_matrix_RT);

			Vector<double> sol_local_RT(nrows_full);

			inv_local_matrix_RT.vmult(sol_local_RT, local_RHS_RT);

			std::vector<unsigned int> dof_ind(dofs_per_cell_RT);
			cell_RT->get_dof_indices(dof_ind);

			for(unsigned int k = 0; k < dofs_per_cell_RT; k++)
				totalDarcyvelocity_RT[dof_ind[k]] = sol_local_RT[k];
		}
	}
    totalDarcyvelocity_RT.compress(VectorOperation::insert);
	return totalDarcyvelocity_RT;
}


template<int dim>
PETScWrappers::MPI::Vector compute_RTk_projection_with_gravity(Triangulation<dim, dim> &triangulation, const unsigned int degree, double theta_pl, double time,
		double time_step, double penalty_pl, double penalty_pl_bdry, std::vector<unsigned int> dirichlet_id_pl, bool use_exact_pl_in_RT0,
		bool use_exact_Sa_in_RT0, bool use_exact_Sv_in_RT0, bool second_order_extrapolation, bool incompressible,
		PETScWrappers::MPI::Vector pl_solution, PETScWrappers::MPI::Vector Sa_solution_n, PETScWrappers::MPI::Vector Sa_solution_nminus1,
		PETScWrappers::MPI::Vector Sv_solution_n, PETScWrappers::MPI::Vector Sv_solution_nminus1, PETScWrappers::MPI::Vector kappa_abs_vec,
		bool use_Sa, bool project_only_kappa, MPI_Comm mpi_communicator, const unsigned int n_mpi_processes, const unsigned int this_mpi_process)
{
	FE_DGQ<dim>     fe(degree);
	DoFHandler<dim> dof_handler(triangulation);

	dof_handler.distribute_dofs(fe);

	IndexSet locally_owned_dofs_DG;
	IndexSet locally_relevant_dofs_DG;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_DG =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
	locally_owned_dofs_DG = locally_owned_dofs_per_proc_DG[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs_DG);

	// Kappa stuff
	FE_DGQ<dim> fe_dg0(0);
	DoFHandler<dim> dof_handler_dg0(triangulation);
	IndexSet locally_owned_dofs_dg0;
	IndexSet locally_relevant_dofs_dg0;

	dof_handler_dg0.distribute_dofs(fe_dg0);
	const std::vector<IndexSet> locally_owned_dofs_per_proc_dg0 =
			DoFTools::locally_owned_dofs_per_subdomain(dof_handler_dg0);
	locally_owned_dofs_dg0 = locally_owned_dofs_per_proc_dg0[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_dg0, locally_relevant_dofs_dg0);

	// RT Projection vector
	PETScWrappers::MPI::Vector totalDarcyvelocity_RT;

	// RT Projection space
	FE_RaviartThomas<dim> fe_RT(degree);
	DoFHandler<dim> dof_handler_RT(triangulation);

	dof_handler_RT.distribute_dofs(fe_RT);

	IndexSet locally_owned_dofs_RT;
	IndexSet locally_relevant_dofs_RT;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_RT =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_RT);
	locally_owned_dofs_RT = locally_owned_dofs_per_proc_RT[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_RT, locally_relevant_dofs_RT);

	// DG space on faces for RT Projection
	FE_FaceP<dim> fe_test_scalar(degree);
	DoFHandler<dim> dof_handler_test_scalar(triangulation);

	dof_handler_test_scalar.distribute_dofs(fe_test_scalar);

	IndexSet locally_owned_dofs_test_scalar;
	IndexSet locally_relevant_dofs_test_scalar;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_test_scalar =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_test_scalar);
	locally_owned_dofs_test_scalar = locally_owned_dofs_per_proc_test_scalar[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_test_scalar, locally_relevant_dofs_test_scalar);

	// RT space on elements for RT projection
	FE_DGRaviartThomas<dim> fe_test_vector(degree-1);
	DoFHandler<dim> dof_handler_test_vector(triangulation);

	dof_handler_test_vector.distribute_dofs(fe_test_vector);

	IndexSet locally_owned_dofs_test_vector;
	IndexSet locally_relevant_dofs_test_vector;

	const std::vector<IndexSet> locally_owned_dofs_per_proc_test_vector =
			  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_test_vector);
	locally_owned_dofs_test_vector = locally_owned_dofs_per_proc_test_vector[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_test_vector, locally_relevant_dofs_test_vector);

	totalDarcyvelocity_RT.reinit(locally_owned_dofs_RT, mpi_communicator);

	const QGauss<dim>     quadrature_formula(fe_RT.degree + 1);
	const QGauss<dim - 1> face_quadrature_formula(fe_RT.degree + 1);

	BoundaryValuesLiquidPressure<dim> boundary_function;
	boundary_function.set_time(time);

    // Densities
    rho_l<dim> rho_l_fcn;
    rho_v<dim> rho_v_fcn;
    rho_a<dim> rho_a_fcn;

    // Mobilities
    lambda_l<dim> lambda_l_fcn;
    lambda_v<dim> lambda_v_fcn;
    lambda_a<dim> lambda_a_fcn;

    // Gravity
	GravitySourceTerm<dim> gravity_fcn;

	gravity_fcn.set_time(time);

    ExactLiquidPressure<dim> pl_fcn;
    ExactAqueousSaturation<dim> Sa_fcn;
	ExactVaporSaturation<dim> Sv_fcn;

	FEValues<dim> fe_values_DG(fe,
							   quadrature_formula,
							   update_values | update_gradients |
							   update_quadrature_points |
							   update_JxW_values);

	FEFaceValues<dim> fe_face_values_DG(fe,
										face_quadrature_formula,
										update_values | update_gradients |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_DG_neighbor(fe,
										face_quadrature_formula,
										update_values | update_gradients |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEValues<dim> fe_values_RT(fe_RT,
							   quadrature_formula,
							   update_values | update_gradients |
							   update_quadrature_points |
							   update_JxW_values);

	FEFaceValues<dim> fe_face_values_RT(fe_RT,
										face_quadrature_formula,
										update_values |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_test_scalar(fe_test_scalar,
										  face_quadrature_formula,
										  update_values |
										  update_normal_vectors |
										  update_quadrature_points |
										  update_JxW_values);

	FEValues<dim> fe_values_test_vector(fe_test_vector,
							   	   	    quadrature_formula,
										update_values | update_gradients |
										update_quadrature_points |
										update_JxW_values);

	FEFaceValues<dim> fe_face_values_test_vector(fe_test_vector,
												 face_quadrature_formula,
												 update_values |
												 update_normal_vectors |
												 update_quadrature_points |
												 update_JxW_values);

	const unsigned int dofs_per_cell_DG = fe.dofs_per_cell;
	const unsigned int dofs_per_cell_RT = fe_RT.dofs_per_cell;
	const unsigned int dofs_per_face_RT = fe_RT.n_dofs_per_face();
	const unsigned int dofs_per_face_test_scalar = fe_test_scalar.n_dofs_per_face();
	const unsigned int dofs_per_face_test_vector = fe_test_vector.n_dofs_per_face();
	const unsigned int dofs_per_cell_test_vector = fe_test_vector.dofs_per_cell;

//	std::cout << "dofs_per_cell_test_vector= " << dofs_per_cell_test_vector << std::endl;
//	std::cout << "dofs_per_face_test_vector= " << dofs_per_face_test_vector << std::endl;
//	std::cout << "dofs_per_face_test_scalar= " << dofs_per_face_test_scalar << std::endl;

	std::vector<std::vector<unsigned int>> fe_support_on_face_scalar(GeometryInfo<dim>::faces_per_cell);
	std::vector<std::vector<unsigned int>> fe_support_on_face_vector(GeometryInfo<dim>::faces_per_cell);
	std::vector<std::vector<unsigned int>> fe_support_on_face_RT(GeometryInfo<dim>::faces_per_cell);

	for (unsigned int face_no : GeometryInfo<dim>::face_indices())
	{
		for (unsigned int i = 0; i < fe_test_scalar.dofs_per_cell; ++i)
		{
			if (fe_test_scalar.has_support_on_face(i, face_no))
				fe_support_on_face_scalar[face_no].push_back(i);
		}
		for (unsigned int i = 0; i < fe_test_vector.dofs_per_cell; ++i)
		{
			if (fe_test_vector.has_support_on_face(i, face_no))
				fe_support_on_face_vector[face_no].push_back(i);
		}
		for (unsigned int i = 0; i < fe_RT.dofs_per_cell; ++i)
		{
			if (fe_RT.has_support_on_face(i, face_no))
				fe_support_on_face_RT[face_no].push_back(i);
		}
	}

	const FEValuesExtractors::Vector velocities(0);

	std::vector<double> basis_fcns_faces_scalar(fe_test_scalar.dofs_per_cell);
	std::vector<double> basis_fcns_faces_vector(fe_test_vector.dofs_per_cell);
	std::vector<Tensor<1,dim>> basis_fcns_faces_RT(fe_RT.dofs_per_cell);

	unsigned int nrows_full = dofs_per_cell_test_vector + GeometryInfo<dim>::faces_per_cell*dofs_per_face_test_scalar;
	unsigned int ncols_full = nrows_full;

	FullMatrix<double> local_matrix_RT(nrows_full, ncols_full);
	Vector<double> local_solution_RT(ncols_full);
	Vector<double> local_RHS_RT(nrows_full);

//	Vector<double> sol_all_faces_one_cell(GeometryInfo<dim>::faces_per_cell*dofs_per_face_test_scalar);

	std::vector<types::global_dof_index> local_dof_indices_DG(dofs_per_cell_DG);
	std::vector<types::global_dof_index> local_dof_indices_RT(dofs_per_cell_RT);

	typename DoFHandler<dim>::active_cell_iterator
		  cell_DG = dof_handler.begin_active(),
		  endc = dof_handler.end(), cell_RT = dof_handler_RT.begin_active(),
		  cell_scalar = dof_handler_test_scalar.begin_active(),
		  cell_vector = dof_handler_test_vector.begin_active();

    const unsigned int n_q_points_DG = fe_values_DG.get_quadrature().size();
    const unsigned int n_q_points_RT = fe_values_RT.get_quadrature().size();
	const unsigned int n_q_points_face_DG = fe_face_values_DG.get_quadrature().size();
	const unsigned int n_q_points_face_test_scalar = fe_face_values_test_scalar.get_quadrature().size();
	const unsigned int n_q_points_cell_test_vector = fe_values_test_vector.get_quadrature().size();
	const unsigned int n_q_points_face_test_vector = fe_face_values_test_vector.get_quadrature().size();

	PETScWrappers::MPI::Vector temp_pl_solution;

	PETScWrappers::MPI::Vector temp_Sa_solution_n;
	PETScWrappers::MPI::Vector temp_Sa_solution_nminus1;

	PETScWrappers::MPI::Vector temp_Sv_solution_n;
	PETScWrappers::MPI::Vector temp_Sv_solution_nminus1;

	PETScWrappers::MPI::Vector temp_kappa;

	temp_pl_solution.reinit(locally_owned_dofs_DG,
							locally_relevant_dofs_DG,
							mpi_communicator);

	temp_Sa_solution_n.reinit(locally_owned_dofs_DG,
							  locally_relevant_dofs_DG,
							  mpi_communicator);

	temp_Sa_solution_nminus1.reinit(locally_owned_dofs_DG,
									locally_relevant_dofs_DG,
									mpi_communicator);

	temp_Sv_solution_n.reinit(locally_owned_dofs_DG,
							  locally_relevant_dofs_DG,
							  mpi_communicator);

	temp_Sv_solution_nminus1.reinit(locally_owned_dofs_DG,
									locally_relevant_dofs_DG,
									mpi_communicator);

	temp_kappa.reinit(locally_owned_dofs_dg0,
					  locally_relevant_dofs_dg0,
					  mpi_communicator);

	temp_pl_solution = pl_solution;

	temp_Sa_solution_n = Sa_solution_n;
	temp_Sa_solution_nminus1 = Sa_solution_nminus1;

	temp_Sv_solution_n = Sv_solution_n;
	temp_Sv_solution_nminus1 = Sv_solution_nminus1;

	temp_kappa = kappa_abs_vec;

//	std::vecor<int> face_done(GeometryInfo<dim>::)
	for (; cell_DG != endc; ++cell_DG, ++cell_RT, ++cell_scalar, ++cell_vector)
	{
		if (cell_DG->subdomain_id() == this_mpi_process)
		{
			fe_values_DG.reinit(cell_DG);
			fe_values_RT.reinit(cell_RT);
			fe_values_test_vector.reinit(cell_vector);

			const auto &q_points = fe_values_test_vector.get_quadrature_points();

			local_matrix_RT = 0.0;
			local_RHS_RT = 0.0;
			local_solution_RT = 0.0;

			std::vector<double> pl_vals_cell(n_q_points_DG);
			fe_values_DG.get_function_values(temp_pl_solution, pl_vals_cell);

			std::vector<double> Sa_vals_cell_n(n_q_points_DG);
			fe_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_cell_n);

			std::vector<double> Sa_vals_cell_nminus1(n_q_points_DG);
			fe_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_cell_nminus1);

			std::vector<double> Sv_vals_cell_n(n_q_points_DG);
			fe_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_cell_n);

			std::vector<double> Sv_vals_cell_nminus1(n_q_points_DG);
			fe_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_cell_nminus1);

			std::vector<Tensor<1, dim>> grad_pl_cell(n_q_points_DG);
			fe_values_DG.get_function_gradients(temp_pl_solution, grad_pl_cell);

			double kappa = temp_kappa[cell_DG->global_active_cell_index()];

			for (unsigned int q = 0; q < n_q_points_cell_test_vector; ++q)
			{
				Tensor<1,dim> neg_pl_grad = -grad_pl_cell[q];

				double pl_value = pl_vals_cell[q];
				double Sa_value_n = Sa_vals_cell_n[q];
				double Sa_value_nminus1 = Sa_vals_cell_nminus1[q];
				double Sv_value_n = Sv_vals_cell_n[q];
				double Sv_value_nminus1 = Sv_vals_cell_nminus1[q];

				if(use_exact_pl_in_RT0)
				{
					pl_fcn.set_time(time);
					pl_value = pl_fcn.value(q_points[q]);

					neg_pl_grad = pl_fcn.gradient(q_points[q]);
					neg_pl_grad *= -1.0;
				}

				if(use_exact_Sa_in_RT0)
				{
					Sa_fcn.set_time(time - time_step);
					Sa_value_n = Sa_fcn.value(q_points[q]);

					Sa_fcn.set_time(time - 2.0*time_step);
					Sa_value_nminus1 = Sa_fcn.value(q_points[q]);
				}

				if(use_exact_Sv_in_RT0)
				{
					Sv_fcn.set_time(time - time_step);
					Sv_value_n = Sv_fcn.value(q_points[q]);

					Sv_fcn.set_time(time - 2.0*time_step);
					Sv_value_nminus1 = Sv_fcn.value(q_points[q]);
				}

				double Sa_nplus1_extrapolation = Sa_value_n;
				double Sv_nplus1_extrapolation = Sv_value_n;

				if(second_order_extrapolation)
				{
					Sa_nplus1_extrapolation *= 2.0;
					Sa_nplus1_extrapolation -= Sa_value_nminus1;

					Sv_nplus1_extrapolation *= 2.0;
					Sv_nplus1_extrapolation -= Sv_value_nminus1;

				}

				double rho_l = rho_l_fcn.value(pl_value);
				double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double rho_a = rho_a_fcn.value(pl_value);

				double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

				double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

				if(incompressible)
					rholambda_t = lambda_l + lambda_v + lambda_a;

				if(project_only_kappa)
					rholambda_t = 1.0;

				Tensor<1,dim> g_val = gravity_fcn.vector_value(q_points[q]);
				double density_g;

				if(use_Sa)
					density_g = rho_a;
				else
					density_g = rho_v;

				g_val *= density_g;

				for(unsigned int i = 0; i < dofs_per_cell_test_vector; ++i)
				{
					for(unsigned int j = 0; j < dofs_per_cell_RT; ++j)
						local_matrix_RT(i, j) +=
								(fe_values_test_vector.shape_value_component(i, q, 1)
								 * fe_values_RT.shape_value_component(j, q, 0)
								 + fe_values_test_vector.shape_value_component(i, q, 0)
								 * fe_values_RT.shape_value_component(j, q, 1))
								* fe_values_test_vector.JxW(q);

					local_RHS_RT[i] += kappa
									   *rholambda_t
									   * (neg_pl_grad[0]
											* fe_values_test_vector.shape_value_component(i, q, 1)
											+ neg_pl_grad[1]
											* fe_values_test_vector.shape_value_component(i, q, 0))
									   * fe_values_test_vector.JxW(q);

					local_RHS_RT[i] += kappa
									   *rholambda_t
									   * (g_val[0]
											* fe_values_test_vector.shape_value_component(i, q, 1)
											+ g_val[1]
											* fe_values_test_vector.shape_value_component(i, q, 0))
									   * fe_values_test_vector.JxW(q);
				}
			}

			for (const auto &face : cell_vector->face_iterators())
			{
				unsigned int face_num = cell_scalar->face_iterator_to_index(face);

				fe_face_values_DG.reinit(cell_DG, face);
				fe_face_values_RT.reinit(cell_RT, face);
				fe_face_values_test_scalar.reinit(cell_scalar, face);
				fe_face_values_test_vector.reinit(cell_vector, face);

				const auto &q_points_face = fe_face_values_test_scalar.get_quadrature_points();

				std::vector<double> g(n_q_points_face_test_scalar);
				boundary_function.value_list(fe_face_values_test_scalar.get_quadrature_points(), g);

				for (unsigned int q = 0; q < n_q_points_face_test_scalar; ++q)
				{
					const Tensor<1, dim> normal = fe_face_values_test_scalar.normal_vector(q);

					for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
						basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

					for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
					{
						const unsigned int ii = fe_support_on_face_scalar[face_num][i];

						for (unsigned int j = 0; j < dofs_per_cell_RT; ++j)
						{
							local_matrix_RT(dofs_per_cell_test_vector + ii, j) +=
									fe_face_values_RT[velocities].value(j, q)
									* normal
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);
						}
					}
				}

				if(face->at_boundary())
				{
					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					std::vector<double> Sa_vals_face_n(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_face_n);

					std::vector<double> Sa_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1);

					std::vector<double> Sv_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_face);

					std::vector<double> Sv_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1);

					std::vector<Tensor<1, dim>> grad_pl_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_gradients(temp_pl_solution, grad_pl_face);

					double kappa = temp_kappa[cell_DG->global_active_cell_index()];

					for (unsigned int q = 0; q < n_q_points_face_test_vector; ++q)
					{
						double pl_value = pl_vals_face[q];
						Tensor<1,dim> neg_pl_grad = grad_pl_face[q];
						neg_pl_grad *= -1.0;

						double Sa_value_n = Sa_vals_face_n[q];
						double Sa_value_nminus1 = Sa_vals_face_nminus1[q];
						double Sv_value_n = Sv_vals_face[q];
						double Sv_value_nminus1 = Sv_vals_face_nminus1[q];

						if(use_exact_pl_in_RT0)
						{
							pl_fcn.set_time(time);
							pl_value = pl_fcn.value(q_points_face[q]);
							neg_pl_grad = pl_fcn.gradient(q_points_face[q]);
							neg_pl_grad *= -1.0;
						}

						if(use_exact_Sa_in_RT0)
						{
							Sa_fcn.set_time(time - time_step);
							Sa_value_n = Sa_fcn.value(q_points_face[q]);

							Sa_fcn.set_time(time - 2.0*time_step);
							Sa_value_nminus1 = Sa_fcn.value(q_points_face[q]);
						}

						if(use_exact_Sv_in_RT0)
						{
							Sv_fcn.set_time(time - time_step);
							Sv_value_n = Sv_fcn.value(q_points_face[q]);

							Sv_fcn.set_time(time - 2.0*time_step);
							Sv_value_nminus1 = Sv_fcn.value(q_points_face[q]);
						}

						double Sa_nplus1_extrapolation = Sa_value_n;
						double Sv_nplus1_extrapolation = Sv_value_n;

						if(second_order_extrapolation)
						{
							Sa_nplus1_extrapolation *= 2.0;
							Sa_nplus1_extrapolation -= Sa_value_nminus1;

							Sv_nplus1_extrapolation *= 2.0;
							Sv_nplus1_extrapolation -= Sv_value_nminus1;

						}

						double rho_l = rho_l_fcn.value(pl_value);
						double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double rho_a = rho_a_fcn.value(pl_value);

						double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
						double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

						double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

						if(incompressible)
							rholambda_t = lambda_l + lambda_v + lambda_a;

						if(project_only_kappa)
							rholambda_t = 1.0;

						Tensor<1,dim> g_val = gravity_fcn.vector_value(q_points[q]);
						double density_g;

						if(use_Sa)
							density_g = rho_a;
						else
							density_g = rho_v;

						g_val *= density_g;

						const Tensor<1, dim> normal = fe_face_values_test_vector.normal_vector(q);

						// Figure out if this face is Dirichlet or Neumann
						bool dirichlet = false;

						for(unsigned int i = 0; i < dirichlet_id_pl.size(); i++)
						{
							if(face->boundary_id() == dirichlet_id_pl[i])
							{
								dirichlet = true;
								break;
							}
						}

						if(dirichlet)
						{
							for(unsigned int i = 0; i < dofs_per_cell_test_vector; ++i)
								local_RHS_RT[i] += theta_pl
												  * rholambda_t
												  * kappa
												  * (pl_value - g[q])
												  * (fe_face_values_test_vector.shape_value_component(i, q, 1)
													 * normal[0]
													 + fe_values_test_vector.shape_value_component(i, q, 0)
													 * normal[1])
												  * fe_face_values_test_vector.JxW(q);
						}

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						double gamma_ch_e = rholambda_t*kappa;
						double h_e = cell_DG->face(face_num)->measure();
						double penalty_factor = (penalty_pl_bdry/h_e) * gamma_ch_e * degree*(degree + dim - 1);

						if(dirichlet)
						{
							for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
							{
								const unsigned int ii = fe_support_on_face_scalar[face_num][i];

								local_RHS_RT[dofs_per_cell_test_vector + ii] +=
										rholambda_t
										* kappa
										* neg_pl_grad
										* normal
										* basis_fcns_faces_scalar[i]
										* fe_face_values_test_scalar.JxW(q);

								local_RHS_RT[dofs_per_cell_test_vector + ii] +=
										rholambda_t
										* kappa
										* g_val
										* normal
										* basis_fcns_faces_scalar[i]
										* fe_face_values_test_scalar.JxW(q);

								local_RHS_RT[dofs_per_cell_test_vector + ii] +=
										(penalty_pl_bdry/cell_DG->face(face_num)->measure())
										* gamma_ch_e
										* (pl_value - g[q])
										* basis_fcns_faces_scalar[i]
										* fe_face_values_test_scalar.JxW(q);
							}
						}
					}
				}
				else // interior faces
				{
					typename DoFHandler<dim>::active_cell_iterator cell_DG_neighbor = cell_DG->neighbor(face_num);
					unsigned int neighbor_index = cell_DG->neighbor_index(face_num);

					fe_face_values_DG_neighbor.reinit(cell_DG_neighbor, face);

					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					std::vector<double> pl_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_pl_solution, pl_vals_face_neighbor);

					std::vector<double> Sa_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_n, Sa_vals_face);

					std::vector<double> Sa_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sa_solution_n, Sa_vals_face_neighbor);

					std::vector<double> Sa_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1);

					std::vector<double> Sa_vals_face_nminus1_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sa_solution_nminus1, Sa_vals_face_nminus1_neighbor);

					std::vector<double> Sv_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_n, Sv_vals_face);

					std::vector<double> Sv_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sv_solution_n, Sv_vals_face_neighbor);

					std::vector<double> Sv_vals_face_nminus1(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1);

					std::vector<double> Sv_vals_face_nminus1_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_Sv_solution_nminus1, Sv_vals_face_nminus1_neighbor);

					std::vector<Tensor<1, dim>> grad_pl_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_gradients(temp_pl_solution, grad_pl_face);

					std::vector<Tensor<1, dim>> grad_pl_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_gradients(temp_pl_solution, grad_pl_face_neighbor);

					double kappa0 = temp_kappa[cell_DG->global_active_cell_index()];
					double kappa1 = temp_kappa[cell_DG_neighbor->global_active_cell_index()];

					for (unsigned int q = 0; q < n_q_points_face_test_vector; ++q)
					{
						double pl_value0 = pl_vals_face[q];
						double pl_value1 = pl_vals_face_neighbor[q];

						double Sa_value0_n = Sa_vals_face[q];
						double Sa_value1_n = Sa_vals_face_neighbor[q];
						double Sa_value0_nminus1 = Sa_vals_face_nminus1[q];
						double Sa_value1_nminus1 = Sa_vals_face_nminus1_neighbor[q];

						double Sv_value0_n = Sv_vals_face[q];
						double Sv_value1_n = Sv_vals_face_neighbor[q];
						double Sv_value0_nminus1 = Sv_vals_face_nminus1[q];
						double Sv_value1_nminus1 = Sv_vals_face_nminus1_neighbor[q];

						Tensor<1,dim> neg_pl_grad = -grad_pl_face[q];
						Tensor<1,dim> neg_pl_grad_neighbor = -grad_pl_face_neighbor[q];

						if(use_exact_pl_in_RT0)
						{
							pl_fcn.set_time(time);
							pl_value0 = pl_fcn.value(q_points_face[q]);
							pl_value1 = pl_value0;

							neg_pl_grad = pl_fcn.gradient(q_points_face[q]);
							neg_pl_grad *= -1.0;

							neg_pl_grad_neighbor = neg_pl_grad;
						}

						if(use_exact_Sa_in_RT0)
						{
							Sa_fcn.set_time(time - time_step);
							Sa_value0_n = Sa_fcn.value(q_points_face[q]);
							Sa_value1_n = Sa_value0_n;

							Sa_fcn.set_time(time - 2.0*time_step);
							Sa_value0_nminus1 = Sa_fcn.value(q_points_face[q]);

							Sa_value1_nminus1 = Sa_value0_nminus1;
						}

						if(use_exact_Sv_in_RT0)
						{
							Sv_fcn.set_time(time - time_step);
							Sv_value0_n = Sv_fcn.value(q_points_face[q]);
							Sv_value1_n = Sv_value0_n;

							Sv_fcn.set_time(time - 2.0*time_step);
							Sv_value0_nminus1 = Sv_fcn.value(q_points_face[q]);
							Sv_value1_nminus1 = Sv_value0_nminus1;
						}

						double Sa_nplus1_extrapolation0 = Sa_value0_n;
						double Sa_nplus1_extrapolation1 = Sa_value1_n;
						double Sv_nplus1_extrapolation0 = Sv_value0_n;
						double Sv_nplus1_extrapolation1 = Sv_value1_n;

						if(second_order_extrapolation)
						{
							Sa_nplus1_extrapolation0 *= 2.0;
							Sa_nplus1_extrapolation0 -= Sa_value0_nminus1;

							Sa_nplus1_extrapolation1 *= 2.0;
							Sa_nplus1_extrapolation1 -= Sa_value1_nminus1;

							Sv_nplus1_extrapolation0 *= 2.0;
							Sv_nplus1_extrapolation0 -= Sv_value0_nminus1;

							Sv_nplus1_extrapolation1 *= 2.0;
							Sv_nplus1_extrapolation1 -= Sv_value1_nminus1;
						}

						double rho_l0 = rho_l_fcn.value(pl_value0);
						double rho_l1 = rho_l_fcn.value(pl_value1);

						double rho_v0 = rho_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double rho_v1 = rho_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

						double rho_a0 = rho_a_fcn.value(pl_value0);
						double rho_a1 = rho_a_fcn.value(pl_value1);

						double lambda_l0 = lambda_l_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double lambda_v0 = lambda_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
						double lambda_a0 = lambda_a_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);

						double lambda_l1 = lambda_l_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
						double lambda_v1 = lambda_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
						double lambda_a1 = lambda_a_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

						double rholambda_t0 = rho_l0*lambda_l0 + rho_v0*lambda_v0 + rho_a0*lambda_a0;
						double rholambda_t1 = rho_l1*lambda_l1 + rho_v1*lambda_v1 + rho_a1*lambda_a1;

						if(incompressible)
						{
							rholambda_t0 = lambda_l0 + lambda_v0 + lambda_a0;
							rholambda_t1 = lambda_l1 + lambda_v1 + lambda_a1;
						}

						if(project_only_kappa)
						{
							rholambda_t0 = 1.0;
							rholambda_t1 = 1.0;
						}

						Tensor<1,dim> g_val = gravity_fcn.vector_value(q_points[q]);

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						double coef0 = rholambda_t0*kappa0;
						double coef1 = rholambda_t1*kappa1;

						double weight0 = 0.0;
						double weight1 = 0.0;

						if(fabs(coef0) > 1.e-14 || fabs(coef1) > 1.e-14)
						{
							weight0 = coef1/(coef0 + coef1);
							weight1 = coef0/(coef0 + coef1);
						}

						const Tensor<1, dim> normal = fe_face_values_test_vector.normal_vector(q);

						for(unsigned int i = 0; i < dofs_per_cell_test_vector; ++i)
							local_RHS_RT[i] += theta_pl
											  * weight0
											  * rholambda_t0
											  * kappa0
											  * (pl_value0 - pl_value1)
											  * (fe_face_values_test_vector.shape_value_component(i, q, 1)
												 * normal[0]
												 + fe_values_test_vector.shape_value_component(i, q, 0)
												 * normal[1])
											  * fe_face_values_test_vector.JxW(q);

						for (unsigned int k = 0; k < fe_support_on_face_scalar[face_num].size(); ++k)
							basis_fcns_faces_scalar[k] = fe_face_values_test_scalar.shape_value(fe_support_on_face_scalar[face_num][k], q);

						double weighted_aver_rhs = AverageGradOperators::weighted_average_rhs<dim>(normal,
									neg_pl_grad, neg_pl_grad_neighbor,
									coef0, coef1,
									weight0, weight1);

						double density_g0;
						double density_g1;

						if(use_Sa)
						{
							density_g0 = rho_a0;
							density_g1 = rho_a1;
						}
						else
						{
							density_g0 = rho_v0;
							density_g1 = rho_v1;
						}

						double coef0_g = rholambda_t0*kappa0*density_g0;
						double coef1_g = rholambda_t1*kappa1*density_g1;

						weight0 = 0.0;
						weight1 = 0.0;

						if(fabs(coef0_g) > 1.e-14 || fabs(coef1_g) > 1.e-14)
						{
							weight0 = coef1_g/(coef0_g + coef1_g);
							weight1 = coef0_g/(coef0_g + coef1_g);
						}

						double weighted_aver_rhs_gravity = AverageGradOperators::weighted_average_rhs<dim>(normal,
									g_val, g_val,
									coef0_g, coef1_g,
									weight0, weight1);

						for(unsigned int i = 0; i < dofs_per_face_test_scalar; ++i)
						{
							const unsigned int ii = fe_support_on_face_scalar[face_num][i];

							local_RHS_RT[dofs_per_cell_test_vector + ii] +=
									(weighted_aver_rhs + weighted_aver_rhs_gravity)
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);

							double gamma_ch_e = 2.0*coef0*coef1/(coef0 + coef1 + 1.e-20);
				            double h_e = cell_DG->face(face_num)->measure();
				            double penalty_factor = (penalty_pl/h_e) * gamma_ch_e * degree*(degree + dim - 1);

							local_RHS_RT[dofs_per_cell_test_vector + ii] +=
									penalty_factor
									* (pl_value0 - pl_value1)
									* basis_fcns_faces_scalar[i]
									* fe_face_values_test_scalar.JxW(q);
						}
					}
				}

			}

			FullMatrix<double> inv_local_matrix_RT(nrows_full, ncols_full);
			inv_local_matrix_RT.invert(local_matrix_RT);

			Vector<double> sol_local_RT(nrows_full);

			inv_local_matrix_RT.vmult(sol_local_RT, local_RHS_RT);

			std::vector<unsigned int> dof_ind(dofs_per_cell_RT);
			cell_RT->get_dof_indices(dof_ind);

			for(unsigned int k = 0; k < dofs_per_cell_RT; k++)
				totalDarcyvelocity_RT[dof_ind[k]] = sol_local_RT[k];
		}

		//totalDarcyvelocity_RT.compress(VectorOperation::insert);
	}
    totalDarcyvelocity_RT.compress(VectorOperation::insert);
	return totalDarcyvelocity_RT;
}



} // namespace RTProjection


#endif // RT_PROJECTION_HH
