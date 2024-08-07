#ifndef PL_PROBLEM_MID_HH
#define PL_PROBLEM_MID_HH

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/distributed/shared_tria.h>
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
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/fe/fe_face.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/base/parameter_handler.h>

#include "AverageGradientOperators.hh"

// PETSc stuff
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <iostream>
#include <fstream>
#include <algorithm>

namespace LiquidPressureMidpoint
{
using namespace dealii;


struct CopyDataFace
{
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> joint_dof_indices;
    std::array<unsigned int, 2>          cell_indices;
    std::array<double, 2>                values;
};



struct CopyData
{
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;
    double                               value;
    unsigned int                         cell_index;

    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_rhs.reinit(dofs_per_cell);

        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
    }
};

template <int dim>
class LiquidPressureProblem_midpoint
{
public:
	LiquidPressureProblem_midpoint(Triangulation<dim, dim> &triangulation_,
			const unsigned int degree_, double time_step_, double theta_n_time_, double theta_pl_, double penalty_pl_,
			double penalty_pl_bdry_, std::vector<unsigned int> dirichlet_id_pl_, bool use_exact_Sa_in_pl_,
			bool use_exact_Sv_in_pl_, double time_, unsigned int timestep_number_,
			bool second_order_time_derivative_, bool second_order_extrapolation_,
			bool use_direct_solver_, bool implicit_time_pl_,
			PETScWrappers::MPI::Vector pl_solution_k_, PETScWrappers::MPI::Vector pl_solution_n_,
			PETScWrappers::MPI::Vector Sa_solution_k_,
			PETScWrappers::MPI::Vector Sa_solution_n_,
			PETScWrappers::MPI::Vector Sv_solution_k_,
			PETScWrappers::MPI::Vector Sv_solution_n_,
			PETScWrappers::MPI::Vector kappa_abs_vec_,
			MPI_Comm mpi_communicator_, const unsigned int n_mpi_processes_, const unsigned int this_mpi_process_);

	void assemble_system_matrix_pressure();
	void solve_pressure();

	PETScWrappers::MPI::Vector pl_solution;
private:
    void setup_system();

    parallel::shared::Triangulation<dim> triangulation;
    const MappingQ1<dim> mapping;

    using ScratchData = MeshWorker::ScratchData<dim>;
    const QGauss<dim>     quadrature;
    const QGauss<dim - 1> face_quadrature;
    // Furthermore we want to use DG elements.
    FE_DGQ<dim>     fe;
    DoFHandler<dim> dof_handler;
    const unsigned int degree;

    MPI_Comm mpi_communicator;

	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;

	ConditionalOStream pcout;

	IndexSet locally_owned_dofs;
	IndexSet locally_relevant_dofs;

    SparsityPattern      sparsity_pattern;

    PETScWrappers::MPI::SparseMatrix system_matrix_pressure;
    PETScWrappers::MPI::Vector right_hand_side_pressure;

    PETScWrappers::MPI::Vector pl_solution_k;
    PETScWrappers::MPI::Vector pl_solution_n;

    PETScWrappers::MPI::Vector Sa_solution_k;
    PETScWrappers::MPI::Vector Sa_solution_n;

    PETScWrappers::MPI::Vector Sv_solution_k;
    PETScWrappers::MPI::Vector Sv_solution_n;

    FE_DGQ<dim> fe_dg0;
	DoFHandler<dim> dof_handler_dg0;
	IndexSet locally_owned_dofs_dg0;
	IndexSet locally_relevant_dofs_dg0;
    PETScWrappers::MPI::Vector kappa_abs_vec;

    double 		 time_step;
    double       time;
    unsigned int timestep_number;
    double       theta_n_time;

    double penalty_pl;
    double penalty_pl_bdry;

    double theta_pl;

    std::vector<unsigned int> dirichlet_id_pl;

    bool second_order_time_derivative;
    bool second_order_extrapolation;
    bool implicit_time_pl;

    bool use_direct_solver;

    bool use_exact_Sa_in_pl;
    bool use_exact_Sv_in_pl;

    AffineConstraints<double> constraints;

};


template <int dim>
LiquidPressureProblem_midpoint<dim>::LiquidPressureProblem_midpoint(Triangulation<dim, dim> &triangulation_,
		const unsigned int degree_, double time_step_, double theta_n_time_,
		double theta_pl_, double penalty_pl_,
		double penalty_pl_bdry_, std::vector<unsigned int> dirichlet_id_pl_, bool use_exact_Sa_in_pl_,
		bool use_exact_Sv_in_pl_, double time_, unsigned int timestep_number_,
		bool second_order_time_derivative_, bool second_order_extrapolation_,
		bool use_direct_solver_, bool implicit_time_pl_,
		PETScWrappers::MPI::Vector pl_solution_k_, PETScWrappers::MPI::Vector pl_solution_n_,
		PETScWrappers::MPI::Vector Sa_solution_k_, PETScWrappers::MPI::Vector Sa_solution_n_,
		PETScWrappers::MPI::Vector Sv_solution_k_, PETScWrappers::MPI::Vector Sv_solution_n_,
		PETScWrappers::MPI::Vector kappa_abs_vec_,
		MPI_Comm mpi_communicator_, const unsigned int n_mpi_processes_, const unsigned int this_mpi_process_)
	: triangulation(MPI_COMM_WORLD)
	, mapping()
	, degree(degree_)
	, fe(degree_)
	, quadrature(degree_ + 1)
  	, face_quadrature(degree_ + 1)
	, time_step(time_step_)
	, theta_n_time(theta_n_time_)
	, theta_pl(theta_pl_)
	, penalty_pl(penalty_pl_)
	, penalty_pl_bdry(penalty_pl_bdry_)
	, dirichlet_id_pl(dirichlet_id_pl_)
	, use_exact_Sa_in_pl(use_exact_Sa_in_pl_)
	, use_exact_Sv_in_pl(use_exact_Sv_in_pl_)
	, time(time_)
	, timestep_number(timestep_number_)
	, second_order_time_derivative(second_order_time_derivative_)
	, second_order_extrapolation(second_order_extrapolation_)
	, implicit_time_pl(implicit_time_pl_)
	, use_direct_solver(use_direct_solver_)
	, pl_solution_k(pl_solution_k_)
	, pl_solution_n(pl_solution_n_)
	, Sa_solution_k(Sa_solution_k_)
	, Sa_solution_n(Sa_solution_n_)
	, Sv_solution_k(Sv_solution_k_)
	, Sv_solution_n(Sv_solution_n_)
	, kappa_abs_vec(kappa_abs_vec_)
	, dof_handler(triangulation)
	, fe_dg0(0)
	, dof_handler_dg0(triangulation)
	, mpi_communicator(mpi_communicator_)
	, n_mpi_processes(n_mpi_processes_)
    , this_mpi_process(this_mpi_process_)
    , pcout(std::cout, (this_mpi_process == 0))
{
	triangulation.copy_triangulation(triangulation_);
}

template <int dim>
void LiquidPressureProblem_midpoint<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);

    constraints.clear();
	constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

	const std::vector<IndexSet> locally_owned_dofs_per_proc =
		  DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
	locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    system_matrix_pressure.reinit(locally_owned_dofs,
			  	  	  	  	  	  locally_owned_dofs,
								  sparsity_pattern,
								  mpi_communicator);

    pl_solution.reinit(locally_owned_dofs, mpi_communicator);
    right_hand_side_pressure.reinit(locally_owned_dofs, mpi_communicator);

    dof_handler_dg0.distribute_dofs(fe_dg0);
	const std::vector<IndexSet> locally_owned_dofs_per_proc_dg0 =
			DoFTools::locally_owned_dofs_per_subdomain(dof_handler_dg0);
	locally_owned_dofs_dg0 = locally_owned_dofs_per_proc_dg0[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_dg0, locally_relevant_dofs_dg0);

}

template <int dim>
void LiquidPressureProblem_midpoint<dim>::assemble_system_matrix_pressure()
{
	setup_system();

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    BoundaryValuesLiquidPressure<dim> boundary_function;
    RightHandSideLiquidPressure<dim> right_hand_side_fcn;
    GravitySourceTerm<dim> gravity_fcn;

    // Liquid Pressure
    ExactLiquidPressure<dim> pl_fcn;

    // Saturations
    ExactAqueousSaturation<dim> Sa_fcn;

    // Vapor saturation
    ExactVaporSaturation<dim> Sv_fcn;

    // Porosity
    porosity<dim> porosity_fcn;

    // Densities
    rho_l<dim> rho_l_fcn;
    rho_v<dim> rho_v_fcn;
    rho_a<dim> rho_a_fcn;

    // Mobilities
    lambda_l<dim> lambda_l_fcn;
    lambda_v<dim> lambda_v_fcn;
    lambda_a<dim> lambda_a_fcn;

    // Capillary pressures
    CapillaryPressurePca<dim> cap_p_pca_fcn;
    CapillaryPressurePcv<dim> cap_p_pcv_fcn;

    // Neumann term
    NeumannTermLiquidPressure<dim> neumann_fcn;

	// Solutions on this processor
    PETScWrappers::MPI::Vector temp_pl_solution_k;
	PETScWrappers::MPI::Vector temp_pl_solution_n;

	PETScWrappers::MPI::Vector temp_Sa_solution_k;
	PETScWrappers::MPI::Vector temp_Sa_solution_n;

	PETScWrappers::MPI::Vector temp_Sv_solution_k;
	PETScWrappers::MPI::Vector temp_Sv_solution_n;

	PETScWrappers::MPI::Vector temp_kappa;

	temp_pl_solution_k.reinit(locally_owned_dofs,
							  locally_relevant_dofs,
							  mpi_communicator);

	temp_pl_solution_n.reinit(locally_owned_dofs,
							  locally_relevant_dofs,
							  mpi_communicator);

	temp_Sa_solution_k.reinit(locally_owned_dofs,
							  locally_relevant_dofs,
							  mpi_communicator);

	temp_Sa_solution_n.reinit(locally_owned_dofs,
							  locally_relevant_dofs,
							  mpi_communicator);

	temp_Sv_solution_k.reinit(locally_owned_dofs,
							  locally_relevant_dofs,
							  mpi_communicator);

	temp_Sv_solution_n.reinit(locally_owned_dofs,
							  locally_relevant_dofs,
							  mpi_communicator);

	temp_kappa.reinit(locally_owned_dofs_dg0,
					  locally_relevant_dofs_dg0,
					  mpi_communicator);

	temp_pl_solution_k = pl_solution_k;
    temp_pl_solution_n = pl_solution_n;

    temp_Sa_solution_k = Sa_solution_k;
    temp_Sa_solution_n = Sa_solution_n;

    temp_Sv_solution_k = Sv_solution_k;
    temp_Sv_solution_n = Sv_solution_n;

	temp_kappa = kappa_abs_vec;

    // This is the function that will be executed for each cell. Contains element integrals
    const auto cell_worker = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
    							 auto &scratch_data,
								 auto & copy_data)
    {
    	const FEValues<dim> &fe_v = scratch_data.reinit(cell);

        const unsigned int n_dofs = fe_v.dofs_per_cell;
        copy_data.reinit(cell, n_dofs);

        const auto &q_points = scratch_data.get_quadrature_points();
        const int n_qpoints = q_points.size();

        const std::vector<double> &JxW  = scratch_data.get_JxW_values();

        std::vector<double>         rhs_values(n_qpoints);
//        right_hand_side_fcn.set_time(time);
        right_hand_side_fcn.set_time(time);
        right_hand_side_fcn.value_list(q_points, rhs_values);

        gravity_fcn.set_time(time);

        std::vector<double> old_pl_vals_k(n_qpoints);
        std::vector<double> old_pl_vals_n(n_qpoints);

        std::vector<double> old_Sa_vals_k(n_qpoints);
        std::vector<Tensor<1, dim>> old_Sa_grads_k(n_qpoints);

        std::vector<double> old_Sa_vals_n(n_qpoints);

        std::vector<double> old_Sv_vals_k(n_qpoints);
        std::vector<Tensor<1, dim>> old_Sv_grads_k(n_qpoints);

        std::vector<double> old_Sv_vals_n(n_qpoints);

        // Obtain values of previous time solutions at integration points
        fe_v.get_function_values(temp_pl_solution_k, old_pl_vals_k);
        fe_v.get_function_values(temp_pl_solution_n, old_pl_vals_n);

        fe_v.get_function_values(temp_Sa_solution_k, old_Sa_vals_k);
        fe_v.get_function_gradients(temp_Sa_solution_k, old_Sa_grads_k);
        fe_v.get_function_values(temp_Sa_solution_n, old_Sa_vals_n);

        fe_v.get_function_values(temp_Sv_solution_k, old_Sv_vals_k);
        fe_v.get_function_gradients(temp_Sv_solution_k, old_Sv_grads_k);
        fe_v.get_function_values(temp_Sv_solution_n, old_Sv_vals_n);

        double kappa = temp_kappa[cell->active_cell_index()];

        for (unsigned int point = 0; point < n_qpoints; ++point)
        {
        	// Get value of pl at current integration point
			double pl_value_k = old_pl_vals_k[point];
			double pl_value_n = old_pl_vals_n[point];

			// Get value of sa at current integration point
			double Sa_value_k = old_Sa_vals_k[point];
			Tensor<1,dim> Sa_grad_k = old_Sa_grads_k[point];
			double Sa_value_n = old_Sa_vals_n[point];

			// Get value of sv at current integration point
			double Sv_value_k = old_Sv_vals_k[point];
			Tensor<1,dim> Sv_grad_k = old_Sv_grads_k[point];
			double Sv_value_n = old_Sv_vals_n[point];

			// Do second order extrapolations if necessary
			double pl_nplus1_extrapolation = pl_value_k;
        	double Sa_nplus1_extrapolation = Sa_value_k;
        	double Sv_nplus1_extrapolation = Sv_value_k;
        	Tensor<1,dim> Sa_grad_nplus1_extrapolation = Sa_grad_k;
        	Tensor<1,dim> Sv_grad_nplus1_extrapolation = Sv_grad_k;

        	// Get coefficient values
        	double phi = porosity_fcn.value(pl_nplus1_extrapolation);
//        	double phi_n = porosity_fcn.value(pl_value_n);
//        	double phi_nminus1 = porosity_fcn.value(pl_value_nminus1);

        	double Sl = ComputeSl<dim>(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
        	double Sl_n = ComputeSl<dim>(pl_value_n, Sa_value_n, Sv_value_n);
//        	double Sl_nminus1 = ComputeSl<dim>(pl_value_nminus1, Sa_value_nminus1, Sv_value_nminus1);
//        	double Sl_nminus2 = ComputeSl<dim>(pl_value_nminus2, Sa_value_nminus2, Sv_value_nminus2);

            double rho_l = rho_l_fcn.value(pl_nplus1_extrapolation);
            double rho_v = rho_v_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
            double rho_a = rho_a_fcn.value(pl_nplus1_extrapolation);

            double rho_l_n = rho_l_fcn.value(pl_value_n);
			double rho_v_n = rho_v_fcn.value(pl_value_n, Sa_value_n, Sv_value_n);
			double rho_a_n = rho_a_fcn.value(pl_value_n);

//            double rho_l_nminus1 = rho_l_fcn.value(pl_value_nminus1);
//			double rho_v_nminus1 = rho_v_fcn.value(pl_value_nminus1, Sa_value_nminus1, Sv_value_nminus1);
//			double rho_a_nminus1 = rho_a_fcn.value(pl_value_nminus1);

//            double rho_l_nminus2 = rho_l_fcn.value(pl_value_nminus2);
//			double rho_v_nminus2 = rho_v_fcn.value(pl_value_nminus2, Sa_value_nminus2, Sv_value_nminus2);
//			double rho_a_nminus2 = rho_a_fcn.value(pl_value_nminus2);

			double rhot = rho_l*Sl + rho_v*Sv_nplus1_extrapolation + rho_a*Sa_nplus1_extrapolation;
			double rhot_n = rho_l_n*Sl_n + rho_v_n*Sv_value_n + rho_a_n*Sa_value_n;
//			double rhot_nminus1 = rho_l_nminus1*Sl_nminus1 + rho_v_nminus1*Sv_value_nminus1 + rho_a_nminus1*Sa_value_nminus1;
//			double rhot_nminus2 = rho_l_nminus2*Sl_nminus2 + rho_v_nminus2*Sv_value_nminus2 + rho_a_nminus2*Sa_value_nminus2;

            double lambda_l = lambda_l_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
            double lambda_v = lambda_v_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
            double lambda_a = lambda_a_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

            double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;
            double lambda_t = lambda_l + lambda_v + lambda_a;

            Tensor<1,dim> pca_grad = cap_p_pca_fcn.num_gradient(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation,
            		Sa_grad_nplus1_extrapolation, Sv_grad_nplus1_extrapolation);
            Tensor<1,dim> pcv_grad = cap_p_pcv_fcn.num_gradient(Sv_nplus1_extrapolation, Sv_grad_nplus1_extrapolation);

            // Coefficients for time terms
            double psi = 0.0;//rhot*porosity_fcn.derivative_wrt_pl(pl_nplus1_extrapolation)
//            		+ phi*(Sl*rho_l_fcn.derivative_wrt_pl(pl_nplus1_extrapolation)
//            				+ Sa_nplus1_extrapolation*rho_a_fcn.derivative_wrt_pl(pl_nplus1_extrapolation)
//            				+ Sv_nplus1_extrapolation*rho_v_fcn.derivative_wrt_pl(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation));

            double nu_a = phi*(rho_l - rho_a);

            double nu_v = phi*(rho_l - rho_v
            		- rho_v_fcn.derivative_wrt_Sv(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation)*Sv_nplus1_extrapolation);

            // These three variables are the terms of the time derivative when it is left as
            // \partial_t(\phi\rho_t) instead of expanding it using chain rule
            double time_term = 0.0;
        	double time_term_n = 0.0;
        	double time_term_nminus1 = 0.0;

            if(timestep_number > 1)
			{
//            	time_term = phi*rhot;
//				time_term_n = phi*rhot_n;
//				time_term_nminus1 = phi_nminus1*rhot_nminus1;
			}

            for (unsigned int i = 0; i < n_dofs; ++i)
            {
                for (unsigned int j = 0; j < n_dofs; ++j)
                {
                	// Diffusion term
					copy_data.cell_matrix(i,j) +=
						lambda_t
						* kappa
						* fe_v.shape_grad(i, point)
						* fe_v.shape_grad(j, point)
						* JxW[point];

					// Time term
//					if(implicit_time_pl)
//					{
//						if(timestep_number == 1 || !second_order_time_derivative)
//							copy_data.cell_matrix(i,j) +=
//									(psi/(theta_n_time*time_step))
//									* fe_v.shape_value(i, point)
//									* fe_v.shape_value(j, point)
//									* JxW[point];
//						else
//							copy_data.cell_matrix(i,j) +=
//									(psi/time_step)
//									* 1.5
//									* fe_v.shape_value(i, point)
//									* fe_v.shape_value(j, point)
//									* JxW[point];
//
//					}
                }

                // Source term
                copy_data.cell_rhs(i) += right_hand_side_fcn.value(q_points[point]) * fe_v.shape_value(i, point) * JxW[point];

                // Time term of pl
//                if(implicit_time_pl)
//                {
//                	if(timestep_number == 1 || !second_order_time_derivative)
//						copy_data.cell_rhs(i) +=
//								(psi/(theta_n_time*time_step))
//								* pl_value_n
//								* fe_v.shape_value(i, point)
//								* JxW[point];
////                	else
////						copy_data.cell_rhs(i) +=
////								(psi/(theta_n_time*time_step))
////								* (2.0*pl_value_n - 0.5*pl_value_nminus1)
////								* fe_v.shape_value(i, point)
////								* JxW[point];
//
//                	if(second_order_time_derivative)
//                	{
//                		// Time term for sa
////                		copy_data.cell_rhs(i) +=
////								(nu_a/time_step)
////								* (2.0*Sa_value_n - 3.0*Sa_value_nminus1 + Sa_value_nminus2)
////								* fe_v.shape_value(i, point)
////								* JxW[point];
////
////                		// Time term for sv
////                		copy_data.cell_rhs(i) +=
////								(nu_v/time_step)
////								* (2.0*Sv_value_n - 3.0*Sv_value_nminus1 + Sv_value_nminus2)
////								* fe_v.shape_value(i, point)
////								* JxW[point];
//                	}
//                	else
//                	{
////                    	copy_data.cell_rhs(i) +=
////                    			(nu_a/time_step)
////    							* (Sa_value_n - Sa_value_nminus1)
////    							* fe_v.shape_value(i, point)
////    							* JxW[point];
////
////                    	copy_data.cell_rhs(i) +=
////                    			(nu_v/time_step)
////    							* (Sv_value_n - Sv_value_nminus1)
////    							* fe_v.shape_value(i, point)
////    							* JxW[point];
//                	}
//
//                }
//                else
//                {
////                	if(second_order_time_derivative)
////                	{
////						copy_data.cell_rhs(i) += -(phi/time_step)*(2.0*rhot_n - 3.0*rhot_nminus1 + rhot_nminus2)
////								* fe_v.shape_value(i, point) * JxW[point];
////                	}
////					else
//                		copy_data.cell_rhs(i) += -(1.0/(theta_n_time*time_step))*(time_term - time_term_n) * fe_v.shape_value(i, point) * JxW[point];
//                }

                // Pca and pcv terms
				copy_data.cell_rhs(i) += - lambda_v * kappa * pcv_grad * fe_v.shape_grad(i, point) * JxW[point];
				copy_data.cell_rhs(i) += lambda_a * kappa * pca_grad * fe_v.shape_grad(i, point) * JxW[point];

                // Gravity term
                copy_data.cell_rhs(i) += kappa*rholambda_t
                		* gravity_fcn.vector_value(q_points[point])
						* fe_v.shape_grad(i, point)
						* JxW[point];

			}
		}
    };

    // This is the function called for boundary faces
    const auto boundary_worker = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                                     const unsigned int &face_no,
									 auto &scratch_data,
									 auto & copy_data)
	{
        const FEFaceValuesBase<dim> &fe_face = scratch_data.reinit(cell, face_no);

        const auto &q_points = scratch_data.get_quadrature_points();
        const int n_qpoints = q_points.size();

        const unsigned int n_facet_dofs = fe_face.dofs_per_cell;
        const std::vector<double> &        JxW     = scratch_data.get_JxW_values();
        const std::vector<Tensor<1, dim>> &normals = scratch_data.get_normal_vectors();

        std::vector<double> g(n_qpoints);
        boundary_function.set_time(time);
        boundary_function.value_list(q_points, g);

        neumann_fcn.set_time(time);

        gravity_fcn.set_time(time);


        std::vector<double> old_pl_vals_k(n_qpoints);

		std::vector<double> old_Sa_vals_k(n_qpoints);
		std::vector<Tensor<1, dim>> old_Sa_grads_k(n_qpoints);

		std::vector<double> old_Sv_vals_k(n_qpoints);
		std::vector<Tensor<1, dim>> old_Sv_grads_k(n_qpoints);

		fe_face.get_function_values(temp_pl_solution_k, old_pl_vals_k);

		fe_face.get_function_values(temp_Sa_solution_k, old_Sa_vals_k);
		fe_face.get_function_gradients(temp_Sa_solution_k, old_Sa_grads_k);

		fe_face.get_function_values(temp_Sv_solution_k, old_Sv_vals_k);
		fe_face.get_function_gradients(temp_Sv_solution_k, old_Sv_grads_k);

        double kappa = temp_kappa[cell->active_cell_index()];

        // Figure out if this face is Dirichlet or Neumann
        bool dirichlet = false;

        for(unsigned int i = 0; i < dirichlet_id_pl.size(); i++)
        {
        	if(cell->face(face_no)->boundary_id() == dirichlet_id_pl[i])
        	{
        		dirichlet = true;
        		break;
        	}
        }

		// Dirichlet part
		if(dirichlet)
		{
			for (unsigned int point = 0; point < n_qpoints; ++point)
			{
				// Get pl, sa and sv at current int point
				double pl_value_k = old_pl_vals_k[point];

				double Sa_value_k = old_Sa_vals_k[point];
				Tensor<1,dim> Sa_grad_k = old_Sa_grads_k[point];

				double Sv_value_k = old_Sv_vals_k[point];
				Tensor<1,dim> Sv_grad_k = old_Sv_grads_k[point];

				// Second order extrapolations if needed
				double pl_nplus1_extrapolation = pl_value_k;
	        	double Sa_nplus1_extrapolation = Sa_value_k;
	        	double Sv_nplus1_extrapolation = Sv_value_k;
	        	Tensor<1,dim> Sa_grad_nplus1_extrapolation = Sa_grad_k;
	        	Tensor<1,dim> Sv_grad_nplus1_extrapolation = Sv_grad_k;

	        	// Coefficients
	            double rho_l = rho_l_fcn.value(pl_nplus1_extrapolation);
	            double rho_v = rho_v_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
	            double rho_a = rho_a_fcn.value(pl_nplus1_extrapolation);

	            double lambda_l = lambda_l_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
	            double lambda_v = lambda_v_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
	            double lambda_a = lambda_a_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

	            double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;
	            double lambda_t = lambda_l + lambda_v + lambda_a;

	            Tensor<1,dim> pca_grad = cap_p_pca_fcn.num_gradient(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation,
	            		Sa_grad_nplus1_extrapolation, Sv_grad_nplus1_extrapolation);
	            Tensor<1,dim> pcv_grad = cap_p_pcv_fcn.num_gradient(Sv_nplus1_extrapolation, Sv_grad_nplus1_extrapolation);

	            // Penalty factors
	            double gamma_pl_e = lambda_t*kappa;
	            double h_e = cell->face(face_no)->measure();
	            double penalty_factor = (penalty_pl_bdry/h_e) * gamma_pl_e * degree*(degree + dim - 1);

				for (unsigned int i = 0; i < n_facet_dofs; ++i)
				{
					for (unsigned int j = 0; j < n_facet_dofs; ++j)
					{
						copy_data.cell_matrix(i, j) +=
								- lambda_t
								* kappa
								* fe_face.shape_value(i, point)
								* fe_face.shape_grad(j, point)
								* normals[point]
								* JxW[point];

						copy_data.cell_matrix(i, j) +=
								theta_pl
								* lambda_t
								* kappa
								* fe_face.shape_value(j, point)
								* fe_face.shape_grad(i, point)
								* normals[point]
								* JxW[point];

						copy_data.cell_matrix(i, j) +=
								penalty_factor
								* fe_face.shape_value(i, point)
								* fe_face.shape_value(j, point)
								* JxW[point];
					}

					copy_data.cell_rhs(i) += penalty_factor
											* fe_face.shape_value(i, point)
											* g[point]
											* JxW[point];

					copy_data.cell_rhs(i) += theta_pl
											* lambda_t
											* kappa
											* fe_face.shape_grad(i, point)
											* normals[point]
											* g[point]
											* JxW[point];

					if(cell->face(face_no)->boundary_id() != 5 && cell->face(face_no)->boundary_id() != 6)
					{
						copy_data.cell_rhs(i) += lambda_v
												* kappa
												* pcv_grad
												* normals[point]
												* fe_face.shape_value(i, point)
												* JxW[point];

						copy_data.cell_rhs(i) -= lambda_a
												* kappa
												* pca_grad
												* normals[point]
												* fe_face.shape_value(i, point)
												* JxW[point];
					}
					copy_data.cell_rhs(i) -= kappa*rholambda_t
					                		* gravity_fcn.vector_value(q_points[point])
											* normals[point]
											* fe_face.shape_value(i, point)
											* JxW[point];

				}
			}
		}
		else // Neumann boundary
		{
			for (unsigned int point = 0; point < n_qpoints; ++point)
			{
				Tensor<1,dim> neumann_term = neumann_fcn.vector_value(q_points[point]);

				// Get pl, sa and sv at current int point
				double pl_value_k = old_pl_vals_k[point];

				double Sa_value_k = old_Sa_vals_k[point];
				Tensor<1,dim> Sa_grad_k = old_Sa_grads_k[point];

				double Sv_value_k = old_Sv_vals_k[point];
				Tensor<1,dim> Sv_grad_k = old_Sv_grads_k[point];

				// Second order extrapolations if needed
				double pl_nplus1_extrapolation = pl_value_k;
	        	double Sa_nplus1_extrapolation = Sa_value_k;
	        	double Sv_nplus1_extrapolation = Sv_value_k;
	        	Tensor<1,dim> Sa_grad_nplus1_extrapolation = Sa_grad_k;
	        	Tensor<1,dim> Sv_grad_nplus1_extrapolation = Sv_grad_k;

				Tensor<1,dim> Sa_grad = old_Sa_grads_k[point];
				Tensor<1,dim> Sv_grad = old_Sv_grads_k[point];

	            double rho_v = rho_v_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
	            double rho_a = rho_a_fcn.value(pl_nplus1_extrapolation);

	            double lambda_v = lambda_v_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double lambda_a = lambda_a_fcn.value(pl_nplus1_extrapolation, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

	            Tensor<1,dim> pca_grad = cap_p_pca_fcn.num_gradient(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation,
	            		Sa_grad_nplus1_extrapolation, Sv_grad_nplus1_extrapolation);
	            Tensor<1,dim> pcv_grad = cap_p_pcv_fcn.num_gradient(Sv_nplus1_extrapolation, Sv_grad_nplus1_extrapolation);

				for (unsigned int i = 0; i < n_facet_dofs; ++i)
				{

//					copy_data.cell_rhs(i) += lambda_v
//											* kappa
//											* pcv_grad
//											* normals[point]
//											* fe_face.shape_value(i, point)
//											* JxW[point];
//
//					copy_data.cell_rhs(i) -= lambda_a
//											* kappa
//											* pca_grad
//											* normals[point]
//											* fe_face.shape_value(i, point)
//											* JxW[point];

					copy_data.cell_rhs(i) += neumann_term
											* normals[point]
											* fe_face.shape_value(i, point)
											* JxW[point];
				}
			}
		}

    };

    // This is the function called on interior faces
    const auto face_worker = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
								 const typename DoFHandler<dim>::active_cell_iterator &ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
								 auto &scratch_data,
								 auto & copy_data)
	{
    	const FEInterfaceValues<dim> &fe_iv = scratch_data.reinit(cell, f, sf, ncell, nf, nsf);

    	const auto &q_points = fe_iv.get_quadrature_points();
        const int n_qpoints = q_points.size();

        const FEFaceValuesBase<dim> &fe_face = fe_iv.get_fe_face_values(0);
        const FEFaceValuesBase<dim> &fe_face_neighbor = fe_iv.get_fe_face_values(1);

        copy_data.face_data.emplace_back();

        CopyDataFace &copy_data_face = copy_data.face_data.back();

        const unsigned int n_dofs        = fe_iv.n_current_interface_dofs();
        copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

        copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);
        copy_data_face.cell_rhs.reinit(n_dofs);

        const std::vector<double> &        JxW     = fe_iv.get_JxW_values();
        const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

        gravity_fcn.set_time(time);

        // Since these are interior faces, we have values of pl, sa, and sv on current and neighboring element
        std::vector<double> old_pl_vals_k(n_qpoints);
        std::vector<double> old_pl_vals_k_neighbor(n_qpoints);

        std::vector<double> old_Sa_vals_k(n_qpoints);
        std::vector<double> old_Sa_vals_k_neighbor(n_qpoints);

        std::vector<Tensor<1, dim>> old_Sa_grads_k(n_qpoints);
        std::vector<Tensor<1, dim>> old_Sa_grads_k_neighbor(n_qpoints);

        std::vector<double> old_Sv_vals_k(n_qpoints);
        std::vector<double> old_Sv_vals_k_neighbor(n_qpoints);

        std::vector<Tensor<1, dim>> old_Sv_grads_k(n_qpoints);
        std::vector<Tensor<1, dim>> old_Sv_grads_k_neighbor(n_qpoints);

        fe_face.get_function_values(temp_pl_solution_k, old_pl_vals_k);
        fe_face_neighbor.get_function_values(temp_pl_solution_k, old_pl_vals_k_neighbor);

        fe_face.get_function_values(temp_Sa_solution_k, old_Sa_vals_k);
        fe_face_neighbor.get_function_values(temp_Sa_solution_k, old_Sa_vals_k_neighbor);

        fe_face.get_function_gradients(temp_Sa_solution_k, old_Sa_grads_k);
		fe_face_neighbor.get_function_gradients(temp_Sa_solution_k, old_Sa_grads_k_neighbor);

        fe_face.get_function_values(temp_Sv_solution_k, old_Sv_vals_k);
        fe_face_neighbor.get_function_values(temp_Sv_solution_k, old_Sv_vals_k_neighbor);

        fe_face.get_function_gradients(temp_Sv_solution_k, old_Sv_grads_k);
		fe_face_neighbor.get_function_gradients(temp_Sv_solution_k, old_Sv_grads_k_neighbor);

        double kappa0 = temp_kappa[cell->active_cell_index()];
        double kappa1 = temp_kappa[ncell->active_cell_index()];

        for (unsigned int point = 0; point < n_qpoints; ++point)
        {
        	// Get pl, sa and sv values on current integration point.
        	// The 0 indicates current element, and 1 indicates neighboring element.
        	double pl_value0_k = old_pl_vals_k[point];
        	double pl_value1_k = old_pl_vals_k_neighbor[point];

			double Sa_value0_k = old_Sa_vals_k[point];
			double Sa_value1_k = old_Sa_vals_k_neighbor[point];

			Tensor<1,dim> Sa_grad0_k = old_Sa_grads_k[point];
			Tensor<1,dim> Sa_grad1_k = old_Sa_grads_k_neighbor[point];

			double Sv_value0_k = old_Sv_vals_k[point];
			double Sv_value1_k = old_Sv_vals_k_neighbor[point];

			Tensor<1,dim> Sv_grad0_k = old_Sv_grads_k[point];
			Tensor<1,dim> Sv_grad1_k = old_Sv_grads_k_neighbor[point];

			// Second order extrapolations if needed
			double pl_nplus1_extrapolation0 = pl_value0_k;
			double pl_nplus1_extrapolation1 = pl_value1_k;
        	double Sa_nplus1_extrapolation0 = Sa_value0_k;
        	double Sa_nplus1_extrapolation1 = Sa_value1_k;
        	double Sv_nplus1_extrapolation0 = Sv_value0_k;
        	double Sv_nplus1_extrapolation1 = Sv_value1_k;
        	Tensor<1,dim> Sa_grad_nplus1_extrapolation0 = Sa_grad0_k;
        	Tensor<1,dim> Sa_grad_nplus1_extrapolation1 = Sa_grad1_k;
        	Tensor<1,dim> Sv_grad_nplus1_extrapolation0 = Sv_grad0_k;
        	Tensor<1,dim> Sv_grad_nplus1_extrapolation1 = Sv_grad1_k;

        	// Coefficients
            double rho_l0 = rho_l_fcn.value(pl_nplus1_extrapolation0);
            double rho_l1 = rho_l_fcn.value(pl_nplus1_extrapolation1);

            double rho_v0 = rho_v_fcn.value(pl_nplus1_extrapolation0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
            double rho_v1 = rho_v_fcn.value(pl_nplus1_extrapolation1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

            double rho_a0 = rho_a_fcn.value(pl_nplus1_extrapolation0);
            double rho_a1 = rho_a_fcn.value(pl_nplus1_extrapolation1);

            double lambda_l0 = lambda_l_fcn.value(pl_nplus1_extrapolation0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
            double lambda_l1 = lambda_l_fcn.value(pl_nplus1_extrapolation1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

			double lambda_v0 = lambda_v_fcn.value(pl_nplus1_extrapolation0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
			double lambda_v1 = lambda_v_fcn.value(pl_nplus1_extrapolation1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

			double lambda_a0 = lambda_a_fcn.value(pl_nplus1_extrapolation0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
			double lambda_a1 = lambda_a_fcn.value(pl_nplus1_extrapolation1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

            double rholambda_t0 = rho_l0*lambda_l0 + rho_v0*lambda_v0 + rho_a0*lambda_a0;
            double rholambda_t1 = rho_l1*lambda_l1 + rho_v1*lambda_v1 + rho_a1*lambda_a1;

            double lambda_t0 = lambda_l0 + lambda_v0 + lambda_a0;
            double lambda_t1 = lambda_l1 + lambda_v1 + lambda_a1;

			Tensor<1,dim> pca_grad0 = cap_p_pca_fcn.num_gradient(Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0,
					Sa_grad_nplus1_extrapolation0, Sv_grad_nplus1_extrapolation0);
			Tensor<1,dim> pca_grad1 = cap_p_pca_fcn.num_gradient(Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1,
					Sa_grad_nplus1_extrapolation1, Sv_grad_nplus1_extrapolation1);

            Tensor<1,dim> pcv_grad0 = cap_p_pcv_fcn.num_gradient(Sv_nplus1_extrapolation0, Sv_grad_nplus1_extrapolation0);
            Tensor<1,dim> pcv_grad1 = cap_p_pcv_fcn.num_gradient(Sv_nplus1_extrapolation1, Sv_grad_nplus1_extrapolation1);

            // Coefficients and weights from diffusion term
            double coef0_diff = lambda_t0*kappa0;
			double coef1_diff = lambda_t1*kappa1;

			double weight0_diff = coef1_diff/(coef0_diff + coef1_diff + 1.e-20);
			double weight1_diff = coef0_diff/(coef0_diff + coef1_diff + 1.e-20);

			// Coefficients and weights from pcv term
			double coef0_pcv = lambda_v0*kappa0;
			double coef1_pcv = lambda_v1*kappa1;

			double weight0_pcv = coef1_pcv/(coef0_pcv + coef1_pcv + 1.e-20);
			double weight1_pcv = coef0_pcv/(coef0_pcv + coef1_pcv + 1.e-20);

			// Coefficients and weights from pca term
			double coef0_pca = lambda_a0*kappa0;
			double coef1_pca = lambda_a1*kappa1;

			double weight0_pca = coef1_pca/(coef0_pca + coef1_pca + 1.e-20);
			double weight1_pca = coef0_pca/(coef0_pca + coef1_pca + 1.e-20);

			// Coefficients and weights from gravity term
			double coef0_g = kappa0*rholambda_t0;
			double coef1_g = kappa1*rholambda_t1;

			double weight0_g = coef1_g/(coef0_g + coef1_g + 1.e-20);
			double weight1_g = coef0_g/(coef0_g + coef1_g + 1.e-20);

            double gamma_pl_e = 2.0*coef0_diff*coef1_diff/(coef0_diff + coef1_diff + 1.e-20);
            double h_e = cell->face(f)->measure();
            double penalty_factor = (penalty_pl/h_e) * gamma_pl_e * degree*(degree + dim - 1);

            for (unsigned int i = 0; i < n_dofs; ++i)
            {
                for (unsigned int j = 0; j < n_dofs; ++j)
                {
                    // Interior face terms from diffusion
                    copy_data_face.cell_matrix(i, j) +=
						penalty_factor
                        * fe_iv.jump(i, point)
                        * fe_iv.jump(j, point)
                        * JxW[point];

                    double weighted_aver_j = AverageGradOperators::weighted_average_gradient(cell, f, sf, ncell, nf,
                            nsf, fe_iv,
							normals[point],
                    		j, point,
							coef0_diff, coef1_diff,
                    		weight0_diff, weight1_diff);

                    copy_data_face.cell_matrix(i, j) +=
                        - fe_iv.jump(i, point)
						* weighted_aver_j
                        * JxW[point];

                    double weighted_aver_i = AverageGradOperators::weighted_average_gradient(cell, f, sf, ncell, nf,
                            nsf, fe_iv,
							normals[point],
                    		i, point,
							coef0_diff, coef1_diff,
                    		weight0_diff, weight1_diff);

                    copy_data_face.cell_matrix(i, j) +=
                    	theta_pl
						* fe_iv.jump(j, point)
						* weighted_aver_i
						* JxW[point];

                }

                // pcv term
				double weighted_aver_rhs1 = AverageGradOperators::weighted_average_rhs(normals[point],
							pcv_grad0, pcv_grad1,
							coef0_pcv, coef1_pcv,
							weight0_pcv, weight1_pcv);

                copy_data_face.cell_rhs(i) +=
                		weighted_aver_rhs1
						* fe_iv.jump(i, point)
						* JxW[point];

                // pca term
				double weighted_aver_rhs2 = AverageGradOperators::weighted_average_rhs(normals[point],
							pca_grad0, pca_grad1,
							coef0_pca, coef1_pca,
							weight0_pca, weight1_pca);

                copy_data_face.cell_rhs(i) -=
                		weighted_aver_rhs2
						* fe_iv.jump(i, point)
						* JxW[point];

                // Gravity term
				double weighted_aver_rhs3 = AverageGradOperators::weighted_average_rhs(normals[point],
							gravity_fcn.vector_value(q_points[point]), gravity_fcn.vector_value(q_points[point]),
							coef0_g, coef1_g,
							weight0_g, weight1_g);

				copy_data_face.cell_rhs(i) -= weighted_aver_rhs3
						* fe_iv.jump(i, point)
						* JxW[point];


            }
        }
    };

    const auto copier = [&](const CopyData &c) {
        constraints.distribute_local_to_global(c.cell_matrix,
                                               c.cell_rhs,
                                               c.local_dof_indices,
											   system_matrix_pressure,
											   right_hand_side_pressure);

        for (auto &cdf : c.face_data)
        {
            constraints.distribute_local_to_global(cdf.cell_matrix,
            									   cdf.cell_rhs,
                                                   cdf.joint_dof_indices,
												   system_matrix_pressure,
												   right_hand_side_pressure);
        }
    };

    const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;

    const UpdateFlags cell_flags = update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values;
    const UpdateFlags face_flags = update_values | update_gradients |
								   update_quadrature_points |
								   update_normal_vectors | update_JxW_values;

    ScratchData scratch_data(mapping, fe, quadrature, cell_flags, face_quadrature, face_flags);
    CopyData         copy_data;

    const auto filtered_iterator_range =
      filter_iterators(dof_handler.active_cell_iterators(),
                       IteratorFilters::LocallyOwnedCell());

    MeshWorker::mesh_loop(filtered_iterator_range,
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
						  MeshWorker::assemble_ghost_faces_once |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);

    system_matrix_pressure.compress(VectorOperation::add);
    right_hand_side_pressure.compress(VectorOperation::add);
}

template <int dim>
void LiquidPressureProblem_midpoint<dim>::solve_pressure()
{
	if(use_direct_solver)
	{
		SolverControl cn;
		PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
	//	solver.set_symmetric_mode(true);
		solver.solve(system_matrix_pressure, pl_solution, right_hand_side_pressure);
	}
	else
	{
		SolverControl solver_control(pl_solution.size(), 1.e-7 * right_hand_side_pressure.l2_norm());

		PETScWrappers::SolverGMRES gmres(solver_control, mpi_communicator);
		PETScWrappers::PreconditionBoomerAMG preconditioner(system_matrix_pressure);

		gmres.solve(system_matrix_pressure, pl_solution, right_hand_side_pressure, preconditioner);

		Vector<double> localized_solution(pl_solution);
		constraints.distribute(localized_solution);

		pl_solution = localized_solution;
	}

}
} // namespace LiquidPressure

#endif //PL_PROBLEM_MID_HH
