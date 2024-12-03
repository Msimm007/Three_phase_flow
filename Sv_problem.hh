#ifndef SV_PROBLEM_HH
#define SV_PROBLEM_HH

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

namespace VaporSaturation
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
class VaporSaturationProblem
{
public:
	VaporSaturationProblem(Triangulation<dim, dim> &triangulation_,
			const unsigned int degree_,double theta_n_time_,
			double theta_Sv_, double penalty_Sv_,
			double penalty_Sv_bdry_, std::vector<unsigned int> dirichlet_id_sv_, bool use_exact_pl_in_Sv_,
			bool use_exact_Sa_in_Sv_,
			bool second_order_time_derivative_, bool second_order_extrapolation_,
			bool use_direct_solver_, bool Stab_v_, bool incompressible_, bool project_Darcy_with_gravity_,
			PETScWrappers::MPI::Vector kappa_abs_vec_,
			const unsigned int degreeRT_, bool project_only_kappa_,
			MPI_Comm mpi_communicator_, const unsigned int n_mpi_processes_, const unsigned int this_mpi_process_);

	void assemble_system_matrix_vapor_saturation(double time_step_,
						     double time_,
						     unsigned int timestep_number_,
						bool rebuild_matrix_,
                                                  const PETScWrappers::MPI::Vector& pl_solution_,						 const PETScWrappers::MPI::Vector& pl_solution_n_,
                                                  const PETScWrappers::MPI::Vector& pl_solution_nminus1_,
                                                  const PETScWrappers::MPI::Vector& Sa_solution_,
                                                  const PETScWrappers::MPI::Vector& Sa_solution_n_, const PETScWrappers::MPI::Vector& Sa_solution_nminus1_,
                                                  const PETScWrappers::MPI::Vector& Sv_solution_n_, const PETScWrappers::MPI::Vector& Sv_solution_nminus1_,
                                                  const PETScWrappers::MPI::Vector& totalDarcyvelocity_RT_);
	void solve_vapor_saturation();

    	void setup_system();

	PETScWrappers::MPI::Vector Sv_solution;
private:

    parallel::shared::Triangulation<dim> triangulation;
    const MappingQ1<dim> mapping;

    const QGauss<dim>     quadrature;
    const QGauss<dim - 1> face_quadrature;

    using ScratchData = MeshWorker::ScratchData<dim>;

    FE_DGQ<dim>     fe;
    DoFHandler<dim> dof_handler;
    const unsigned int degree;

    MPI_Comm mpi_communicator;

	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;

	ConditionalOStream pcout;

	IndexSet locally_owned_dofs;
	IndexSet locally_relevant_dofs;

	IndexSet locally_owned_dofs_RT;
	IndexSet locally_relevant_dofs_RT;

    SparsityPattern      sparsity_pattern;

    PETScWrappers::MPI::SparseMatrix system_matrix_vapor_saturation;
	PETScWrappers::MPI::Vector right_hand_side_vapor_saturation;

    PETScWrappers::MPI::Vector pl_solution_nminus1;
    PETScWrappers::MPI::Vector pl_solution_n;
    PETScWrappers::MPI::Vector pl_solution;


    PETScWrappers::MPI::Vector Sv_solution_nminus1;
    PETScWrappers::MPI::Vector Sv_solution_n;


    PETScWrappers::MPI::Vector Sa_solution_nminus1;
    PETScWrappers::MPI::Vector Sa_solution;
    PETScWrappers::MPI::Vector Sa_solution_n;


    FE_DGQ<dim> fe_dg0;
	DoFHandler<dim> dof_handler_dg0;
	IndexSet locally_owned_dofs_dg0;
	IndexSet locally_relevant_dofs_dg0;
    PETScWrappers::MPI::Vector kappa_abs_vec;

    double 		 time_step;
    double       time;
    unsigned int timestep_number;
    bool rebuild_matrix;	
    double       theta_n_time;

    double penalty_Sv;
    double penalty_Sv_bdry;

    std::vector<unsigned int> dirichlet_id_sv;

    double theta_Sv;
    bool Stab_v;

    bool incompressible;
    bool second_order_time_derivative;
    bool second_order_extrapolation;

    bool use_direct_solver;

    bool project_Darcy_with_gravity;
    bool project_only_kappa;

    bool use_exact_pl_in_Sv;
    bool use_exact_Sa_in_Sv;

	
    PETScWrappers::MPI::Vector totalDarcyvelocity_RT;

    const unsigned int degreeRT;
	FE_RaviartThomas<dim> fe_RT;
	DoFHandler<dim> dof_handler_RT;

    AffineConstraints<double> constraints;

};


template <int dim>
VaporSaturationProblem<dim>::VaporSaturationProblem(Triangulation<dim, dim> &triangulation_,
		const unsigned int degree_,  double theta_n_time_,
		double theta_Sv_, double penalty_Sv_,
		double penalty_Sv_bdry_, std::vector<unsigned int> dirichlet_id_sv_, bool use_exact_pl_in_Sv_,
		bool use_exact_Sa_in_Sv_,
		bool second_order_time_derivative_, bool second_order_extrapolation_,
		bool use_direct_solver_,bool Stab_v_, bool incompressible_, bool project_Darcy_with_gravity_,
		PETScWrappers::MPI::Vector kappa_abs_vec_,
		const unsigned int degreeRT_, bool project_only_kappa_,
		MPI_Comm mpi_communicator_, const unsigned int n_mpi_processes_, const unsigned int this_mpi_process_)
	: triangulation(MPI_COMM_WORLD)
	, mapping()
	, degree(degree_)
	, fe(degree_)
	, quadrature(degree_ + 1)
	, face_quadrature(degree_ + 1)
	, degreeRT(degreeRT_)
	, fe_RT(degreeRT_)
	, theta_n_time(theta_n_time_)
	, theta_Sv(theta_Sv_)
	, penalty_Sv(penalty_Sv_)
	, penalty_Sv_bdry(penalty_Sv_bdry_)
	, dirichlet_id_sv(dirichlet_id_sv_)
	, use_exact_pl_in_Sv(use_exact_pl_in_Sv_)
	, use_exact_Sa_in_Sv(use_exact_Sa_in_Sv_)
	, second_order_time_derivative(second_order_time_derivative_)
	, second_order_extrapolation(second_order_extrapolation_)
    , Stab_v(Stab_v_)
	, incompressible(incompressible_)
	, use_direct_solver(use_direct_solver_)
	, project_Darcy_with_gravity(project_Darcy_with_gravity_)
	, project_only_kappa(project_only_kappa_)
	, kappa_abs_vec(kappa_abs_vec_)
	, dof_handler(triangulation)
	, dof_handler_RT(triangulation)
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
void VaporSaturationProblem<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    dof_handler_RT.distribute_dofs(fe_RT);

    constraints.clear();
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    const std::vector<IndexSet> locally_owned_dofs_per_proc =
		  DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
	locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

	Sv_solution.reinit(locally_owned_dofs, mpi_communicator);
        dof_handler_dg0.distribute_dofs(fe_dg0);
	const std::vector<IndexSet> locally_owned_dofs_per_proc_dg0 =
			DoFTools::locally_owned_dofs_per_subdomain(dof_handler_dg0);
	locally_owned_dofs_dg0 = locally_owned_dofs_per_proc_dg0[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_dg0, locally_relevant_dofs_dg0);
}

template <int dim>
void VaporSaturationProblem<dim>::assemble_system_matrix_vapor_saturation( double time_step_,
				double time_, 	
				unsigned int timestep_number_,
				bool rebuild_matrix_,
                                const PETScWrappers::MPI::Vector& pl_solution_,
				const PETScWrappers::MPI::Vector& pl_solution_n_,
                                const PETScWrappers::MPI::Vector& pl_solution_nminus1_,
                                const PETScWrappers::MPI::Vector& Sa_solution_,
                                const PETScWrappers::MPI::Vector& Sa_solution_n_, 
				const PETScWrappers::MPI::Vector& Sa_solution_nminus1_,
                                const PETScWrappers::MPI::Vector& Sv_solution_n_,
				const PETScWrappers::MPI::Vector& Sv_solution_nminus1_,
                                const PETScWrappers::MPI::Vector& totalDarcyvelocity_RT_)
{

	// time terms
	time_step = time_step_;
	time = time_;
	timestep_number = timestep_number_;

	rebuild_matrix = rebuild_matrix_;
	
	if (rebuild_matrix){

	system_matrix_vapor_saturation.reinit(locally_owned_dofs,
					  locally_owned_dofs,
					  sparsity_pattern,
					  mpi_communicator);
	}	

	right_hand_side_vapor_saturation.reinit(locally_owned_dofs, mpi_communicator);

	const FEValuesExtractors::Vector velocities(0);

	using Iterator = typename DoFHandler<dim>::active_cell_iterator;
	BoundaryValuesVaporSaturation<dim> boundary_function;
	RightHandSideVaporSaturation<dim> right_hand_side_fcn;
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

    // Stabilization term
    StabVaporSaturation<dim> kappa_tilde_v_fcn;
    double kappa_tilde_v = kappa_tilde_v_fcn.value();

	// Capillary pressures
	CapillaryPressurePcv<dim> cap_p_pcv_fcn;

	// Neumann term
	NeumannTermVaporSaturation<dim> neumann_fcn;

	PETScWrappers::MPI::Vector temp_pl_solution;
	PETScWrappers::MPI::Vector temp_pl_solution_n;
	PETScWrappers::MPI::Vector temp_pl_solution_nminus1;
    PETScWrappers::MPI::Vector temp_pl_solution_kplus1;

	PETScWrappers::MPI::Vector temp_Sa_solution;
	PETScWrappers::MPI::Vector temp_Sa_solution_n;
	PETScWrappers::MPI::Vector temp_Sa_solution_nminus1;
    PETScWrappers::MPI::Vector temp_Sa_solution_kplus1;

	PETScWrappers::MPI::Vector temp_Sv_solution_n;
	PETScWrappers::MPI::Vector temp_Sv_solution_nminus1;
    PETScWrappers::MPI::Vector temp_Sv_solution_k;


    PETScWrappers::MPI::Vector temp_totalDarcyVelocity_RT;

	PETScWrappers::MPI::Vector temp_kappa;

	temp_pl_solution.reinit(locally_owned_dofs,
							locally_relevant_dofs,
							mpi_communicator);

	temp_pl_solution_n.reinit(locally_owned_dofs,
							  locally_relevant_dofs,
							  mpi_communicator);

	temp_pl_solution_nminus1.reinit(locally_owned_dofs,
									locally_relevant_dofs,
									mpi_communicator);
    temp_pl_solution_kplus1.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);


    temp_Sa_solution.reinit(locally_owned_dofs,
							locally_relevant_dofs,
							mpi_communicator);

	temp_Sa_solution_n.reinit(locally_owned_dofs,
							  locally_relevant_dofs,
							  mpi_communicator);

	temp_Sa_solution_nminus1.reinit(locally_owned_dofs,
									locally_relevant_dofs,
									mpi_communicator);
    temp_Sa_solution_kplus1.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);

	temp_Sv_solution_n.reinit(locally_owned_dofs,
							  locally_relevant_dofs,
							  mpi_communicator);

	temp_Sv_solution_nminus1.reinit(locally_owned_dofs,
									locally_relevant_dofs,
									mpi_communicator);
    temp_Sv_solution_k.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              mpi_communicator);

	temp_totalDarcyVelocity_RT.reinit(locally_owned_dofs_RT,
									  locally_relevant_dofs_RT,
									  mpi_communicator);

	temp_kappa.reinit(locally_owned_dofs_dg0,
					  locally_relevant_dofs_dg0,
					  mpi_communicator);

	temp_pl_solution = pl_solution_;
	temp_pl_solution_n = pl_solution_n_;
	temp_pl_solution_nminus1 = pl_solution_nminus1_;


    temp_Sa_solution = Sa_solution_;
	temp_Sa_solution_n = Sa_solution_n_;
	temp_Sa_solution_nminus1 = Sa_solution_nminus1_;


    temp_Sv_solution_n = Sv_solution_n_;
	temp_Sv_solution_nminus1 = Sv_solution_nminus1_;


    temp_totalDarcyVelocity_RT = totalDarcyvelocity_RT_;

	temp_kappa = kappa_abs_vec;

	// Volume integrals
	const auto cell_worker = [&](const auto &cell,
								 auto &scratch_data,
								 auto & copy_data)
	{
		const FEValues<dim> &fe_v = scratch_data.reinit(cell);

		const unsigned int n_dofs = fe_v.dofs_per_cell;
		copy_data.reinit(cell, n_dofs);

		const auto &q_points = fe_v.get_quadrature_points();
		const int n_qpoints = q_points.size();

		const std::vector<double> &JxW  = fe_v.get_JxW_values();

		FEValues<dim> fe_values_RT(fe_RT,
								   quadrature,
								   update_values);

		typename DoFHandler<dim>::cell_iterator cell_RT(&triangulation,
				cell->level(), cell->index(), &dof_handler_RT);

		fe_values_RT.reinit(cell_RT);

		std::vector<double>         rhs_values(n_qpoints);
		right_hand_side_fcn.set_time(time);
		right_hand_side_fcn.value_list(q_points, rhs_values);

		gravity_fcn.set_time(time);

		std::vector<double> pl_vals(n_qpoints);
		std::vector<double> old_pl_vals(n_qpoints);
		std::vector<double> old_pl_vals_nminus1(n_qpoints);
		std::vector<Tensor<1, dim>> pl_grads(n_qpoints);

		std::vector<double> Sa_vals(n_qpoints);
		std::vector<double> old_Sa_vals(n_qpoints);
		std::vector<double> old_Sa_vals_nminus1(n_qpoints);

		std::vector<double> old_Sv_vals(n_qpoints);
		std::vector<double> old_Sv_vals_nminus1(n_qpoints);
        std::vector<Tensor<1, dim>> old_Sv_grads(n_qpoints);

		fe_v.get_function_values(temp_pl_solution, pl_vals);
		fe_v.get_function_values(temp_pl_solution_n, old_pl_vals);
		fe_v.get_function_values(temp_pl_solution_nminus1, old_pl_vals_nminus1);
		fe_v.get_function_gradients(temp_pl_solution, pl_grads);

		fe_v.get_function_values(temp_Sa_solution, Sa_vals);
		fe_v.get_function_values(temp_Sa_solution_n, old_Sa_vals);
		fe_v.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1);

		fe_v.get_function_values(temp_Sv_solution_n, old_Sv_vals);
		fe_v.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1);
        fe_v.get_function_gradients(temp_Sv_solution_n, old_Sv_grads);

		std::vector<Tensor<1, dim>> DarcyVelocities(n_qpoints);
		fe_values_RT[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities);


		double kappa = temp_kappa[cell->global_active_cell_index()];

		for (unsigned int point = 0; point < n_qpoints; ++point)
		{
			double pl_value = pl_vals[point];
			double pl_value_n = old_pl_vals[point];
			double pl_value_nminus1 = old_pl_vals_nminus1[point];
			Tensor<1,dim> pl_grad = pl_grads[point];

			if(use_exact_pl_in_Sv)
			{
				pl_fcn.set_time(time);

				pl_value = pl_fcn.value(q_points[point]);
				pl_grad = pl_fcn.gradient(q_points[point]);

				pl_fcn.set_time(time - time_step);

				pl_value_n = pl_fcn.value(q_points[point]);

				pl_fcn.set_time(time - 2.0*time_step);

				pl_value_nminus1 = pl_fcn.value(q_points[point]);
			}

			double Sa_value = Sa_vals[point];
			double Sa_value_n = old_Sa_vals[point];
			double Sa_value_nminus1 = old_Sa_vals_nminus1[point];

			if(use_exact_Sa_in_Sv)
			{
				Sa_fcn.set_time(time);

				Sa_value = Sa_fcn.value(q_points[point]);

				Sa_fcn.set_time(time - time_step);

				Sa_value_n = Sa_fcn.value(q_points[point]);

				Sa_fcn.set_time(time - 2.0*time_step);

				Sa_value_nminus1 = Sa_fcn.value(q_points[point]);

			}

			double Sv_value_n = old_Sv_vals[point];
			double Sv_value_nminus1 = old_Sv_vals_nminus1[point];
                        Tensor<1,dim> Sv_grad_n = old_Sv_grads[point];
			Tensor<1,dim> totalDarcyVelo = DarcyVelocities[point];

			double Sv_nplus1_extrapolation = Sv_value_n;
			double Sa_nplus1_extrapolation = Sa_value_n;
			Tensor<1,dim> totalDarcyVelo_extrapolation = totalDarcyVelo;

			if(second_order_extrapolation)
			{
				Sv_nplus1_extrapolation *= 2.0;
				Sv_nplus1_extrapolation -= Sv_value_nminus1;

				Sa_nplus1_extrapolation *= 2.0;
				Sa_nplus1_extrapolation -= Sa_value_nminus1;

			}

			double phi_nplus1 = porosity_fcn.value(pl_value);
			double phi_n = porosity_fcn.value(pl_value_n);
			double phi_nminus1 = porosity_fcn.value(pl_value_nminus1);

			double rho_l = rho_l_fcn.value(pl_value);
			double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
			double rho_v_extr = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
			double rho_a = rho_a_fcn.value(pl_value);

			double rho_v_n = rho_v_fcn.value(pl_value_n, Sa_value_n, Sv_value_n);
			double rho_v_nminus1 = rho_v_fcn.value(pl_value_nminus1, Sa_value_nminus1, Sv_value_nminus1);

			if(incompressible)
			{
				rho_l = rho_v = rho_a = 1.0;
				rho_v_n = 1.0;
				rho_v_nminus1 = 1.0;
			}

			double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
			double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
			double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

			double lambda_l_extr = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
			double lambda_v_extr = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
			double lambda_a_extr = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

			double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;
			double rholambda_t_extr = rho_l*lambda_l_extr + rho_v_extr*lambda_v_extr + rho_a*lambda_a_extr;

			double dpcv_dSv = cap_p_pcv_fcn.derivative_wrt_Sv(Sv_nplus1_extrapolation);

			for (unsigned int i = 0; i < n_dofs; ++i)
			{
				for (unsigned int j = 0; j < n_dofs; ++j)
				{
					// Time term
					if(timestep_number == 1 || !second_order_time_derivative)
					{
                        // Time term
						copy_data.cell_matrix(i,j) +=
							(1.0/time_step)
							* phi_n
							* rho_v_n
							* fe_v.shape_value(i, point)
							* fe_v.shape_value(j, point)
							* JxW[point];
					}
					else
					{
						copy_data.cell_matrix(i,j) +=
							(1.0/time_step)
							* 1.5
							* phi_nplus1
							* rho_v
							* fe_v.shape_value(i, point)
							* fe_v.shape_value(j, point)
							* JxW[point];
					}
					// Diffusion Term
                    if(Stab_v)
                    {
                        copy_data.cell_matrix(i,j) +=
                                kappa_tilde_v
                                * kappa
                                * fe_v.shape_grad(i, point)
                                * fe_v.shape_grad(j, point)
                                * JxW[point];
                    }
                    else
                    {
                        copy_data.cell_matrix(i,j) +=
                                 rho_v
                                * lambda_v
                                * dpcv_dSv
                                * kappa
                                * fe_v.shape_grad(i, point)
                                * fe_v.shape_grad(j, point)
                                * JxW[point];
                    }
				}
				// Source term
				copy_data.cell_rhs(i) += right_hand_side_fcn.value(q_points[point])
						* fe_v.shape_value(i, point)
						* JxW[point];
				// Time term
				if(timestep_number == 1 || !second_order_time_derivative)
				{
					copy_data.cell_rhs(i) += (1.0/time_step) * phi_n * rho_v_n * Sv_value_n
							* fe_v.shape_value(i, point) * JxW[point];

				}
				else
				{
					copy_data.cell_rhs(i) += (1.0/time_step)
											* (2.0 * phi_n * rho_v_n * Sv_value_n
													- 0.5 * phi_nminus1 * rho_v_nminus1 * Sv_value_nminus1)
											* fe_v.shape_value(i, point) * JxW[point];
				}

                // Diffusion term moved to RHS - stab method
                if (Stab_v)
                {
                    copy_data.cell_rhs(i) += (-rho_v * lambda_v * dpcv_dSv + kappa_tilde_v)
                                             * kappa * Sv_grad_n
                                             * fe_v.shape_grad(i, point) * JxW[point];
                }

				// Darcy term. Coefficient depends on what was projected
				if(project_only_kappa)
					copy_data.cell_rhs(i) += (rho_v*lambda_v) * totalDarcyVelo_extrapolation
							* fe_v.shape_grad(i, point) * JxW[point];
				else
					copy_data.cell_rhs(i) += (rho_v*lambda_v/rholambda_t_extr) * totalDarcyVelo_extrapolation
							* fe_v.shape_grad(i, point) * JxW[point];

				// Gravity term
				if(!project_Darcy_with_gravity)
					copy_data.cell_rhs(i) += kappa*rho_v_fcn.value(pl_value_n, Sa_value_n, Sv_value_n)*rho_v*lambda_v
							* gravity_fcn.vector_value(q_points[point])
							* fe_v.shape_grad(i, point)
							* JxW[point];
			}
		}
	};
	// Boundary face integrals
    const auto boundary_worker = [&](const auto &cell,
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

    	FEFaceValues<dim> fe_face_values_RT(fe_RT,
    										face_quadrature,
    										update_values);

		typename DoFHandler<dim>::cell_iterator cell_RT(&triangulation,
				cell->level(), cell->index(), &dof_handler_RT);

		fe_face_values_RT.reinit(cell_RT, face_no);

		std::vector<double> g(n_qpoints);
		boundary_function.set_time(time);
		boundary_function.value_list(q_points, g);

		gravity_fcn.set_time(time);

		neumann_fcn.set_time(time);


		std::vector<double> pl_vals(n_qpoints);
		std::vector<Tensor<1, dim>> pl_grads(n_qpoints);

		std::vector<double> Sa_vals(n_qpoints);
		std::vector<double> old_Sa_vals(n_qpoints);
		std::vector<double> old_Sa_vals_nminus1(n_qpoints);

		std::vector<double> old_Sv_vals(n_qpoints);
		std::vector<double> old_Sv_vals_nminus1(n_qpoints);
                std::vector<Tensor<1, dim>> old_Sv_grads(n_qpoints);

		fe_face.get_function_values(temp_pl_solution, pl_vals);
		fe_face.get_function_gradients(temp_pl_solution, pl_grads);

		fe_face.get_function_values(temp_Sa_solution, Sa_vals);
		fe_face.get_function_values(temp_Sa_solution_n, old_Sa_vals);
		fe_face.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1);

		fe_face.get_function_values(temp_Sv_solution_n, old_Sv_vals);
		fe_face.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1);
                fe_face.get_function_gradients(temp_Sv_solution_n, old_Sv_grads);

		std::vector<Tensor<1, dim>> DarcyVelocities(n_qpoints);
		fe_face_values_RT[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities);

		double kappa = temp_kappa[cell->global_active_cell_index()];

        // Figure out if this face is Dirichlet or Neumann
        bool dirichlet = false;

        for(unsigned int i = 0; i < dirichlet_id_sv.size(); i++)
        {
        	if(cell->face(face_no)->boundary_id() == dirichlet_id_sv[i])
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
				double pl_value = pl_vals[point];
				Tensor<1,dim> pl_grad = pl_grads[point];

				if(use_exact_pl_in_Sv)
				{
					pl_fcn.set_time(time);

					pl_value = pl_fcn.value(q_points[point]);
					pl_grad = pl_fcn.gradient(q_points[point]);
				}
				double Sa_value = Sa_vals[point];
				double Sa_value_n = old_Sa_vals[point];
				double Sa_value_nminus1 = old_Sa_vals_nminus1[point];

				if(use_exact_Sa_in_Sv)
				{
					Sa_fcn.set_time(time);

					Sa_value = Sa_fcn.value(q_points[point]);

					Sa_fcn.set_time(time - time_step);

					Sa_value_n = Sa_fcn.value(q_points[point]);

					Sa_fcn.set_time(time - 2.0*time_step);

					Sa_value_nminus1 = Sa_fcn.value(q_points[point]);

				}

				double Sv_value_n = old_Sv_vals[point];
				double Sv_value_nminus1 = old_Sv_vals_nminus1[point];
                                Tensor<1,dim> Sv_grad_n = old_Sv_grads[point];
				Tensor<1,dim> totalDarcyVelo = DarcyVelocities[point];

				double Sv_nplus1_extrapolation = Sv_value_n;
				double Sa_nplus1_extrapolation = Sa_value_n;
				Tensor<1,dim> totalDarcyVelo_extrapolation = totalDarcyVelo;

				if(second_order_extrapolation)
				{
					Sv_nplus1_extrapolation *= 2.0;
					Sv_nplus1_extrapolation -= Sv_value_nminus1;

					Sa_nplus1_extrapolation *= 2.0;
					Sa_nplus1_extrapolation -= Sa_value_nminus1;

				}

				double rho_l = rho_l_fcn.value(pl_value);
				double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double rho_v_extr = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double rho_a = rho_a_fcn.value(pl_value);

				if(incompressible)
				{
					rho_l = rho_v = rho_a = 1.0;
				}

				double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

				double lambda_l_extr = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double lambda_v_extr = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
				double lambda_a_extr = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

				double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;
				double rholambda_t_extr = rho_l*lambda_l_extr + rho_v_extr*lambda_v_extr + rho_a*lambda_a_extr;

				double dpcv_dSv = cap_p_pcv_fcn.derivative_wrt_Sv(Sv_nplus1_extrapolation);

				double gamma_Sv_e = fabs(rho_v*lambda_v*kappa*dpcv_dSv);
				gamma_Sv_e += sqrt(totalDarcyVelo_extrapolation*totalDarcyVelo_extrapolation);

				double h_e = cell->face(face_no)->measure();
				double penalty_factor = (penalty_Sv_bdry/h_e) * gamma_Sv_e * degree*(degree + dim - 1);

				for (unsigned int i = 0; i < n_facet_dofs; ++i)
				{

					for (unsigned int j = 0; j < n_facet_dofs; ++j)
					{

                        if(Stab_v)
                        {
                            // Diffusion term
                            copy_data.cell_matrix(i, j) -=
                                    kappa_tilde_v
                                    * kappa
                                    * fe_face.shape_value(i, point)
                                    * fe_face.shape_grad(j, point)
                                    * normals[point]
                                    * JxW[point];

                            //Theta term
                            copy_data.cell_matrix(i, j) +=
                                    theta_Sv
                                    * kappa_tilde_v
                                    * kappa
                                    * fe_face.shape_grad(i, point)
                                    * normals[point]
                                    * fe_face.shape_value(j, point)
                                    * JxW[point];
                        }
                        else
                        {
                            // Diffusion term
                            copy_data.cell_matrix(i, j) -=
                                    rho_v
                                    * lambda_v
                                    * dpcv_dSv
                                    * kappa
                                    * fe_face.shape_value(i, point)
                                    * fe_face.shape_grad(j, point)
                                    * normals[point]
                                    * JxW[point];

                            //theta term
                            copy_data.cell_matrix(i, j) +=
                                    theta_Sv
                                    * rho_v
                                    * lambda_v
                                    * dpcv_dSv
                                    * kappa
                                    * fe_face.shape_grad(i, point)
                                    * normals[point]
                                    * fe_face.shape_value(j, point)
                                    * JxW[point];

                        }
						// Boundary condition
						copy_data.cell_matrix(i, j) +=
								penalty_factor
								* fe_face.shape_value(i, point)
								* fe_face.shape_value(j, point)
								* JxW[point];
					}
						// Boundary condition
						copy_data.cell_rhs(i) += penalty_factor
							* fe_face.shape_value(i, point)
							* g[point]
							* JxW[point];

                    if (Stab_v)
                    {
                        copy_data.cell_rhs(i) += (rho_v*lambda_v*dpcv_dSv - kappa_tilde_v) // added to RHS
                                                 * kappa
                                                 * Sv_grad_n
                                                 * normals[point]
                                                 * fe_face.shape_value(i, point)
                                                 * JxW[point];

                        copy_data.cell_rhs(i) += theta_Sv
                                                 * kappa_tilde_v
                                                 * kappa
                                                 * fe_face.shape_grad(i, point)
                                                 * normals[point]
                                                 * g[point]
                                                 * JxW[point];
                    }
                    else
                    {
                        // No term added to RHS
                        copy_data.cell_rhs(i) += theta_Sv
                                                 * rho_v
                                                 * lambda_v
                                                 * dpcv_dSv
                                                 * kappa
                                                 * fe_face.shape_grad(i, point)
                                                 * normals[point]
                                                 * g[point]
                                                 * JxW[point];
                    }
					// Darcy vel
					if(project_only_kappa)
						copy_data.cell_rhs(i) -= (rho_v*lambda_v)
							* totalDarcyVelo_extrapolation
							* normals[point]
							* fe_face.shape_value(i, point)
							* JxW[point];
					else
						copy_data.cell_rhs(i) -= (rho_v*lambda_v/rholambda_t_extr)
							* totalDarcyVelo_extrapolation
							* normals[point]
							* fe_face.shape_value(i, point)
							* JxW[point];

					if(!project_Darcy_with_gravity)
						copy_data.cell_rhs(i) -= kappa*rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation)*rho_v*lambda_v
							* gravity_fcn.vector_value(q_points[point])
							* normals[point]
							* fe_face.shape_value(i, point)
							* JxW[point];


				}
			}
		}
		else // Neumann boundary
		{
			for (unsigned int point = 0; point < q_points.size(); ++point)
			{
				double pl_value = pl_vals[point];
				Tensor<1,dim> pl_grad = pl_grads[point];

				if(use_exact_pl_in_Sv)
				{
					pl_fcn.set_time(time);

					pl_value = pl_fcn.value(q_points[point]);
					pl_grad = pl_fcn.gradient(q_points[point]);
				}

				double Sa_value = Sa_vals[point];

				if(use_exact_Sa_in_Sv)
				{
					Sa_fcn.set_time(time);
					Sa_value = Sa_fcn.value(q_points[point]);
				}

				double Sv_value_n = old_Sv_vals[point];
				double Sv_value_nminus1 = old_Sv_vals_nminus1[point];

				double Sv_nplus1_extrapolation = Sv_value_n;

				if(second_order_extrapolation)
				{
					Sv_nplus1_extrapolation *= 2.0;
					Sv_nplus1_extrapolation -= Sv_value_nminus1;
				}

				Tensor<1,dim> neumann_term = neumann_fcn.vector_value(q_points[point]);
				Tensor<1,dim> totalDarcyVelo = DarcyVelocities[point];

				double rho_l = rho_l_fcn.value(pl_value);
				double rho_v = rho_v_fcn.value(pl_value, Sa_value, Sv_nplus1_extrapolation);
				double rho_a = rho_a_fcn.value(pl_value);

				if(incompressible)
	            {
	            	rho_l = rho_v = rho_a = 1.0;
	            }

				double lambda_l = lambda_l_fcn.value(pl_value, Sa_value, Sv_nplus1_extrapolation);
				double lambda_v = lambda_v_fcn.value(pl_value, Sa_value, Sv_nplus1_extrapolation);
				double lambda_a = lambda_a_fcn.value(pl_value, Sa_value, Sv_nplus1_extrapolation);

				double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

				for (unsigned int i = 0; i < n_facet_dofs; ++i)
				{

					if(project_only_kappa)
						copy_data.cell_rhs(i) -= rho_v*lambda_v
												* totalDarcyVelo
												* normals[point]
												* fe_face.shape_value(i, point)
												* JxW[point];
					else
						copy_data.cell_rhs(i) -= (rho_v*lambda_v/rholambda_t)
												* totalDarcyVelo
												* normals[point]
												* fe_face.shape_value(i, point)
												* JxW[point];

					copy_data.cell_rhs(i) += neumann_term
											* normals[point]
											* fe_face.shape_value(i, point)
											* JxW[point];
				}
			}
		}

	};

        // Interior faces integrals
	const auto face_worker = [&](const auto &cell,
				     const unsigned int &f,
				     const unsigned int &sf,
				     const auto &ncell,
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

		FEFaceValues<dim> fe_face_values_RT(fe_RT,
						    face_quadrature,
						    update_values);

		FEFaceValues<dim> fe_face_values_RT_neighbor(fe_RT,
							     face_quadrature,
							     update_values);

		typename DoFHandler<dim>::cell_iterator cell_RT(&triangulation,
				cell->level(), cell->index(), &dof_handler_RT);
		typename DoFHandler<dim>::cell_iterator cell_RT_neighbor(&triangulation,
				ncell->level(), ncell->index(), &dof_handler_RT);

		fe_face_values_RT.reinit(cell_RT, f);
		fe_face_values_RT_neighbor.reinit(cell_RT_neighbor, nf);

		gravity_fcn.set_time(time);

		std::vector<double> pl_vals(n_qpoints);
		std::vector<double> pl_vals_neighbor(n_qpoints);

		std::vector<Tensor<1, dim>> pl_grads(n_qpoints);
		std::vector<Tensor<1, dim>> pl_grads_neighbor(n_qpoints);

		std::vector<double> Sa_vals(n_qpoints);
		std::vector<double> Sa_vals_neighbor(n_qpoints);

		std::vector<double> old_Sa_vals(n_qpoints);
		std::vector<double> old_Sa_vals_neighbor(n_qpoints);

		std::vector<double> old_Sa_vals_nminus1(n_qpoints);
		std::vector<double> old_Sa_vals_nminus1_neighbor(n_qpoints);

		std::vector<double> old_Sv_vals(n_qpoints);
		std::vector<double> old_Sv_vals_nminus1(n_qpoints);
        std::vector<Tensor<1, dim>> old_Sv_grads(n_qpoints);

		std::vector<double> old_Sv_vals_neighbor(n_qpoints);
		std::vector<double> old_Sv_vals_nminus1_neighbor(n_qpoints);
        std::vector<Tensor<1, dim>> old_Sv_grads_neighbor(n_qpoints);

		fe_face.get_function_values(temp_pl_solution, pl_vals);
		fe_face_neighbor.get_function_values(temp_pl_solution, pl_vals_neighbor);

		fe_face.get_function_gradients(temp_pl_solution, pl_grads);
		fe_face_neighbor.get_function_gradients(temp_pl_solution, pl_grads_neighbor);

		fe_face.get_function_values(temp_Sa_solution, Sa_vals);
		fe_face_neighbor.get_function_values(temp_Sa_solution, Sa_vals_neighbor);

		fe_face.get_function_values(temp_Sa_solution_n, old_Sa_vals);
		fe_face_neighbor.get_function_values(temp_Sa_solution_n, old_Sa_vals_neighbor);

		fe_face.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1);
		fe_face_neighbor.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1_neighbor);

		fe_face.get_function_values(temp_Sv_solution_n, old_Sv_vals);
		fe_face.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1);
        fe_face.get_function_gradients(temp_Sv_solution_n, old_Sv_grads);

		fe_face_neighbor.get_function_values(temp_Sv_solution_n, old_Sv_vals_neighbor);
		fe_face_neighbor.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1_neighbor);
                fe_face_neighbor.get_function_gradients(temp_Sv_solution_n, old_Sv_grads_neighbor);

		std::vector<Tensor<1, dim>> DarcyVelocities(n_qpoints);
		fe_face_values_RT[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities);

		std::vector<Tensor<1, dim>> DarcyVelocities_neighbor(n_qpoints);
		fe_face_values_RT_neighbor[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities_neighbor);

		double kappa0 = temp_kappa[cell->global_active_cell_index()];
		double kappa1 = temp_kappa[ncell->global_active_cell_index()];

		for (unsigned int point = 0; point < n_qpoints; ++point)
		{
			double pl_value0 = pl_vals[point];
			double pl_value1 = pl_vals_neighbor[point];

			Tensor<1,dim> pl_grad0 = pl_grads[point];
			Tensor<1,dim> pl_grad1 = pl_grads_neighbor[point];

			if(use_exact_pl_in_Sv)
			{
				pl_fcn.set_time(time);

				pl_value0 = pl_fcn.value(q_points[point]);
				pl_value1 = pl_value0;

				pl_grad0 = pl_fcn.gradient(q_points[point]);
				pl_grad1 = pl_grad0;

			}

			double Sa_value0 = Sa_vals[point];
			double Sa_value1 = Sa_vals_neighbor[point];
			double Sa_value0_n = old_Sa_vals[point];
			double Sa_value1_n = old_Sa_vals_neighbor[point];
			double Sa_value0_nminus1 = old_Sa_vals_nminus1[point];
			double Sa_value1_nminus1 = old_Sa_vals_nminus1_neighbor[point];

			if(use_exact_Sa_in_Sv)
			{
				Sa_fcn.set_time(time);

				Sa_value0 = Sa_fcn.value(q_points[point]);
				Sa_value1 = Sa_value0;

				Sa_fcn.set_time(time - time_step);

				Sa_value0_n = Sa_fcn.value(q_points[point]);
				Sa_value1_n = Sa_value0_n;

				Sa_fcn.set_time(time - 2.0*time_step);

				Sa_value0_nminus1 = Sa_fcn.value(q_points[point]);
				Sa_value1_nminus1 = Sa_value0_nminus1;
			}

			double Sv_value0_n = old_Sv_vals[point];
			double Sv_value1_n = old_Sv_vals_neighbor[point];
			double Sv_value0_nminus1 = old_Sv_vals_nminus1[point];
			double Sv_value1_nminus1 = old_Sv_vals_nminus1_neighbor[point];
            Tensor<1,dim> Sv_grad0_n = old_Sv_grads[point];
            Tensor<1,dim> Sv_grad1_n = old_Sv_grads_neighbor[point];

			Tensor<1,dim> totalDarcyVelo0 = DarcyVelocities[point];
			Tensor<1,dim> totalDarcyVelo1 = DarcyVelocities_neighbor[point];

			double Sv_nplus1_extrapolation0 = Sv_value0_n;
			double Sv_nplus1_extrapolation1 = Sv_value1_n;
			double Sa_nplus1_extrapolation0 = Sa_value0_n;
			double Sa_nplus1_extrapolation1 = Sa_value1_n;
			Tensor<1,dim> totalDarcyVelo_extrapolation0 = totalDarcyVelo0;
			Tensor<1,dim> totalDarcyVelo_extrapolation1 = totalDarcyVelo1;

			if(second_order_extrapolation)
			{
				Sv_nplus1_extrapolation0 *= 2.0;
				Sv_nplus1_extrapolation0 -= Sv_value0_nminus1;

				Sv_nplus1_extrapolation1 *= 2.0;
				Sv_nplus1_extrapolation1 -= Sv_value1_nminus1;

				Sa_nplus1_extrapolation0 *= 2.0;
				Sa_nplus1_extrapolation0 -= Sa_value0_nminus1;

				Sa_nplus1_extrapolation1 *= 2.0;
				Sa_nplus1_extrapolation1 -= Sa_value1_nminus1;

			}

			double rho_l0 = rho_l_fcn.value(pl_value0);
			double rho_l1 = rho_l_fcn.value(pl_value1);

			double rho_v0 = rho_v_fcn.value(pl_value0, Sa_value0, Sv_nplus1_extrapolation0);
			double rho_v1 = rho_v_fcn.value(pl_value1, Sa_value1, Sv_nplus1_extrapolation1);

			double rho_v_extr0 = rho_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
			double rho_v_extr1 = rho_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

			double rho_a0 = rho_a_fcn.value(pl_value0);
			double rho_a1 = rho_a_fcn.value(pl_value1);

            		if(incompressible)
            		{
            			rho_l0 = rho_v0 = rho_a0 = 1.0;
            			rho_l1 = rho_v1 = rho_a1 = 1.0;
            		}

			double lambda_l0 = lambda_l_fcn.value(pl_value0, Sa_value0, Sv_nplus1_extrapolation0);
			double lambda_v0 = lambda_v_fcn.value(pl_value0, Sa_value0, Sv_nplus1_extrapolation0);
			double lambda_a0 = lambda_a_fcn.value(pl_value0, Sa_value0, Sv_nplus1_extrapolation0);

			double lambda_l1 = lambda_l_fcn.value(pl_value1, Sa_value1, Sv_nplus1_extrapolation1);
			double lambda_v1 = lambda_v_fcn.value(pl_value1, Sa_value1, Sv_nplus1_extrapolation1);
			double lambda_a1 = lambda_a_fcn.value(pl_value1, Sa_value1, Sv_nplus1_extrapolation1);

			double lambda_l_extr0 = lambda_l_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
			double lambda_v_extr0 = lambda_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
			double lambda_a_extr0 = lambda_a_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);

			double lambda_l_extr1 = lambda_l_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
			double lambda_v_extr1 = lambda_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
			double lambda_a_extr1 = lambda_a_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

			double rholambda_t0 = rho_l0*lambda_l0 + rho_v0*lambda_v0 + rho_a0*lambda_a0;
			double rholambda_t1 = rho_l1*lambda_l1 + rho_v1*lambda_v1 + rho_a1*lambda_a1;

			double rholambda_t_extr0 = rho_l0*lambda_l_extr0 + rho_v_extr0*lambda_v_extr0 + rho_a0*lambda_a_extr0;
			double rholambda_t_extr1 = rho_l1*lambda_l_extr1 + rho_v_extr1*lambda_v_extr1 + rho_a1*lambda_a_extr1;

			double dpcv_dSv0 = cap_p_pcv_fcn.derivative_wrt_Sv(Sv_nplus1_extrapolation0);
			double dpcv_dSv1 = cap_p_pcv_fcn.derivative_wrt_Sv(Sv_nplus1_extrapolation1);

			// Diffusion coefficients and weights
			double coef0_diff = rho_v0*lambda_v0*kappa0*dpcv_dSv0;
			double coef1_diff = rho_v1*lambda_v1*kappa1*dpcv_dSv1;
			double weight0_diff = coef1_diff/(coef0_diff + coef1_diff + 1.e-20);
			double weight1_diff = coef0_diff/(coef0_diff + coef1_diff + 1.e-20);

            // Diffusion coefficients and weights for stab method
             double coef0_diff_stab = fabs(kappa0*kappa_tilde_v);
             double coef1_diff_stab = fabs(kappa1*kappa_tilde_v);

             double weight0_diff_stab = coef1_diff_stab/(coef0_diff_stab + coef1_diff_stab + 1.e-20);
             double weight1_diff_stab = coef0_diff_stab/(coef0_diff_stab + coef1_diff_stab + 1.e-20);

		    double coef0_Sv_stab = (-rho_v0*lambda_v0*dpcv_dSv0+kappa_tilde_v)*kappa0;
            double coef1_Sv_stab = (-rho_v1*lambda_v1*dpcv_dSv1+kappa_tilde_v)*kappa1;

            double weight0_Sv_stab = coef1_Sv_stab/(coef0_Sv_stab + coef1_Sv_stab + 1.e-20);
            double weight1_Sv_stab = coef0_Sv_stab/(coef0_Sv_stab + coef1_Sv_stab + 1.e-20);

			double gamma_Sv_e = fabs(2.0*coef0_diff*coef1_diff/(coef0_diff + coef1_diff + 1.e-20));

			double h_e = cell->face(f)->measure();
			double penalty_factor = (penalty_Sv/h_e) * gamma_Sv_e * degree*(degree + dim - 1);

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

                    if (Stab_v)
                    {
                        double weighted_aver_j_stab = AverageGradOperators::weighted_average_gradient<dim>(cell, f, sf, ncell, nf,
                                                                                                           nsf, fe_iv,
                                                                                                           normals[point],
                                                                                                           j, point,
                                                                                                           coef0_diff_stab, coef1_diff_stab,
                                                                                                           weight0_diff_stab, weight1_diff_stab);
                        copy_data_face.cell_matrix(i, j) -=
                                fe_iv.jump(i, point)
                                * weighted_aver_j_stab
                                * JxW[point];

                        double weighted_aver_i_stab = AverageGradOperators::weighted_average_gradient<dim>(cell, f, sf, ncell, nf,
                                                                                                           nsf, fe_iv,
                                                                                                           normals[point],
                                                                                                           i, point,
                                                                                                           coef0_diff_stab, coef1_diff_stab,
                                                                                                           weight0_diff_stab, weight1_diff_stab);
                        copy_data_face.cell_matrix(i, j) +=
                                theta_Sv
                                * fe_iv.jump(j, point)
                                * weighted_aver_i_stab
                                * JxW[point];
                    }
                    else
                    {
                        double weighted_aver_j = AverageGradOperators::weighted_average_gradient<dim>(cell, f, sf, ncell, nf,
                                                                                                      nsf, fe_iv,
                                                                                                      normals[point],
                                                                                                      j, point,
                                                                                                      coef0_diff, coef1_diff,
                                                                                                      weight0_diff, weight1_diff);
                        copy_data_face.cell_matrix(i, j) -=
                                fe_iv.jump(i, point)
                                * weighted_aver_j
                                * JxW[point];

                        double weighted_aver_i = AverageGradOperators::weighted_average_gradient<dim>(cell, f, sf, ncell, nf,
                                                                                                      nsf, fe_iv,
                                                                                                      normals[point],
                                                                                                      i, point,
                                                                                                      coef0_diff, coef1_diff,
                                                                                                      weight0_diff, weight1_diff);
                        copy_data_face.cell_matrix(i, j) +=
                                theta_Sv
                                * fe_iv.jump(j, point)
                                * weighted_aver_i
                                * JxW[point];
                    }

				}

                            // Sv term added to the RHS
                            if (Stab_v)
                            {
                                double weighted_aver_rhs0_stab = AverageGradOperators::weighted_average_rhs<dim>(normals[point],
                                                                                                                 Sv_grad0_n, Sv_grad1_n,
                                                                                                                 coef0_Sv_stab, coef1_Sv_stab,
                                                                                                                 weight0_Sv_stab, weight1_Sv_stab);

                                copy_data_face.cell_rhs(i) -=
                                        weighted_aver_rhs0_stab
                                        * fe_iv.jump(i, point)
                                        * JxW[point];
                            }
				// Darcy velocity and upwind stuff
				Tensor<1,dim> g_val = gravity_fcn.vector_value(q_points[point]);
				double coef0_darcy, coef1_darcy;

				if(project_only_kappa)
				{
					coef0_darcy = rho_v0*lambda_v0;
					coef1_darcy = rho_v1*lambda_v1;
				}
				else
				{
					coef0_darcy = rho_v0*lambda_v0/rholambda_t_extr0;
					coef1_darcy = rho_v1*lambda_v1/rholambda_t_extr1;
				}

				double Dg0 = kappa0*rho_v_fcn.value(pl_value0, Sa_value0, Sv_nplus1_extrapolation0)*rho_v0*lambda_v0;
				double Dg1 = kappa1*rho_v_fcn.value(pl_value1, Sa_value1, Sv_nplus1_extrapolation1)*rho_v1*lambda_v1;

				double psi_upwind = coef0_darcy*totalDarcyVelo_extrapolation0*normals[point]
									+ coef1_darcy*totalDarcyVelo_extrapolation1*normals[point]
									+ Dg0*g_val*normals[point] + Dg1*g_val*normals[point];
				double average_uRT = 0.5*(totalDarcyVelo_extrapolation0*normals[point]
										+ totalDarcyVelo_extrapolation1*normals[point]);

				if(psi_upwind >= 0.0)
				{
					copy_data_face.cell_rhs(i) -=
							coef0_darcy
							* average_uRT
							* fe_iv.jump(i, point)
							* JxW[point];
				}
				else
				{
					copy_data_face.cell_rhs(i) -=
							coef1_darcy
							* average_uRT
							* fe_iv.jump(i, point)
							* JxW[point];
				}

				// Gravity terms
				double coef0_g, coef1_g;

				if(!project_Darcy_with_gravity)
				{
					coef0_g = kappa0*rho_v_fcn.value(pl_value0, Sa_value0, Sv_nplus1_extrapolation0)*rho_v0*lambda_v0;
					coef1_g = kappa1*rho_v_fcn.value(pl_value1, Sa_value1, Sv_nplus1_extrapolation1)*rho_v1*lambda_v1;

					double weight0_g = coef1_g/(coef0_g + coef1_g + 1.e-20);
					double weight1_g = coef0_g/(coef0_g + coef1_g + 1.e-20);

					double weighted_aver_rhs3 = AverageGradOperators::weighted_average_rhs(normals[point],
								g_val, g_val,
								coef0_g, coef1_g,
								weight0_g, weight1_g);

					copy_data_face.cell_rhs(i) -= weighted_aver_rhs3
							* fe_iv.jump(i, point)
							* JxW[point];
				}

			}
		}
	};

	const auto copier = [&](const CopyData &c) {
		constraints.distribute_local_to_global(c.cell_matrix,
							   c.cell_rhs,
							   c.local_dof_indices,
							   system_matrix_vapor_saturation,
							   right_hand_side_vapor_saturation);

		for (auto &cdf : c.face_data)
		{
			constraints.distribute_local_to_global(cdf.cell_matrix,
								   cdf.cell_rhs,
								   cdf.joint_dof_indices,
								   system_matrix_vapor_saturation,
								   right_hand_side_vapor_saturation);
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

    system_matrix_vapor_saturation.compress(VectorOperation::add);
    right_hand_side_vapor_saturation.compress(VectorOperation::add);
}

template <int dim>
void VaporSaturationProblem<dim>::solve_vapor_saturation()
{
	if(use_direct_solver)
	{
		SolverControl cn;
		PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
	//	solver.set_symmetric_mode(true);
		solver.solve(system_matrix_vapor_saturation, Sv_solution, right_hand_side_vapor_saturation);
	}
	else
	{
		SolverControl solver_control(pl_solution.size(), 1.e-7 * right_hand_side_vapor_saturation.l2_norm());

		PETScWrappers::SolverGMRES gmres(solver_control, mpi_communicator);
		PETScWrappers::PreconditionBoomerAMG preconditioner(system_matrix_vapor_saturation);

		gmres.solve(system_matrix_vapor_saturation, Sv_solution, right_hand_side_vapor_saturation, preconditioner);

		Vector<double> localized_solution(Sv_solution);
		constraints.distribute(localized_solution);

		Sv_solution = localized_solution;
	}
}
} // namespace VaporSaturation

#endif //SV_PROBLEM_HH
