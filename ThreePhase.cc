#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
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
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/base/timer.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/base/parameter_handler.h>

#include "AverageGradientOperators.hh"

#include "RT_projection.hh"
#include "pl_problem.hh"
#include "Sa_problem.hh"
#include "Sv_problem.hh"

// for midpoint method

#include "RT_projection_midpoint.hh"
#include "pl_problem_midpoint.hh"
#include "Sa_problem_midpoint.hh"
#include "Sv_problem_midpoint.hh"

// PETSc stuff
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>


#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace CouplingPressureSaturation
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

class ParameterReader : public Subscriptor
{
	public:
	ParameterReader(ParameterHandler &);
	void read_parameters(const std::string &);

	private:
	void declare_parameters();
	ParameterHandler &prm;
};

ParameterReader::ParameterReader(ParameterHandler &paramhandler)
	: prm(paramhandler)
{}


void ParameterReader::declare_parameters()
{

	prm.enter_subsection("Time discretization parameters");
	{
		prm.declare_entry("Final time",
						  "1.0",
						  Patterns::Double(0.0));

		prm.declare_entry("Initial time step",
						  "1.0",
						  Patterns::Double(0.0));

		prm.declare_entry("Second order time derivative",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Second order extrapolation",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Use implicit time term in pl",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Create initial perturbation",
						  "false",
						  Patterns::Bool());
		prm.declare_entry("Midpoint method",
						  "false",
						  Patterns::Bool());
        prm.declare_entry("Theta_n",
                          "0.5",
                          Patterns::Double(0.0));
		
	}
	prm.leave_subsection();


	prm.enter_subsection("Spatial discretization parameters");
	{
		prm.declare_entry("Dimension",
						  "2",
						  Patterns::Integer(2));

		prm.declare_entry("Degree",
						  "1",
						  Patterns::Integer(1));

		prm.declare_entry("Initial level of refinement",
						  "0",
						  Patterns::Integer(0));

		prm.declare_entry("Final level of refinement",
						  "7",
						  Patterns::Integer(0));

		prm.declare_entry("Incompressible",
						  "true",
						  Patterns::Bool());

        prm.declare_entry("Stab_t",
                          "true",
                          Patterns::Bool());						  

        prm.declare_entry("Stab_a",
                          "true",
                          Patterns::Bool());
        prm.declare_entry("Stab_v",
                          "true",
                          Patterns::Bool());

		prm.declare_entry("Theta pl",
						  "1.0",
						  Patterns::Double());

		prm.declare_entry("Theta Sa",
						  "1.0",
						  Patterns::Double());

		prm.declare_entry("Theta Sv",
						  "1.0",
						  Patterns::Double());

		prm.declare_entry("Penalty pl",
						  "1.0",
						  Patterns::Double(0.0));

		prm.declare_entry("Penalty Sa",
						  "1.0",
						  Patterns::Double(0.0));

		prm.declare_entry("Penalty Sv",
						  "1.0",
						  Patterns::Double(0.0));

		prm.declare_entry("Penalty pl boundary",
						  "1.0",
						  Patterns::Double(0.0));

		prm.declare_entry("Penalty Sa boundary",
						  "1.0",
						  Patterns::Double(0.0));

		prm.declare_entry("Penalty Sv boundary",
						  "1.0",
						  Patterns::Double(0.0));

		prm.declare_entry("Project to RT0",
						  "true",
						  Patterns::Bool());

		prm.declare_entry("Project to RT0 with kappa only",
						  "true",
						  Patterns::Bool());

		prm.declare_entry("Use direct solver for linear systems",
						  "true",
						  Patterns::Bool());

		prm.declare_entry("Use exact pl in Sa",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Use exact pl in Sv",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Use exact pl in RT",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Use exact Sa in pl",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Use exact Sa in Sv",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Use exact Sa in RT",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Use exact Sv in pl",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Use exact Sv in Sa",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Use exact Sv in RT",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Project Darcy with gravity",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Add explicit artificial viscosity",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Add implicit artificial viscosity",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Artificial viscosity multiple for Sa",
						  "1.0",
						  Patterns::Double(0.0));

		prm.declare_entry("Two phase problem",
						  "false",
						  Patterns::Bool());

	}
	prm.leave_subsection();

	prm.enter_subsection("Output parameters");
	{
		prm.declare_entry("print vtk",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Frequency for vtk printing",
						  "1",
						  Patterns::Integer(1));

		prm.declare_entry("Compute errors",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Compute energy",
						  "false",
						  Patterns::Bool());
	}
	prm.leave_subsection();

	prm.enter_subsection("Continuation from previous solution");
	{
		prm.declare_entry("Output solution",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Frequency for solution output",
						  "1",
						  Patterns::Integer(1));

		prm.declare_entry("Continue solution",
						  "false",
						  Patterns::Bool());

		prm.declare_entry("Time step number",
						  "1",
						  Patterns::Integer(1));
	}
	prm.leave_subsection();
}

void ParameterReader::read_parameters(const std::string &parameter_file)
{
	declare_parameters();

	prm.parse_input(parameter_file);
}

template <int dim>
class CoupledPressureSaturationProblem
{
public:
	CoupledPressureSaturationProblem(const unsigned int degree,  const unsigned int degreeRT_,
			double delta_t, double tf, const unsigned int refinement, ParameterHandler &param);
    void run();

private:
    void load_gmsh_mesh();

    void setup_system();

    void compute_errors() const;
    void compute_energy_norm_error_pl() const;
    void output_vtk_initial_cond() const;
    void output_vtk() const;
    void output_solution() const;
    void read_initial_condition();
    void create_kappa_abs_vector();
    double calculate_energy(double &real_energy);

    using ScratchData = MeshWorker::ScratchData<dim>;

    parallel::shared::Triangulation<dim>   triangulation;

    std::vector<unsigned int> dirichlet_id_pl, dirichlet_id_sa, dirichlet_id_sv;

    const MappingQ1<dim> mapping;

    // DG space
    FE_DGQ<dim>     fe;
    DoFHandler<dim> dof_handler;
    const unsigned int degree;

    // RT Projection space
	FE_RaviartThomas<dim> fe_RT;
	DoFHandler<dim> dof_handler_RT;
	const unsigned int degreeRT;

	MPI_Comm mpi_communicator;

	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;

	ConditionalOStream pcout;

	IndexSet locally_owned_dofs;
	IndexSet locally_relevant_dofs;

	IndexSet locally_owned_dofs_RT;
	IndexSet locally_relevant_dofs_RT;

    PETScWrappers::MPI::Vector pl_solution;
    PETScWrappers::MPI::Vector pl_solution_n;
    PETScWrappers::MPI::Vector pl_solution_k;
    PETScWrappers::MPI::Vector pl_solution_kplus1;
    PETScWrappers::MPI::Vector pl_solution_nminus1;
    PETScWrappers::MPI::Vector pl_solution_nminus2;


    PETScWrappers::MPI::Vector Sa_solution;
    PETScWrappers::MPI::Vector Sa_solution_n;
    PETScWrappers::MPI::Vector Sa_solution_k;
    PETScWrappers::MPI::Vector Sa_solution_kplus1;
    PETScWrappers::MPI::Vector Sa_solution_nminus1;
    PETScWrappers::MPI::Vector Sa_solution_nminus2;

    PETScWrappers::MPI::Vector Sv_solution;
    PETScWrappers::MPI::Vector Sv_solution_n;
    PETScWrappers::MPI::Vector Sv_solution_k;
    PETScWrappers::MPI::Vector Sv_solution_kplus1;
    PETScWrappers::MPI::Vector Sv_solution_nminus1;
    PETScWrappers::MPI::Vector Sv_solution_nminus2;

    PETScWrappers::MPI::Vector pl_difference;
    PETScWrappers::MPI::Vector Sa_difference;
    PETScWrappers::MPI::Vector Sv_difference;

    // RT Projection vector
	PETScWrappers::MPI::Vector totalDarcyvelocity_RT_Sa;
	PETScWrappers::MPI::Vector totalDarcyvelocity_RT_Sa_n;
	PETScWrappers::MPI::Vector totalDarcyvelocity_RT_Sv;
	PETScWrappers::MPI::Vector totalDarcyvelocity_RT_Sv_n;

	// Kappa stuff
	FE_DGQ<dim> fe_dg0;
	DoFHandler<dim> dof_handler_dg0;
	IndexSet locally_owned_dofs_dg0;
	IndexSet locally_relevant_dofs_dg0;
	PETScWrappers::MPI::Vector kappa_abs_vec;

    ParameterHandler &prm;

    double 		 time_step;
    double       time;
    unsigned int timestep_number;
    double 		 final_time;
	bool 		 midpoint_method;
    double       theta_n_time;

    bool incompressible;

	bool Stab_t;
    bool Stab_a;
    bool Stab_v;

    double penalty_pl;
    double penalty_Sa;
    double penalty_Sv;

    double penalty_pl_bdry;
    double penalty_Sa_bdry;
    double penalty_Sv_bdry;

    double theta_pl;
    double theta_Sa;
    double theta_Sv;

    unsigned int ref_level;

    bool print_vtk;
    int vtk_freq;
    bool compute_errors_sol;
    bool compute_energy;

    bool output_sol;
    int output_sol_freq;
    bool continue_solution;

    bool second_order_time_derivative;
    bool second_order_extrapolation;

    bool implicit_time_pl;

    bool create_initial_perturbation;

    bool use_exact_pl_in_Sa;
    bool use_exact_pl_in_Sv;
    bool use_exact_pl_in_RT;
    bool use_exact_Sa_in_pl;
    bool use_exact_Sa_in_Sv;
    bool use_exact_Sa_in_RT;
    bool use_exact_Sv_in_pl;
    bool use_exact_Sv_in_Sa;
    bool use_exact_Sv_in_RT;

    bool project_to_RT0;
    bool project_Darcy_with_gravity;
    bool project_only_kappa;

    bool use_direct_solver;

    bool artificial_visc_exp;
    bool artificial_visc_imp;
    double art_visc_multiple_Sa;

    bool two_phase;

    AffineConstraints<double> constraints;

};



template <int dim>
CoupledPressureSaturationProblem<dim>::CoupledPressureSaturationProblem(const unsigned int degree, const unsigned int degreeRT_,
		double delta_t, double tf, const unsigned int refinement, ParameterHandler &param)
	: triangulation(MPI_COMM_WORLD)
	, mapping()
	, degree(degree)
	, degreeRT(degreeRT_)
	, fe(degree)
	, fe_RT(degreeRT_)
	, time_step(delta_t)
	, final_time(tf)
	, ref_level(refinement)
	, dof_handler(triangulation)
	, dof_handler_RT(triangulation)
	, fe_dg0(0)
	, dof_handler_dg0(triangulation)
	, prm(param)
	, mpi_communicator(MPI_COMM_WORLD)
	, n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
	, this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
	, pcout(std::cout, (this_mpi_process == 0))
{
	prm.enter_subsection("Time discretization parameters");

	second_order_time_derivative = prm.get_bool("Second order time derivative");
	second_order_extrapolation = prm.get_bool("Second order extrapolation");
    implicit_time_pl = prm.get_bool("Use implicit time term in pl");
    create_initial_perturbation = prm.get_bool("Create initial perturbation");
	midpoint_method = prm.get_bool("Midpoint method");
    theta_n_time = prm.get_double("Theta_n");


	prm.leave_subsection();

	prm.enter_subsection("Spatial discretization parameters");


    incompressible = prm.get_bool("Incompressible");
	Stab_t = prm.get_bool("Stab_t");
    Stab_a = prm.get_bool("Stab_a");
    Stab_v = prm.get_bool("Stab_v");

	theta_pl = prm.get_double("Theta pl");
	theta_Sa = prm.get_double("Theta Sa");
	theta_Sv = prm.get_double("Theta Sv");

	penalty_pl = prm.get_double("Penalty pl");
	penalty_Sa = prm.get_double("Penalty Sa");
	penalty_Sv = prm.get_double("Penalty Sv");

	penalty_pl_bdry = prm.get_double("Penalty pl boundary");
	penalty_Sa_bdry = prm.get_double("Penalty Sa boundary");
	penalty_Sv_bdry = prm.get_double("Penalty Sv boundary");

	project_to_RT0 = prm.get_bool("Project to RT0");
	project_only_kappa = prm.get_bool("Project to RT0 with kappa only");
	project_Darcy_with_gravity = prm.get_bool("Project Darcy with gravity");

	use_direct_solver = prm.get_bool("Use direct solver for linear systems");

	use_exact_pl_in_Sa = prm.get_bool("Use exact pl in Sa");
	use_exact_pl_in_Sv = prm.get_bool("Use exact pl in Sv");
	use_exact_pl_in_RT = prm.get_bool("Use exact pl in RT");
	use_exact_Sa_in_pl = prm.get_bool("Use exact Sa in pl");
	use_exact_Sa_in_Sv = prm.get_bool("Use exact Sa in Sv");
	use_exact_Sa_in_RT = prm.get_bool("Use exact Sa in RT");
	use_exact_Sv_in_pl = prm.get_bool("Use exact Sv in pl");
	use_exact_Sv_in_Sa = prm.get_bool("Use exact Sv in Sa");
	use_exact_Sv_in_RT = prm.get_bool("Use exact Sv in RT");

	artificial_visc_exp = prm.get_bool("Add explicit artificial viscosity");
	artificial_visc_imp = prm.get_bool("Add implicit artificial viscosity");
	art_visc_multiple_Sa = prm.get_double("Artificial viscosity multiple for Sa");

	two_phase = prm.get_bool("Two phase problem");

	prm.leave_subsection();

	prm.enter_subsection("Output parameters");
	print_vtk = prm.get_bool("print vtk");
	vtk_freq = prm.get_integer("Frequency for vtk printing");
	compute_errors_sol = prm.get_bool("Compute errors");
	compute_energy = prm.get_bool("Compute energy");
	prm.leave_subsection();

	prm.enter_subsection("Continuation from previous solution");
	output_sol = prm.get_bool("Output solution");
	output_sol_freq = prm.get_integer("Frequency for solution output");
	continue_solution = prm.get_bool("Continue solution");

	if(continue_solution)
		timestep_number = prm.get_integer("Time step number");
	else
		timestep_number = 2;

	prm.leave_subsection();


	time = timestep_number*time_step;
}
template <int dim>
void CoupledPressureSaturationProblem<dim>::create_kappa_abs_vector()
{
	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		if(cell->subdomain_id() == this_mpi_process)
		{
			double kappa_val = compute_kappa_value<dim>(cell);
			kappa_abs_vec[cell->global_active_cell_index()] = kappa_val;
		}
		kappa_abs_vec.compress(VectorOperation::insert);
	}
}

template <int dim>
void CoupledPressureSaturationProblem<dim>::setup_system()
{

    dof_handler.distribute_dofs(fe);
    dof_handler_RT.distribute_dofs(fe_RT);

    const std::vector<IndexSet> locally_owned_dofs_per_proc =
		  DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
	locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

	const std::vector<IndexSet> locally_owned_dofs_per_proc_RT =
		  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_RT);
	locally_owned_dofs_RT = locally_owned_dofs_per_proc_RT[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_RT, locally_relevant_dofs_RT);

	pl_solution.reinit(locally_owned_dofs, mpi_communicator);
	pl_solution_n.reinit(locally_owned_dofs, mpi_communicator);
	pl_solution_k.reinit(locally_owned_dofs, mpi_communicator);
	pl_solution_kplus1.reinit(locally_owned_dofs, mpi_communicator);
	pl_solution_nminus1.reinit(locally_owned_dofs, mpi_communicator);
	pl_solution_nminus2.reinit(locally_owned_dofs, mpi_communicator);


	Sa_solution.reinit(locally_owned_dofs, mpi_communicator);
	Sa_solution_n.reinit(locally_owned_dofs, mpi_communicator);
	Sa_solution_k.reinit(locally_owned_dofs, mpi_communicator);
	Sa_solution_kplus1.reinit(locally_owned_dofs, mpi_communicator);
	Sa_solution_nminus1.reinit(locally_owned_dofs, mpi_communicator);
	Sa_solution_nminus2.reinit(locally_owned_dofs, mpi_communicator);


	Sv_solution.reinit(locally_owned_dofs, mpi_communicator);
	Sv_solution_n.reinit(locally_owned_dofs, mpi_communicator);
	Sv_solution_k.reinit(locally_owned_dofs, mpi_communicator);
	Sv_solution_kplus1.reinit(locally_owned_dofs, mpi_communicator);
	Sv_solution_nminus1.reinit(locally_owned_dofs, mpi_communicator);
	Sv_solution_nminus2.reinit(locally_owned_dofs, mpi_communicator);
	pl_difference.reinit(locally_owned_dofs, mpi_communicator);
	Sa_difference.reinit(locally_owned_dofs, mpi_communicator);
	Sv_difference.reinit(locally_owned_dofs, mpi_communicator);

	totalDarcyvelocity_RT_Sa.reinit(locally_owned_dofs_RT, mpi_communicator);
	totalDarcyvelocity_RT_Sa_n.reinit(locally_owned_dofs_RT, mpi_communicator);

	totalDarcyvelocity_RT_Sv.reinit(locally_owned_dofs_RT, mpi_communicator);
	totalDarcyvelocity_RT_Sv_n.reinit(locally_owned_dofs_RT, mpi_communicator);

	dof_handler_dg0.distribute_dofs(fe_dg0);
	const std::vector<IndexSet> locally_owned_dofs_per_proc_dg0 =
		  DoFTools::locally_owned_dofs_per_subdomain(dof_handler_dg0);
	locally_owned_dofs_dg0 = locally_owned_dofs_per_proc_dg0[this_mpi_process];

	DoFTools::extract_locally_relevant_dofs(dof_handler_dg0, locally_relevant_dofs_dg0);

	kappa_abs_vec.reinit(locally_owned_dofs_dg0, mpi_communicator);

    constraints.close();
}

template <int dim>
void CoupledPressureSaturationProblem<dim>::compute_errors() const
{
    // PrescribedSolution::ExactSolution<dim> exact_solution;
	ExactLiquidPressure<dim> exact_solution_pressure;
	exact_solution_pressure.set_time(final_time);
    Vector<double> cellwise_errors_pl(triangulation.n_active_cells());

    ExactAqueousSaturation<dim> exact_solution_aqueous_saturation;
    exact_solution_aqueous_saturation.set_time(final_time);
	Vector<double> cellwise_errors_Sa(triangulation.n_active_cells());

    ExactVaporSaturation<dim> exact_solution_vapor_saturation;
    exact_solution_vapor_saturation.set_time(final_time);
	Vector<double> cellwise_errors_Sv(triangulation.n_active_cells());

    QTrapezoid<1>     q_trapez;
    QIterated<dim> quadrature(q_trapez, degree + 2);

    // With this, we can then let the library compute the errors and output
    // them to the screen:
    VectorTools::integrate_difference(dof_handler,
    								  pl_solution,
									  exact_solution_pressure,
									  cellwise_errors_pl,
                                      quadrature,
                                      VectorTools::L2_norm);
    const double pl_l2_error =
        VectorTools::compute_global_error(triangulation,
        								  cellwise_errors_pl,
                                          VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
    								  Sa_solution,
									  exact_solution_aqueous_saturation,
									  cellwise_errors_Sa,
                                      quadrature,
                                      VectorTools::L2_norm);
    const double Sa_l2_error =
        VectorTools::compute_global_error(triangulation,
        								  cellwise_errors_Sa,
                                          VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
    								  Sv_solution,
									  exact_solution_vapor_saturation,
									  cellwise_errors_Sv,
                                      quadrature,
                                      VectorTools::L2_norm);
    const double Sv_l2_error =
        VectorTools::compute_global_error(triangulation,
        								  cellwise_errors_Sv,
                                          VectorTools::L2_norm);


    pcout << "Errors: " << std::endl;
    pcout << "||e_pl||_L2 = " << pl_l2_error << std::endl;
    pcout << "||e_Sa||_L2 = " << Sa_l2_error << std::endl;
    pcout << "||e_Sv||_L2 = " << Sv_l2_error << std::endl;

    compute_energy_norm_error_pl();
}

template <int dim>
void CoupledPressureSaturationProblem<dim>::compute_energy_norm_error_pl() const
{
	ExactLiquidPressure<dim> exact_solution_pressure;
	exact_solution_pressure.set_time(final_time);

	PETScWrappers::MPI::Vector temp_pl_solution;
	temp_pl_solution.reinit(locally_owned_dofs,
							locally_relevant_dofs,
							mpi_communicator);

	temp_pl_solution = pl_solution;

	const QGauss<dim>     quadrature_formula(degree + 1);
	const QGauss<dim - 1> face_quadrature_formula(degree + 1);

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

	typename DoFHandler<dim>::active_cell_iterator
			  cell_DG = dof_handler.begin_active(),
			  endc = dof_handler.end();

	double cell_term = 0.0;
	double face_term = 0.0;

	for (; cell_DG != endc; ++cell_DG)
	{
		if (cell_DG->subdomain_id() == this_mpi_process)
		{
			fe_values_DG.reinit(cell_DG);

			unsigned int cell_index = cell_DG->active_cell_index();

			const auto &q_points = fe_values_DG.get_quadrature_points();
			const unsigned int n_q_points_DG = fe_values_DG.get_quadrature().size();
			const std::vector<double> &JxW = fe_values_DG.get_JxW_values();

			std::vector<double> pl_vals_cell(n_q_points_DG);
			fe_values_DG.get_function_values(temp_pl_solution, pl_vals_cell);

			std::vector<Tensor<1, dim>> grad_pl_cell(n_q_points_DG);
			fe_values_DG.get_function_gradients(temp_pl_solution, grad_pl_cell);

			Tensor<1, dim> grad_exact;

			double norm_square = 0;

			for (unsigned int q = 0; q < n_q_points_DG; ++q)
			{
				Tensor<1,dim> pl_grad = grad_pl_cell[q];
				grad_exact = exact_solution_pressure.gradient(q_points[q]);
				norm_square += (pl_grad - grad_exact).norm_square() * JxW[q];
			}

			cell_term += norm_square;

			for (const auto &face : cell_DG->face_iterators())
			{
				fe_face_values_DG.reinit(cell_DG, face);

				unsigned int face_num = cell_DG->face_iterator_to_index(face);

				const auto &q_points = fe_face_values_DG.get_quadrature_points();
				const unsigned int n_q_points_face_DG = fe_face_values_DG.get_quadrature().size();
				const std::vector<double> &JxW = fe_face_values_DG.get_JxW_values();

				if(face->at_boundary())
				{
					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					double difference_norm_square = 0.;
					for (unsigned int q = 0; q < n_q_points_face_DG; ++q)
					{
						double pl_value = pl_vals_face[q];

						const double diff = (exact_solution_pressure.value(q_points[q]) - pl_value);
						difference_norm_square += diff * diff * JxW[q];
					}

					face_term += (1.0/cell_DG->face(face_num)->measure()) * difference_norm_square;
				}
				else
				{
					typename DoFHandler<dim>::active_cell_iterator cell_DG_neighbor = cell_DG->neighbor(face_num);
					unsigned int neighbor_index = cell_DG->neighbor_index(face_num);

					if(cell_index >= neighbor_index)
						continue;

					fe_face_values_DG_neighbor.reinit(cell_DG_neighbor, face);

					std::vector<double> pl_vals_face(n_q_points_face_DG);
					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);

					std::vector<double> pl_vals_face_neighbor(n_q_points_face_DG);
					fe_face_values_DG_neighbor.get_function_values(temp_pl_solution, pl_vals_face_neighbor);

					double pl_jump_square = 0;
					for (unsigned int q = 0; q < n_q_points_face_DG; ++q)
					{
						double pl_value0 = pl_vals_face[q];
						double pl_value1 = pl_vals_face_neighbor[q];

						double jump_pl = pl_value0 - pl_value1;
						pl_jump_square += jump_pl * jump_pl * JxW[q];
					}

					face_term += (1.0/cell_DG->face(face_num)->measure()) * pl_jump_square;

				}
			}
		}
	}

     const double energy_error = std::sqrt(cell_term + face_term);

     pcout << "Triple norm error p_l: " << std::endl;
     pcout << "|||e_pl||| = " << energy_error << std::endl;
}

template <int dim>
void CoupledPressureSaturationProblem<dim>::output_vtk_initial_cond() const
{
	DataOutBase::VtkFlags flags;
	//flags.compression_level = DataOutBase::VtkFlags::best_speed;

    // To print kappa_abs vector
    kappa_abs_vec.update_ghost_values();

    DataOut<dim> data_kappa;

    data_kappa.attach_dof_handler(dof_handler_dg0);
    data_kappa.add_data_vector(kappa_abs_vec, "kappa", DataOut<dim>::type_dof_data);
    data_kappa.build_patches(mapping);

    data_kappa.set_flags(flags);
    const std::string filename_kappa=
            "kappa.vtu";
    data_kappa.write_vtu_in_parallel(filename_kappa, mpi_communicator);
    // end of printing kappa_abs vector

	pl_solution_nminus1.update_ghost_values();

	DataOut<dim> data_out_pl_nminus1;

	data_out_pl_nminus1.attach_dof_handler(dof_handler);
	data_out_pl_nminus1.add_data_vector(pl_solution_nminus1, "pl");
	data_out_pl_nminus1.build_patches(mapping);

	data_out_pl_nminus1.set_flags(flags);
	const std::string filename_pl_0 =
		        "solution_pl-000.vtu";
	data_out_pl_nminus1.write_vtu_in_parallel(filename_pl_0, mpi_communicator);

	pl_solution_n.update_ghost_values();

	DataOut<dim> data_out_pl_n;

	data_out_pl_n.attach_dof_handler(dof_handler);
	data_out_pl_n.add_data_vector(pl_solution_n, "pl");
	data_out_pl_n.build_patches(mapping);

	data_out_pl_n.set_flags(flags);
	const std::string filename_pl_1 =
		        "solution_pl-001.vtu";
	data_out_pl_n.write_vtu_in_parallel(filename_pl_1, mpi_communicator);

	// Sa
	Sa_solution_nminus1.update_ghost_values();

	DataOut<dim> data_out_Sa_nminus1;

	data_out_Sa_nminus1.attach_dof_handler(dof_handler);
	data_out_Sa_nminus1.add_data_vector(Sa_solution_nminus1, "Sa");
	data_out_Sa_nminus1.build_patches(mapping);

	data_out_Sa_nminus1.set_flags(flags);
	const std::string filename_Sa_0 =
				"solution_Sa-000.vtu";
	data_out_Sa_nminus1.write_vtu_in_parallel(filename_Sa_0, mpi_communicator);

	Sa_solution_n.update_ghost_values();

	DataOut<dim> data_out_Sa_n;

	data_out_Sa_n.attach_dof_handler(dof_handler);
	data_out_Sa_n.add_data_vector(Sa_solution_n, "Sa");
	data_out_Sa_n.build_patches(mapping);

	data_out_Sa_n.set_flags(flags);
	const std::string filename_Sa_1 =
				"solution_Sa-001.vtu";
	data_out_Sa_n.write_vtu_in_parallel(filename_Sa_1, mpi_communicator);

	// Sv
	if(!two_phase)
	{
		Sv_solution_nminus1.update_ghost_values();

		DataOut<dim> data_out_Sv_nminus1;

		data_out_Sv_nminus1.attach_dof_handler(dof_handler);
		data_out_Sv_nminus1.add_data_vector(Sv_solution_nminus1, "Sv");
		data_out_Sv_nminus1.build_patches(mapping);

		data_out_Sv_nminus1.set_flags(flags);
		const std::string filename_Sv_0 =
					"solution_Sv-000.vtu";
		data_out_Sv_nminus1.write_vtu_in_parallel(filename_Sv_0, mpi_communicator);

		Sv_solution_n.update_ghost_values();

		DataOut<dim> data_out_Sv_n;

		data_out_Sv_n.attach_dof_handler(dof_handler);
		data_out_Sv_n.add_data_vector(Sv_solution_n, "Sv");
		data_out_Sv_n.build_patches(mapping);

		data_out_Sv_n.set_flags(flags);
		const std::string filename_Sv_1 =
					"solution_Sv-001.vtu";
		data_out_Sv_n.write_vtu_in_parallel(filename_Sv_1, mpi_communicator);
	}
}

template <int dim>
void CoupledPressureSaturationProblem<dim>::output_vtk() const
{
	DataOutBase::VtkFlags flags;
	//flags.compression_level = DataOutBase::VtkFlags::best_speed;

	pl_solution.update_ghost_values();

	DataOut<dim> data_out_pl;

	data_out_pl.attach_dof_handler(dof_handler);
	data_out_pl.add_data_vector(pl_solution, "pl");
	data_out_pl.build_patches(mapping);

	data_out_pl.set_flags(flags);
	const std::string filename_pl =
		        "solution_pl-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
	data_out_pl.write_vtu_in_parallel(filename_pl, mpi_communicator);

	// Sa
	Sa_solution.update_ghost_values();

	DataOut<dim> data_out_Sa;

	data_out_Sa.attach_dof_handler(dof_handler);
	data_out_Sa.add_data_vector(Sa_solution, "Sa");
	data_out_Sa.build_patches(mapping);

	data_out_Sa.set_flags(flags);
	const std::string filename_Sa =
			"solution_Sa-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
	data_out_Sa.write_vtu_in_parallel(filename_Sa, mpi_communicator);

	// Sv
	if(!two_phase)
	{
		Sv_solution.update_ghost_values();

		DataOut<dim> data_out_Sv;

		data_out_Sv.attach_dof_handler(dof_handler);
		data_out_Sv.add_data_vector(Sv_solution, "Sv");
		data_out_Sv.build_patches(mapping);

		data_out_Sv.set_flags(flags);
		const std::string filename_Sv =
				"solution_Sv-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
		data_out_Sv.write_vtu_in_parallel(filename_Sv, mpi_communicator);
	}
}

template <int dim>
void CoupledPressureSaturationProblem<dim>::output_solution() const
{
	Vector<double> Sa_serial_nminus2(Sa_solution_nminus2), pl_serial_nminus2(pl_solution_nminus2);
	Vector<double> Sa_serial_nminus1(Sa_solution_nminus1), pl_serial_nminus1(pl_solution_nminus1);
	Vector<double> Sa_serial_n(Sa_solution_n), pl_serial_n(pl_solution_n);
	Vector<double> Sa_serial(Sa_solution), pl_serial(pl_solution);

	std::ofstream myfile_sa, myfile_pl;

	std::string myfilename_sa;
	std::string myfilename_pl;

	if(second_order_extrapolation || second_order_time_derivative)
	{
		myfilename_sa = "Sa_solution" + Utilities::int_to_string(timestep_number-3, 3);
		myfilename_pl = "pl_solution" + Utilities::int_to_string(timestep_number-3, 3);

		myfile_sa.open(myfilename_sa);
		myfile_pl.open(myfilename_pl);

		myfile_sa << Sa_serial_nminus2.size() << std::endl;
		myfile_pl << pl_serial_nminus2.size() << std::endl;

		for(unsigned int jj = 0; jj < Sa_serial_nminus2.size(); jj++)
		{
			myfile_sa << std::setprecision(16) << Sa_serial_nminus2[jj] << std::endl;
			myfile_pl << std::setprecision(16) << pl_serial_nminus2[jj] << std::endl;
		}

		myfile_sa.close();
		myfile_pl.close();
	}

	myfilename_sa = "Sa_solution" + Utilities::int_to_string(timestep_number-2, 3);
	myfilename_pl = "pl_solution" + Utilities::int_to_string(timestep_number-2, 3);

	myfile_sa.open(myfilename_sa);
	myfile_pl.open(myfilename_pl);

	myfile_sa << Sa_serial_nminus1.size() << std::endl;
	myfile_pl << pl_serial_nminus1.size() << std::endl;

	for(unsigned int jj = 0; jj < Sa_serial_nminus1.size(); jj++)
	{
		myfile_sa << std::setprecision(16) << Sa_serial_nminus1[jj] << std::endl;
		myfile_pl << std::setprecision(16) << pl_serial_nminus1[jj] << std::endl;
	}

	myfile_sa.close();
	myfile_pl.close();

	myfilename_sa = "Sa_solution" + Utilities::int_to_string(timestep_number-1, 3);
	myfilename_pl = "pl_solution" + Utilities::int_to_string(timestep_number-1, 3);

	myfile_sa.open(myfilename_sa);
	myfile_pl.open(myfilename_pl);

	myfile_sa << Sa_serial_n.size() << std::endl;
	myfile_pl << pl_serial_n.size() << std::endl;

	for(unsigned int jj = 0; jj < Sa_serial_n.size(); jj++)
	{
		myfile_sa << std::setprecision(16) << Sa_serial_n[jj] << std::endl;
		myfile_pl << std::setprecision(16) << pl_serial_n[jj] << std::endl;
	}

	myfile_sa.close();
	myfile_pl.close();

	myfilename_sa = "Sa_solution" + Utilities::int_to_string(timestep_number, 3);
	myfilename_pl = "pl_solution" + Utilities::int_to_string(timestep_number, 3);

	myfile_sa.open(myfilename_sa);
	myfile_pl.open(myfilename_pl);

	myfile_sa << Sa_serial.size() << std::endl;
	myfile_pl << pl_serial.size() << std::endl;

	for(unsigned int jj = 0; jj < Sa_serial.size(); jj++)
	{
		myfile_sa << std::setprecision(16) << Sa_serial[jj] << std::endl;
		myfile_pl << std::setprecision(16) << pl_serial[jj] << std::endl;
	}

	myfile_sa.close();
	myfile_pl.close();

	if(!two_phase)
	{
		Vector<double> Sv_serial_nminus2(Sv_solution_nminus2);
		Vector<double> Sv_serial_nminus1(Sv_solution_nminus1);
		Vector<double> Sv_serial_n(Sv_solution_n);
		Vector<double> Sv_serial(Sv_solution);

		std::ofstream myfile_sv;

		std::string myfilename_sv;

		if(second_order_extrapolation || second_order_time_derivative)
		{
			myfilename_sv = "Sv_solution" + Utilities::int_to_string(timestep_number-3, 3);

			myfile_sv.open(myfilename_sv);

			myfile_sv << Sv_serial_nminus2.size() << std::endl;

			for(unsigned int jj = 0; jj < Sv_serial_nminus2.size(); jj++)
				myfile_sv << std::setprecision(16) << Sv_serial_nminus2[jj] << std::endl;

			myfile_sv.close();
		}

		myfilename_sv = "Sv_solution" + Utilities::int_to_string(timestep_number-2, 3);

		myfile_sv.open(myfilename_sv);

		myfile_sv << Sv_serial_nminus1.size() << std::endl;

		for(unsigned int jj = 0; jj < Sv_serial_nminus1.size(); jj++)
			myfile_sv << std::setprecision(16) << Sv_serial_nminus1[jj] << std::endl;

		myfile_sv.close();

		myfilename_sv = "Sv_solution" + Utilities::int_to_string(timestep_number-1, 3);

		myfile_sv.open(myfilename_sv);

		myfile_sv << Sv_serial_n.size() << std::endl;

		for(unsigned int jj = 0; jj < Sv_serial_n.size(); jj++)
			myfile_sv << std::setprecision(16) << Sv_serial_n[jj] << std::endl;

		myfile_sv.close();

		myfilename_sv = "Sv_solution" + Utilities::int_to_string(timestep_number, 3);

		myfile_sv.open(myfilename_sv);

		myfile_sv << Sv_serial.size() << std::endl;

		for(unsigned int jj = 0; jj < Sv_serial.size(); jj++)
			myfile_sv << std::setprecision(16) << Sv_serial[jj] << std::endl;

		myfile_sv.close();
	}
}

template <int dim>
void CoupledPressureSaturationProblem<dim>::read_initial_condition()
{
	std::ifstream myfile_sa, myfile_pl;

	std::string myfilename_sa;
	std::string myfilename_pl;

	if(second_order_extrapolation || second_order_time_derivative)
	{
		myfilename_sa = "Sa_solution"
				+ Utilities::int_to_string(timestep_number-3, 3);
		myfilename_pl = "pl_solution"
				+ Utilities::int_to_string(timestep_number-3, 3);

		myfile_sa.open(myfilename_sa);
		myfile_pl.open(myfilename_pl);

		unsigned int size_sol;

		myfile_sa >> size_sol;
		myfile_pl >> size_sol;

		Vector<double> Sa_serial(size_sol), pl_serial(size_sol);

		for(unsigned int i = 0; i < size_sol; i++)
		{
			myfile_sa >> Sa_serial[i];
			myfile_pl >> pl_serial[i];
		}

		Sa_solution_nminus2 = Sa_serial;
		pl_solution_nminus2 = pl_serial;

		myfile_sa.close();
		myfile_pl.close();
	}

	myfilename_sa = "Sa_solution"
			+ Utilities::int_to_string(timestep_number-2, 3);
	myfilename_pl = "pl_solution"
			+ Utilities::int_to_string(timestep_number-2, 3);

	myfile_sa.open(myfilename_sa);
	myfile_pl.open(myfilename_pl);

	unsigned int size_sol;

	myfile_sa >> size_sol;
	myfile_pl >> size_sol;

	Vector<double> Sa_serial(size_sol), pl_serial(size_sol);

	for(unsigned int i = 0; i < size_sol; i++)
	{
		myfile_sa >> Sa_serial[i];
		myfile_pl >> pl_serial[i];
	}

	Sa_solution_nminus1 = Sa_serial;
	pl_solution_nminus1 = pl_serial;

	myfile_sa.close();
	myfile_pl.close();

	myfilename_sa = "Sa_solution"
			+ Utilities::int_to_string(timestep_number-1, 3);
	myfilename_pl = "pl_solution"
			+ Utilities::int_to_string(timestep_number-1, 3);

	myfile_sa.open(myfilename_sa);
	myfile_pl.open(myfilename_pl);

	myfile_sa >> size_sol;
	myfile_pl >> size_sol;


	for(unsigned int i = 0; i < size_sol; i++)
	{
		myfile_sa >> Sa_serial[i];
		myfile_pl >> pl_serial[i];
	}

	Sa_solution_n = Sa_serial;
	pl_solution_n = pl_serial;

	myfile_sa.close();
	myfile_pl.close();

	myfilename_sa = "Sa_solution"
			+ Utilities::int_to_string(timestep_number, 3);
	myfilename_pl = "pl_solution"
			+ Utilities::int_to_string(timestep_number, 3);

	myfile_sa.open(myfilename_sa);
	myfile_pl.open(myfilename_pl);

	myfile_sa >> size_sol;
	myfile_pl >> size_sol;

	for(unsigned int i = 0; i < size_sol; i++)
	{
		myfile_sa >> Sa_serial[i];
		myfile_pl >> pl_serial[i];
	}

	myfile_sa.close();
	myfile_pl.close();

	if(!two_phase)
	{
		std::ifstream myfile_sv;

		std::string myfilename_sv;

		if(second_order_extrapolation || second_order_time_derivative)
		{
			myfilename_sv = "Sv_solution"
					+ Utilities::int_to_string(timestep_number-3, 3);

			myfile_sv.open(myfilename_sv);

			unsigned int size_sol;

			myfile_sv >> size_sol;

			Vector<double> Sv_serial(size_sol);

			for(unsigned int i = 0; i < size_sol; i++)
				myfile_sv >> Sv_serial[i];

			Sv_solution_nminus2 = Sv_serial;

			myfile_sv.close();
		}

		myfilename_sv = "Sv_solution"
				+ Utilities::int_to_string(timestep_number-2, 3);

		myfile_sv.open(myfilename_sv);

		unsigned int size_sol;

		myfile_sv >> size_sol;

		Vector<double> Sv_serial(size_sol);

		for(unsigned int i = 0; i < size_sol; i++)
			myfile_sv >> Sv_serial[i];

		Sv_solution_nminus1 = Sv_serial;

		myfile_sv.close();

		myfilename_sv = "Sv_solution"
				+ Utilities::int_to_string(timestep_number-1, 3);

		myfile_sv.open(myfilename_sv);

		myfile_sv >> size_sol;


		for(unsigned int i = 0; i < size_sol; i++)
			myfile_sv >> Sv_serial[i];

		Sv_solution_n = Sv_serial;

		myfile_sv.close();

		myfilename_sa = "Sv_solution"
				+ Utilities::int_to_string(timestep_number, 3);

		myfile_sv.open(myfilename_sv);

		myfile_sv >> size_sol;

		for(unsigned int i = 0; i < size_sol; i++)
			myfile_sv >> Sv_serial[i];

		myfile_sv.close();
	}
}

template <int dim>
void CoupledPressureSaturationProblem<dim>::load_gmsh_mesh()
{
	GridIn<dim> gridin;

    gridin.attach_triangulation(triangulation);
    std::ifstream f("example.msh");
    gridin.read_msh(f);
}

template <int dim>
double CoupledPressureSaturationProblem<dim>::calculate_energy(double &real_energy)
{
	double gamma_a = 5.8275;
	double gamma_l = 0.5398;
	double gamma_al = 3.712;

	double phi = 0.2;

	ExactAqueousSaturation<dim> exact_sa;
	exact_sa.set_time(time);

//	AqueousPressure<dim> pressure_a;
	lambda_l<dim> lambda_l;
	lambda_a<dim> lambda_a;

	PETScWrappers::MPI::Vector temp_Sa_solution;
	temp_Sa_solution.reinit(locally_owned_dofs,
							locally_relevant_dofs,
							mpi_communicator);

	temp_Sa_solution = Sa_solution;

	PETScWrappers::MPI::Vector temp_pl_solution;
	temp_pl_solution.reinit(locally_owned_dofs,
							locally_relevant_dofs,
							mpi_communicator);

	temp_pl_solution = pl_solution;

	PETScWrappers::MPI::Vector temp_kappa;
	temp_kappa.reinit(locally_owned_dofs_dg0,
						  locally_relevant_dofs_dg0,
						  mpi_communicator);

	temp_kappa = kappa_abs_vec;

	const QGauss<dim>     quadrature_formula(degree + 1);
	const QGauss<dim - 1> face_quadrature_formula(degree + 1);

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

	typename DoFHandler<dim>::active_cell_iterator
			  cell_DG = dof_handler.begin_active(),
			  endc = dof_handler.end();

	double energy_val = 0.0;
	real_energy = 0.0;
	double source_val = 0.0;

	for (; cell_DG != endc; ++cell_DG)
	{
		if (cell_DG->subdomain_id() == this_mpi_process)
		{
			fe_values_DG.reinit(cell_DG);

			unsigned int cell_index = cell_DG->active_cell_index();

			const auto &q_points = fe_values_DG.get_quadrature_points();
			const unsigned int n_q_points_DG = fe_values_DG.get_quadrature().size();
			const std::vector<double> &JxW = fe_values_DG.get_JxW_values();

			std::vector<double> Sa_vals_cell(n_q_points_DG);
			fe_values_DG.get_function_values(temp_Sa_solution, Sa_vals_cell);

			for (unsigned int q = 0; q < n_q_points_DG; ++q)
			{
				double Sa_val = Sa_vals_cell[q];
				double real_Sa = exact_sa.value(q_points[q]);

				energy_val += gamma_a*Sa_val*(log(Sa_val) - 1.0) + gamma_l*(1.0 - Sa_val)*(log(1.0 - Sa_val) - 1.0)
						+ gamma_al*Sa_val*(1.0 - Sa_val);
				real_energy += gamma_a*real_Sa*(log(real_Sa) - 1.0) + gamma_l*(1.0 - real_Sa)*(log(1.0 - real_Sa) - 1.0)
										+ gamma_al*real_Sa*(1.0 - real_Sa);
			}

			double kappa = temp_kappa[cell_DG->active_cell_index()];

			for (const auto &face : cell_DG->face_iterators())
			{
				fe_face_values_DG.reinit(cell_DG, face);

				unsigned int face_num = cell_DG->face_iterator_to_index(face);

				const auto &q_points = fe_face_values_DG.get_quadrature_points();
				const unsigned int n_q_points_face_DG = fe_face_values_DG.get_quadrature().size();
				const std::vector<double> &JxW = fe_face_values_DG.get_JxW_values();

				if(face->at_boundary())
				{
					std::vector<double> pl_vals_face(n_q_points_face_DG);
					std::vector<double> Sa_vals_face(n_q_points_face_DG);
					std::vector<Tensor<1,dim>> pl_grads_face(n_q_points_face_DG);
					std::vector<Tensor<1,dim>> Sa_grads_face(n_q_points_face_DG);

					fe_face_values_DG.get_function_values(temp_pl_solution, pl_vals_face);
					fe_face_values_DG.get_function_values(temp_Sa_solution, Sa_vals_face);
					fe_face_values_DG.get_function_gradients(temp_pl_solution, pl_grads_face);
					fe_face_values_DG.get_function_gradients(temp_Sa_solution, Sa_grads_face);

					const std::vector<Tensor<1, dim>> &normals = fe_face_values_DG.get_normal_vectors();

					for (unsigned int q = 0; q < n_q_points_face_DG; ++q)
					{
						double pl_value = pl_vals_face[q];
						double Sa_value = Sa_vals_face[q];
						Tensor<1,dim> pl_grad = pl_grads_face[q];
						Tensor<1,dim> Sa_grad = Sa_grads_face[q];

//						double pa_value = pressure_a.value(pl_value, Sa_value, 0.0);
//						Tensor<1,dim> grad_pa = pressure_a.num_gradient(Sa_value, 0.0, pl_grad, Sa_grad, Sa_grad);

						double lambda_a_val = lambda_a.value(pl_value, Sa_value, 0.0);
						double lambda_l_val = lambda_l.value(pl_value, Sa_value, 0.0);

//						source_val += lambda_a_val*kappa*(grad_pa*normals[q])*pa_value;
						source_val += lambda_l_val*kappa*(pl_grad*normals[q])*pl_value;
					}


				}
			}
		}
	}

	real_energy *= phi;
	return phi*energy_val;// - source_val;
}

template <int dim>
void CoupledPressureSaturationProblem<dim>::run()
{



//	create_mesh();
	create_mesh<dim>(triangulation, ref_level, dirichlet_id_pl, dirichlet_id_sa, dirichlet_id_sv);
//	load_gmsh_mesh();

	pcout << "    Number of active cells:       "
		  << triangulation.n_active_cells() << " (by partition:";
	for (unsigned int p = 0; p < n_mpi_processes; ++p)
	  pcout << (p == 0 ? ' ' : '+')
			<< (GridTools::count_cells_with_subdomain_association(
				 triangulation, p));
	pcout << ")" << std::endl;

    setup_system();

    pcout << "    Number of degrees of freedom: " << dof_handler.n_dofs()
		  << " (by partition:";
	for (unsigned int p = 0; p < n_mpi_processes; ++p)
	  pcout << (p == 0 ? ' ' : '+')
			<< (DoFTools::count_dofs_with_subdomain_association(dof_handler,
																p));
	pcout << ")" << std::endl;

	if(create_initial_perturbation)
		create_initial_Sa_vector<dim>(triangulation, mpi_communicator, n_mpi_processes, this_mpi_process);

	if(continue_solution)
	{
		read_initial_condition();

		if(two_phase)
		{
			Sv_solution_nminus2 = 0.0;
			Sv_solution_nminus1 = 0.0;
			Sv_solution_n = 0.0;
			Sv_solution = 0.0;
		}

	}
	else
	{
		VectorTools::project(dof_handler,
							 constraints,
							 QGauss<dim>(fe.degree + 1),
							 InitialValuesLiquidPressure<dim>(),
							 pl_solution_n);

		VectorTools::project(dof_handler,
							 constraints,
							 QGauss<dim>(fe.degree + 1),
							 InitialValuesAqueousSaturation<dim>(),
							 Sa_solution_n);

		if(two_phase)
			Sv_solution_n = 0.0;
		else
			VectorTools::project(dof_handler,
								 constraints,
								 QGauss<dim>(fe.degree + 1),
								 InitialValuesVaporSaturation<dim>(),
								 Sv_solution_n);
	}

    pl_solution_nminus1 = pl_solution_n;
    Sa_solution_nminus1 = Sa_solution_n;
    Sv_solution_nminus1 = Sv_solution_n;

    InitialValuesLiquidPressure_dt<dim> pl_fcn;
	InitialValuesAqueousSaturation_dt<dim> Sa_fcn;
	InitialValuesVaporSaturation_dt<dim> Sv_fcn;

	pl_fcn.set_time(time_step);
	Sa_fcn.set_time(time_step);
	Sv_fcn.set_time(time_step);

	VectorTools::project(dof_handler,
						 constraints,
						 QGauss<dim>(fe.degree + 1),
						 pl_fcn,
						 pl_solution_n);

	VectorTools::project(dof_handler,
						 constraints,
						 QGauss<dim>(fe.degree + 1),
						 Sa_fcn,
						 Sa_solution_n);

	if(two_phase)
		Sv_solution_n = 0.0;
	else
		VectorTools::project(dof_handler,
							 constraints,
							 QGauss<dim>(fe.degree + 1),
							 Sv_fcn,
							 Sv_solution_n);

    pl_solution_nminus2 = pl_solution_nminus1;
    Sa_solution_nminus2 = Sa_solution_nminus1;
    Sv_solution_nminus2 = Sv_solution_nminus1;

    create_kappa_abs_vector();

    if(print_vtk)
    	output_vtk_initial_cond();

    unsigned int degreeRT;

    if(project_to_RT0)
    	degreeRT = 0;
    else
    	degreeRT = degree;

    // Vectors to save computation time
    Vector<double> assemble_time(final_time/time_step - timestep_number + 1);
    Vector<double> solver_time(final_time/time_step - timestep_number + 1);
    Vector<double> print_time(final_time/time_step - timestep_number + 1);
    Vector<double> RTproj_time(final_time/time_step - timestep_number + 1);
    Vector<int> iterations_per_time(final_time/time_step - timestep_number + 1);
    Vector<double> max_Sa_per_time(final_time/time_step - timestep_number + 1);
    Vector<double> min_Sa_per_time(final_time/time_step - timestep_number + 1);
    Vector<double> energy_per_time(final_time/time_step - timestep_number + 1);
    Timer timer(mpi_communicator);
    Timer total_timer(mpi_communicator);



    std::ofstream iter_file;
	iter_file.open("iterations_old");

	std::ofstream errors_file;
	errors_file.open("errors");

	std::ofstream energy_file;
	energy_file.open("energies");

    unsigned int index_time = 0;
    double total_time = 0.0;

	LiquidPressure::LiquidPressureProblem<dim> pl_problem(triangulation, degree,
    			theta_pl, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_Sa_in_pl,
    			use_exact_Sv_in_pl,
    			second_order_time_derivative, second_order_extrapolation,
				use_direct_solver, incompressible, implicit_time_pl,
    			kappa_abs_vec, mpi_communicator, n_mpi_processes, this_mpi_process);

	pl_problem.setup_system();

	AqueousSaturation::AqueousSaturationProblem<dim> Sa_problem(triangulation, degree,
					theta_Sa, penalty_Sa, penalty_Sa_bdry, dirichlet_id_sa, use_exact_pl_in_Sa,use_exact_Sv_in_Sa,
					second_order_time_derivative, second_order_extrapolation,
					use_direct_solver,Stab_a, incompressible, project_Darcy_with_gravity, artificial_visc_exp,
					artificial_visc_imp, art_visc_multiple_Sa,
					kappa_abs_vec, degreeRT, project_only_kappa,
					mpi_communicator, n_mpi_processes, this_mpi_process);

	Sa_problem.setup_system();

    bool rebuild_Sa_mat = true;
    for (; time <= final_time + 1.e-12; time += time_step, ++timestep_number)
    {
        pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

		// if midpoint, runs midpoint method with resepctive header files. else, runs bdf algorithm
		if(midpoint_method && incompressible)
		{
			        std::cerr << "ERROR! Midpoint method not working" << std::endl;
					std::abort();
		} 

		else if(midpoint_method && !incompressible){
			        std::cerr << "ERROR! Midpoint method must have incompressible to be true" << std::endl;
					std::abort();
		}
		else
		// run normal bdf scheme
		{
        timer.reset();
		timer.start();
        pl_problem.assemble_system_matrix_pressure(time_step, time, timestep_number,pl_solution_n,
													pl_solution_nminus1,pl_solution_nminus2, Sa_solution_n,
													Sa_solution_nminus1,Sa_solution_nminus2,Sv_solution_n,
													Sv_solution_nminus1,
													Sv_solution_nminus2);
        timer.stop();

        assemble_time[index_time] = timer.cpu_time();
		pcout << "Elapsed CPU time for pl assemble: " << timer.cpu_time() << " seconds." << std::endl;

        timer.reset();
		timer.start();
        pl_problem.solve_pressure();
        timer.stop();

        solver_time[index_time] = timer.cpu_time();
		pcout << "Elapsed CPU time for pl solve: " << timer.cpu_time() << " seconds." << std::endl;

        pl_solution = pl_problem.pl_solution;

		timer.reset();
		timer.start();
        if(project_to_RT0)
        {
        	if(project_Darcy_with_gravity)
        	{
				totalDarcyvelocity_RT_Sa = RT_Projection::compute_RT0_projection_with_gravity(triangulation,
						degree, theta_pl, time, time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
						use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation, incompressible,
						pl_solution, Sa_solution_n, Sa_solution_nminus1,
						Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec,
						true, project_only_kappa, mpi_communicator, n_mpi_processes, this_mpi_process);

				totalDarcyvelocity_RT_Sv = RT_Projection::compute_RT0_projection_with_gravity(triangulation,
						degree, theta_pl, time, time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
						use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation, incompressible,
						pl_solution, Sa_solution_n, Sa_solution_nminus1,
						Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec,
						false, project_only_kappa, mpi_communicator, n_mpi_processes, this_mpi_process);
        	}
        	else
        	{
				totalDarcyvelocity_RT_Sa = RT_Projection::compute_RT0_projection<dim>(triangulation, degree, theta_pl, time,
						time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
						use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation, incompressible,
						pl_solution, Sa_solution_n, Sa_solution_nminus1,
						Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec, project_only_kappa,
						mpi_communicator, n_mpi_processes, this_mpi_process);

				totalDarcyvelocity_RT_Sv = totalDarcyvelocity_RT_Sa;
        	}

        }
        else // Project to RTk
        {
        	if(project_Darcy_with_gravity)
			{
        		totalDarcyvelocity_RT_Sa = RT_Projection::compute_RTk_projection_with_gravity<dim>(triangulation, degree, theta_pl, time,
        							time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
        							use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation, incompressible,
        							pl_solution, Sa_solution_n, Sa_solution_nminus1,
        							Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec,
									true, project_only_kappa, mpi_communicator, n_mpi_processes, this_mpi_process);

        		totalDarcyvelocity_RT_Sv = RT_Projection::compute_RTk_projection_with_gravity<dim>(triangulation, degree, theta_pl, time,
									time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
									use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation, incompressible,
									pl_solution, Sa_solution_n, Sa_solution_nminus1,
									Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec,
									false, project_only_kappa, mpi_communicator, n_mpi_processes, this_mpi_process);
			}
        	else
        	{
					totalDarcyvelocity_RT_Sa = RT_Projection::compute_RTk_projection<dim>(triangulation, degree, theta_pl, time,
							time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
							use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation, incompressible,
							pl_solution, Sa_solution_n, Sa_solution_nminus1,
							Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec, project_only_kappa,
							mpi_communicator, n_mpi_processes, this_mpi_process);

				totalDarcyvelocity_RT_Sv = totalDarcyvelocity_RT_Sa;
        	}

        }

        timer.stop();
        RTproj_time[index_time] = timer.cpu_time();
        pcout << "Elapsed CPU time for RT Projection: " << timer.cpu_time() << " seconds." << std::endl;

		totalDarcyvelocity_RT_Sa_n = totalDarcyvelocity_RT_Sa;
		totalDarcyvelocity_RT_Sv_n = totalDarcyvelocity_RT_Sv;

		timer.reset();
		timer.start();
		Sa_problem.assemble_system_matrix_aqueous_saturation(time_step,time, timestep_number,								 rebuild_Sa_mat,
														pl_solution, pl_solution_n, pl_solution_nminus1,
														Sa_solution_n, Sa_solution_nminus1,
														Sv_solution_n, Sv_solution_nminus1,
														totalDarcyvelocity_RT_Sa);
		timer.stop();

		assemble_time[index_time] += timer.cpu_time();
		pcout << std::endl;
		pcout << "Elapsed CPU time for Sa assemble: " << timer.cpu_time() << " seconds." << std::endl;

		timer.reset();
		timer.start();
		Sa_problem.solve_aqueous_saturation(pl_solution);
		timer.stop();

		solver_time[index_time] += timer.cpu_time();
		pcout << "Elapsed CPU time for Sa solve: " << timer.cpu_time() << " seconds." << std::endl;

		Sa_solution = Sa_problem.Sa_solution;

		if(two_phase)
			Sv_solution = 0.0;
		else
		{
			VaporSaturation::VaporSaturationProblem<dim> Sv_problem(triangulation, degree, time_step,theta_n_time,
						theta_Sv, penalty_Sv, penalty_Sv_bdry, dirichlet_id_sv, use_exact_pl_in_Sv,
						use_exact_Sa_in_Sv, time, timestep_number,
						second_order_time_derivative, second_order_extrapolation,
						use_direct_solver, Stab_v, incompressible, project_Darcy_with_gravity,
						pl_solution, pl_solution_n, pl_solution_nminus1,pl_solution_kplus1,
						Sa_solution, Sa_solution_n, Sa_solution_nminus1, Sa_solution_kplus1,
						Sv_solution_n, Sv_solution_nminus1, Sv_solution_k,
						kappa_abs_vec, totalDarcyvelocity_RT_Sv, degreeRT, project_only_kappa,
						mpi_communicator, n_mpi_processes, this_mpi_process);

			timer.reset();
			timer.start();
			Sv_problem.assemble_system_matrix_vapor_saturation();
			timer.stop();

			assemble_time[index_time] += timer.cpu_time();
			pcout << std::endl;
			pcout << "Elapsed CPU time for Sv assemble: " << timer.cpu_time() << " seconds." << std::endl;

			timer.reset();
			timer.start();
			Sv_problem.solve_vapor_saturation();
			timer.stop();

			solver_time[index_time] += timer.cpu_time();
			pcout << "Elapsed CPU time for Sv solve: " << timer.cpu_time() << " seconds." << std::endl;

			Sv_solution = Sv_problem.Sv_solution;
		}

		pl_solution_nminus2 = pl_solution_nminus1;
        pl_solution_nminus1 = pl_solution_n;
        pl_solution_n = pl_solution;

        Sa_solution_nminus2 = Sa_solution_nminus1;
        Sa_solution_nminus1 = Sa_solution_n;
        Sa_solution_n = Sa_solution;

        Sv_solution_nminus2 = Sv_solution_nminus1;
        Sv_solution_nminus1 = Sv_solution_n;
        Sv_solution_n = Sv_solution;

        timer.reset();
		timer.start();
        if(print_vtk && timestep_number % vtk_freq == 0)
        	output_vtk();
        if(output_sol && timestep_number % output_sol_freq == 0)
        	output_solution();
        timer.stop();
        pcout << "Elapsed CPU time for output results: " << timer.cpu_time() << " seconds." << std::endl;
        pcout << std::endl;

        totalDarcyvelocity_RT_Sa_n = totalDarcyvelocity_RT_Sa;
        totalDarcyvelocity_RT_Sa = 0.0;

        totalDarcyvelocity_RT_Sv_n = totalDarcyvelocity_RT_Sv;
        totalDarcyvelocity_RT_Sv = 0.0;

//        QTrapezoid<1>     q_trapez;
//        QIterated<dim> quadrature(q_trapez, degree + 2);
//        PETScWrappers::MPI::Vector temp_Sa_solution;
//    	temp_Sa_solution.reinit(locally_owned_dofs,
//    							  locally_relevant_dofs,
//    							  mpi_communicator);
//        temp_Sa_solution = Sa_solution;

//        Vector<double> cellwise_errors_Sa(triangulation.n_active_cells());

        // With this, we can then let the library compute the errors and output
        // them to the screen:
//        VectorTools::integrate_difference(dof_handler,
//        								  temp_Sa_solution,
//    									  Functions::ZeroFunction<dim>(1),
//										  cellwise_errors_Sa,
//                                          quadrature,
//                                          VectorTools::Linfty_norm);
//
//        max_Sa_per_time[index_time] = cellwise_errors_Sa.linfty_norm();
////        cellwise_errors_Sa *= -1.0;
//        VectorTools::integrate_difference(dof_handler,
//        								  temp_Sa_solution,
//										  Functions::ConstantFunction<dim>(100.0,1),
//										  cellwise_errors_Sa,
//                                          quadrature,
//                                          VectorTools::Linfty_norm);
//
//        min_Sa_per_time[index_time] = 100.0 - cellwise_errors_Sa.linfty_norm();

        double real_energy, num_energy;

        if (compute_energy)
        {
        	num_energy = calculate_energy(real_energy);
        	energy_file << num_energy << " " << real_energy;
			energy_file << std::endl;
        }

//    	iter_file << min_Sa_per_time[index_time];
//    	iter_file << std::endl;
//    	iter_file << max_Sa_per_time[index_time];
//    	iter_file << std::endl;
//    	iter_file << num_energy;
//		iter_file << std::endl;
//		iter_file << real_energy;
//		iter_file << std::endl;


		}
		
		ExactLiquidPressure<dim> exact_solution_pressure;
		exact_solution_pressure.set_time(time);
		Vector<double> cellwise_errors_pl2(triangulation.n_active_cells());

		ExactAqueousSaturation<dim> exact_solution_aqueous_saturation;
		exact_solution_aqueous_saturation.set_time(time);
		Vector<double> cellwise_errors_Sa2(triangulation.n_active_cells());

		QTrapezoid<1>     q_trapez2;
		QIterated<dim> quadrature2(q_trapez2, degree + 2);

		VectorTools::integrate_difference(dof_handler,
										  pl_solution,
										  exact_solution_pressure,
										  cellwise_errors_pl2,
										  quadrature2,
										  VectorTools::L2_norm);
		const double pl_l2_error =
			VectorTools::compute_global_error(triangulation,
											  cellwise_errors_pl2,
											  VectorTools::L2_norm);

		VectorTools::integrate_difference(dof_handler,
										  Sa_solution,
										  exact_solution_aqueous_saturation,
										  cellwise_errors_Sa2,
										  quadrature2,
										  VectorTools::L2_norm);
		const double Sa_l2_error =
			VectorTools::compute_global_error(triangulation,
											  cellwise_errors_Sa2,
											  VectorTools::L2_norm);

		errors_file << pl_l2_error << "  " << Sa_l2_error << std::endl;

        index_time ++;
    }

    total_timer.stop();
    total_time = total_timer.cpu_time();

    // Save computation times
    std::ofstream time_file;
    time_file.open("times");

    time_file << "Average assemble time = " << assemble_time.mean_value() << std::endl;
    time_file << "Average solver time = " << solver_time.mean_value() << std::endl;
    time_file << "Average RT Projection time = " << RTproj_time.mean_value() << std::endl;
    total_time /= index_time;
    time_file << "Average total time = " << total_time << std::endl;

    time_file.close();

//    std::ofstream iter_file;
//	iter_file.open("iterations");

//	iter_file << min_Sa_per_time;
//	iter_file << std::endl;
//	iter_file << max_Sa_per_time;

	iter_file.close();

    if(compute_errors_sol)
    	compute_errors();

}
} 
int main(int argc, char *argv[])
{

    try
    {
    	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        ParameterHandler prm;
        CouplingPressureSaturation::ParameterReader  param(prm);
        param.read_parameters("parameters.prm");

        prm.enter_subsection("Time discretization parameters");

        double tf = prm.get_double("Final time");

        double init_delta_t = prm.get_double("Initial time step");

        prm.leave_subsection();

        double delta_t = init_delta_t;

        prm.enter_subsection("Spatial discretization parameters");
        int dimension = prm.get_integer("Dimension");
        const unsigned int fe_degree = prm.get_integer("Degree");
        const unsigned int init_refinement_level = prm.get_integer("Initial level of refinement");
        const unsigned int final_refinement_level = prm.get_integer("Final level of refinement");
        bool project_to_RT0 = prm.get_bool("Project to RT0");
        prm.leave_subsection();

        unsigned int degreeRT;
		if(project_to_RT0)
			degreeRT = 0;
		else
			degreeRT = fe_degree;

        for(unsigned int refinement_level = init_refinement_level; refinement_level <= final_refinement_level; refinement_level++)
        {
        	if(dimension == 2)
        	{
				CouplingPressureSaturation::CoupledPressureSaturationProblem<2> dgmethod(fe_degree, degreeRT,
						delta_t, tf, refinement_level, prm);
				dgmethod.run();
        	}
        	else if(dimension == 3)
        	{
				CouplingPressureSaturation::CoupledPressureSaturationProblem<3> dgmethod(fe_degree, degreeRT,
						delta_t, tf, refinement_level, prm);
				dgmethod.run();
        	}

			delta_t /= 2.0;
        }
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
