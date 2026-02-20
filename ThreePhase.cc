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

#include "RT_projection.hh"
#include "pl_problem.hh"
#include "Sa_problem.hh"
#include "Sv_problem.hh"

// utilitiy functions
#include "utilities/param_tpf_utilities.hh"

// aux functions and primary variables
#include "aux_primary.hh"


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

        //Initial variable at time tnminus2
	pl_fcn.set_time(-2*time_step);
	Sa_fcn.set_time(-2*time_step);
	Sv_fcn.set_time(-2*time_step);
        VectorTools::project(dof_handler,
                             constraints,
                             QGauss<dim>(fe.degree + 1),
                             pl_fcn,
                             pl_solution_nminus2);
        VectorTools::project(dof_handler,
                             constraints,
                             QGauss<dim>(fe.degree + 1),
                             Sa_fcn,
                             Sa_solution_nminus2);
        if(two_phase)
            Sv_solution_nminus2 = 0.0;
        else
            VectorTools::project(dof_handler,
                                 constraints,
                                 QGauss<dim>(fe.degree + 1),
                                 Sv_fcn,
                                 Sv_solution_nminus2);

        //Initial variable at time tnminus1
	pl_fcn.set_time(-time_step);
	Sa_fcn.set_time(-time_step);
	Sv_fcn.set_time(-time_step);
        VectorTools::project(dof_handler,
                             constraints,
                             QGauss<dim>(fe.degree + 1),
                             pl_fcn,
                             pl_solution_nminus1);
        VectorTools::project(dof_handler,
                             constraints,
                             QGauss<dim>(fe.degree + 1),
                             Sa_fcn,
                             Sa_solution_nminus1);
        if(two_phase)
            Sv_solution_nminus1 = 0.0;
        else
            VectorTools::project(dof_handler,
                                 constraints,
                                 QGauss<dim>(fe.degree + 1),
                                 Sv_fcn,
                                 Sv_solution_nminus1);

        //Initial variable at time tn
	pl_fcn.set_time(0*time_step);
	Sa_fcn.set_time(0*time_step);
	Sv_fcn.set_time(0*time_step);
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

	// new for tracking total time in loop
	Vector<double> total_loop_time(final_time/time_step - timestep_number + 1);

	

    Timer timer(mpi_communicator);
    Timer total_timer(mpi_communicator);

	// for keeping track of total time in loop
	Timer loop_timer(mpi_communicator);



    // std::ofstream iter_file;
	// iter_file.open("iterations_old");

	// std::ofstream errors_file;
	// errors_file.open("errors");

	// std::ofstream energy_file;
	// energy_file.open("energies");

    unsigned int index_time = 0;
    double total_time = 0.0;

	double loop_time = 0.0;


	loop_timer.reset();
	loop_timer.start();		

	LiquidPressure::LiquidPressureProblem<dim> pl_problem(triangulation, degree,
    			theta_pl, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_Sa_in_pl,
    			use_exact_Sv_in_pl,
    			second_order_time_derivative, second_order_extrapolation,
				use_direct_solver, Stab_pl, incompressible, implicit_time_pl,
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

	VaporSaturation::VaporSaturationProblem<dim> Sv_problem(triangulation, degree,
						theta_Sv, penalty_Sv, penalty_Sv_bdry, dirichlet_id_sv, use_exact_pl_in_Sv,
						use_exact_Sa_in_Sv, 
						second_order_time_derivative, second_order_extrapolation,
						use_direct_solver, Stab_v, incompressible, project_Darcy_with_gravity,
						kappa_abs_vec, degreeRT, project_only_kappa,
						mpi_communicator, n_mpi_processes, this_mpi_process);

	Sv_problem.setup_system();


    bool rebuild_pl_mat = true;
    bool rebuild_Sa_mat = true;
    bool rebuild_Sv_mat = true;
	
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
        pl_problem.assemble_system_matrix_pressure(time_step, time, timestep_number,rebuild_pl_mat,pl_solution_n,
													pl_solution_nminus1,pl_solution_nminus2, Sa_solution_n,
													Sa_solution_nminus1,Sa_solution_nminus2,Sv_solution_n,
													Sv_solution_nminus1,
													Sv_solution_nminus2);
        timer.stop();

        assemble_time[index_time] = timer.cpu_time();
		pcout << "Elapsed CPU time for pl assemble: " << timer.cpu_time() << " seconds."<< std::endl;

        timer.reset();
		timer.start();
        pl_problem.solve_pressure();
        timer.stop();

        solver_time[index_time] = timer.cpu_time();
		pcout << "Elapsed CPU time for pl solve: " << timer.cpu_time() << " seconds." << std::endl ;

        pl_solution = pl_problem.pl_solution;

		if(Stab_pl){
			rebuild_pl_mat = false;
		}

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
		Sa_problem.assemble_system_matrix_aqueous_saturation(time_step,time, timestep_number, rebuild_Sa_mat,
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

		if(Stab_a){
			rebuild_Sa_mat = false;
		}

		if(two_phase)
			Sv_solution = 0.0;
		else
		{
			timer.reset();
			timer.start();
			Sv_problem.assemble_system_matrix_vapor_saturation(time_step,time, timestep_number,rebuild_Sv_mat,
															pl_solution, pl_solution_n, pl_solution_nminus1,
															Sa_solution, Sa_solution_n, Sa_solution_nminus1,
															Sv_solution_n, Sv_solution_nminus1,
															totalDarcyvelocity_RT_Sv);
			timer.stop();

			assemble_time[index_time] += timer.cpu_time();
			pcout << std::endl;
			pcout << "Elapsed CPU time for Sv assemble: " << timer.cpu_time() << " seconds." << std::endl;

			timer.reset();
			timer.start();
			Sv_problem.solve_vapor_saturation(pl_solution);
			timer.stop();

			solver_time[index_time] += timer.cpu_time();
			pcout << "Elapsed CPU time for Sv solve: " << timer.cpu_time() << " seconds." << std::endl;

			Sv_solution = Sv_problem.Sv_solution;

			if(Stab_v){
				rebuild_Sv_mat = false;
			}
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

		loop_timer.stop();
		total_loop_time[loop_time] = loop_timer.cpu_time();
		pcout << "Elapsed CPU total time for the loop in time iteration: " << loop_timer.cpu_time() << " seconds."<< std::endl;

		loop_timer.reset();
		loop_timer.start();

        // double real_energy, num_energy;

        // if (compute_energy)
        // {
        // 	num_energy = calculate_energy(real_energy);
        // 	energy_file << num_energy << " " << real_energy;
		// 	energy_file << std::endl;
        // }


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

		//errors_file << pl_l2_error << "  " << Sa_l2_error << std::endl;

        index_time ++;
    }




    total_timer.stop();
    total_time = total_timer.cpu_time();
	pcout << "Total Time: " << total_time << " seconds." << std::endl;

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

		bool div4_delta_t = prm.get_bool("delta_t div 4");

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
			if(div4_delta_t){

				delta_t /= 4.0;
			}
			else
			{
				delta_t /= 2.0;
			}

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
