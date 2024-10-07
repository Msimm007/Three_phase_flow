//
// Created by mbsimmon on 9/10/24.
//

#ifndef THREEPHASE_MIDPOINT_METHOD_HH
#define THREEPHASE_MIDPOINT_METHOD_HH
//if(midpoint_method && incompressible)
//{
//// Initialize pl_k, Sa_k, Sv_k
////        pl_solution_k = (1 + theta_n_time)*pl_solution_n - theta_n_time*pl_solution_nminus1;
//pl_solution_k = pl_solution_n;
////        pl_solution_k.compress(VectorOperation::insert);
//pl_solution_k *= (1.0 + theta_n_time)/theta_n_time;
//pl_solution_k -= pl_solution_nminus1;
////        pl_solution_k.compress(VectorOperation::add);
//pl_solution_k *= theta_n_time;
//
////        Sa_solution_k = (1 + theta_n_time)*Sa_solution_n - theta_n_time*Sa_solution_nminus1;
//Sa_solution_k = Sa_solution_n;
//Sa_solution_k.compress(VectorOperation::insert);
//Sa_solution_k *= (1.0 + theta_n_time)/theta_n_time;
//Sa_solution_k -= Sa_solution_nminus1;
//Sa_solution_k.compress(VectorOperation::add);
//Sa_solution_k *= theta_n_time;
//
////        Sv_solution_k = (1 + theta_n_time)*Sv_solution_n - theta_n_time*Sv_solution_nminus1;
//Sv_solution_k = Sv_solution_n;
//Sv_solution_k.compress(VectorOperation::insert);
//Sv_solution_k *= (1.0 + theta_n_time)/theta_n_time;
//Sv_solution_k -= Sv_solution_nminus1;
//Sv_solution_k.compress(VectorOperation::add);
//Sv_solution_k *= theta_n_time;
//
//unsigned int num_iter = 1;
//
//for (unsigned int kk = 0; kk < 100; kk++)
//{
//LiquidPressure::LiquidPressureProblem<dim> pl_problem(triangulation, degree, time_step,
//                                                      theta_pl, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_Sa_in_pl,
//                                                      use_exact_Sv_in_pl, time, timestep_number,
//                                                      second_order_time_derivative, second_order_extrapolation,
//                                                      use_direct_solver, Stab_t, incompressible, implicit_time_pl,
//                                                      pl_solution_n, pl_solution_nminus1,
//                                                      pl_solution_nminus2,
//                                                      Sa_solution_n, Sa_solution_nminus1,
//                                                      Sa_solution_nminus2,
//                                                      Sv_solution_n, Sv_solution_nminus1,
//                                                      Sv_solution_nminus2,
//                                                      kappa_abs_vec, mpi_communicator, n_mpi_processes, this_mpi_process);
//timer.reset();
//timer.start();
//pl_problem.assemble_system_matrix_pressure();
//timer.stop();
//assemble_time[index_time] = timer.cpu_time();
////			pcout << "Elapsed CPU time for pl assemble: " << timer.cpu_time() << " seconds." << std::endl;
//
//timer.reset();
//timer.start();
//pl_problem.solve_pressure();
//timer.stop();
////
////			solver_time[index_time] = timer.cpu_time();
//////			pcout << "Elapsed CPU time for pl solve: " << timer.cpu_time() << " seconds." << std::endl;
//
//pl_solution_kplus1 = pl_problem.pl_solution;
////        	pl_solution_kplus1 = pl_solution_k;
//timer.reset();
//timer.start();
//if(project_to_RT0)
//{
//if(project_Darcy_with_gravity)
//{
////				totalDarcyvelocity_RT_Sa = RT_Projection::compute_RT0_projection_with_gravity(triangulation,
////						degree, theta_pl, time, time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
////						use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation,
////						pl_solution, Sa_solution_n, Sa_solution_nminus1,
////						Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec,
////						true, project_only_kappa, mpi_communicator, n_mpi_processes, this_mpi_process);
////
////				totalDarcyvelocity_RT_Sv = RT_Projection::compute_RT0_projection_with_gravity(triangulation,
////						degree, theta_pl, time, time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
////						use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation,
////						pl_solution, Sa_solution_n, Sa_solution_nminus1,
////						Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec,
////						false, project_only_kappa, mpi_communicator, n_mpi_processes, this_mpi_process);
//}
//else
//{
//totalDarcyvelocity_RT_Sa = RT_Projection::compute_RT0_projection<dim>(triangulation, degree, theta_pl, time,
//                                                                      time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
//                                                                      use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation, incompressible,
//                                                                      pl_solution, Sa_solution_n, Sa_solution_nminus1,
//                                                                      Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec, project_only_kappa,
//                                                                      mpi_communicator, n_mpi_processes, this_mpi_process);
//
//totalDarcyvelocity_RT_Sv = totalDarcyvelocity_RT_Sa;
//}
//
//}
//else // Project to RTk
//{
//if(project_Darcy_with_gravity)
//{
////	        		totalDarcyvelocity_RT_Sa = RT_Projection::compute_RTk_projection_with_gravity<dim>(triangulation, degree, theta_pl, time,
////	        							time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
////	        							use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation,
////	        							pl_solution, Sa_solution_n, Sa_solution_nminus1,
////	        							Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec,
////										true, project_only_kappa, mpi_communicator, n_mpi_processes, this_mpi_process);
////
////	        		totalDarcyvelocity_RT_Sv = RT_Projection::compute_RTk_projection_with_gravity<dim>(triangulation, degree, theta_pl, time,
////										time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
////										use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation,
////										pl_solution, Sa_solution_n, Sa_solution_nminus1,
////										Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec,
////										false, project_only_kappa, mpi_communicator, n_mpi_processes, this_mpi_process);
//}
//else
//{
//totalDarcyvelocity_RT_Sa = RT_Projection::compute_RTk_projection<dim>(triangulation, degree, theta_pl, time,
//                                                                      time_step, penalty_pl, penalty_pl_bdry, dirichlet_id_pl, use_exact_pl_in_RT,
//                                                                      use_exact_Sa_in_RT, use_exact_Sv_in_RT, second_order_extrapolation, incompressible,
//                                                                      pl_solution, Sa_solution_n, Sa_solution_nminus1,
//                                                                      Sv_solution_n, Sv_solution_nminus1, kappa_abs_vec, project_only_kappa,
//                                                                      mpi_communicator, n_mpi_processes, this_mpi_process);
//
//totalDarcyvelocity_RT_Sv = totalDarcyvelocity_RT_Sa;
//}
//
//}
//
//timer.stop();
//RTproj_time[index_time] = timer.cpu_time();
////			pcout << "Elapsed CPU time for RT Projection: " << timer.cpu_time() << " seconds." << std::endl;
//
//totalDarcyvelocity_RT_Sa_n = totalDarcyvelocity_RT_Sa;
//totalDarcyvelocity_RT_Sv_n = totalDarcyvelocity_RT_Sv;
//
//AqueousSaturationMidpoint::AqueousSaturationProblem_midpoint<dim> Sa_problem_midpoint(triangulation, degree, time_step, theta_n_time,
//                                                                                      theta_Sa, penalty_Sa, penalty_Sa_bdry, dirichlet_id_sa, use_exact_pl_in_Sa,
//                                                                                      use_exact_Sv_in_Sa, time_theta, timestep_number,
//                                                                                      second_order_time_derivative, second_order_extrapolation,
//                                                                                      use_direct_solver, Stab_a, project_Darcy_with_gravity, artificial_visc_exp,
//                                                                                      artificial_visc_imp, art_visc_multiple_Sa,
//                                                                                      pl_solution_kplus1,
//                                                                                      Sa_solution_k, Sa_solution_n,
//                                                                                      Sv_solution_k,
//                                                                                      kappa_abs_vec, totalDarcyvelocity_RT_Sa, degreeRT, project_only_kappa,
//                                                                                      mpi_communicator, n_mpi_processes, this_mpi_process);
//
//timer.reset();
//timer.start();
//Sa_problem_midpoint.assemble_system_matrix_aqueous_saturation();
//timer.stop();
//
//assemble_time[index_time] += timer.cpu_time();
////			pcout << std::endl;
////			pcout << "Elapsed CPU time for Sa assemble: " << timer.cpu_time() << " seconds." << std::endl;
//
//timer.reset();
//timer.start();
//Sa_problem_midpoint.solve_aqueous_saturation();
//timer.stop();
//
//solver_time[index_time] += timer.cpu_time();
////			pcout << "Elapsed CPU time for Sa solve: " << timer.cpu_time() << " seconds." << std::endl;
//
//Sa_solution_kplus1 = Sa_problem_midpoint.Sa_solution;
//Sa_solution_kplus1.compress(VectorOperation::insert);
//
//if(two_phase)
//Sv_solution = 0.0;
//else
//{
//VaporSaturationMidpoint::VaporSaturationProblem_midpoint<dim> Sv_problem_midpoint(triangulation, degree, time_step, theta_n_time,
//                                                                                  theta_Sv, penalty_Sv, penalty_Sv_bdry, dirichlet_id_sv, use_exact_pl_in_Sv,
//                                                                                  use_exact_Sa_in_Sv, time_theta, timestep_number,
//                                                                                  second_order_time_derivative, second_order_extrapolation,
//                                                                                  use_direct_solver, Stab_v, project_Darcy_with_gravity,
//                                                                                  pl_solution_kplus1,
//                                                                                  Sa_solution_kplus1,
//                                                                                  Sv_solution_k, Sv_solution_n,
//                                                                                  kappa_abs_vec, totalDarcyvelocity_RT_Sv, degreeRT, project_only_kappa,
//                                                                                  mpi_communicator, n_mpi_processes, this_mpi_process);
//
//timer.reset();
//timer.start();
//Sv_problem_midpoint.assemble_system_matrix_vapor_saturation();
//timer.stop();
//
//assemble_time[index_time] += timer.cpu_time();
////				pcout << std::endl;
////				pcout << "Elapsed CPU time for Sv assemble: " << timer.cpu_time() << " seconds." << std::endl;
//
//timer.reset();
//timer.start();
//Sv_problem_midpoint.solve_vapor_saturation();
//timer.stop();
//
//solver_time[index_time] += timer.cpu_time();
////				pcout << "Elapsed CPU time for Sv solve: " << timer.cpu_time() << " seconds." << std::endl;
//
//Sv_solution_kplus1 = Sv_problem_midpoint.Sv_solution;
//}
//
//// Check convergence
//pl_difference = pl_solution_kplus1;
//pl_difference.compress(VectorOperation::insert);
//pl_difference -= pl_solution_k;
//pl_difference.compress(VectorOperation::add);
//
//Sa_difference = Sa_solution_kplus1;
//Sa_difference.compress(VectorOperation::insert);
//Sa_difference -= Sa_solution_k;
//Sa_difference.compress(VectorOperation::add);
//
//Sv_difference = Sv_solution_kplus1;
//Sv_difference.compress(VectorOperation::insert);
//Sv_difference -= Sv_solution_k;
//Sv_difference.compress(VectorOperation::add);
//
//double pl_norm = pl_difference.l2_norm()/pl_solution_kplus1.l2_norm();
//double Sa_norm = Sa_difference.l2_norm()/Sa_solution_kplus1.l2_norm();
//double Sv_norm = 0.0;
//if(!two_phase)
//Sv_norm = Sv_difference.l2_norm()/Sv_solution_kplus1.l2_norm();
//
//pcout << "pl_norm = " << pl_norm << " Sa_norm = " << Sa_norm << " Sv_norm = " << Sv_norm << std::endl;
//if(pl_norm < 1.e-5 && Sa_norm < 1.e-5 && Sv_norm < 1.e-5)
//break;
////			pl_difference = pl_solution_kplus1;
////			pl_difference.compress(VectorOperation::insert);
////			pl_difference = pl_solution_kplus1;
////						pl_difference.compress(VectorOperation::insert);
//
//// Update values
//pl_solution_k = pl_solution_kplus1;
//pl_solution_k.compress(VectorOperation::insert);
//Sa_solution_k = Sa_solution_kplus1;
//Sa_solution_k.compress(VectorOperation::insert);
//Sv_solution_k = Sv_solution_kplus1;
//Sv_solution_k.compress(VectorOperation::insert);
//num_iter++;
//
//
//iterations_per_time[index_time] = num_iter;
//
//// Forward Euler step
//pl_solution = pl_solution_kplus1;
//pl_solution.compress(VectorOperation::insert);
////        if(theta_n_time < 1.0)
////        {
////			pl_solution *= 1.0/(1.0 - theta_n_time);
////			pl_solution -= pl_solution_n;
////			pl_solution.compress(VectorOperation::add);
////			pl_solution *= (1.0 - theta_n_time)/theta_n_time;
////        }
//
////        Sa_solution = 0.0;
//Sa_solution = Sa_solution_kplus1;
////        Sa_solution.add(1.0/theta_n_time, Sa_solution_kplus1);
////        Sa_solution = Sa_solution_kplus1;
//Sa_solution.compress(VectorOperation::insert);
//if(theta_n_time < 1.0)
//{
//Sa_solution *= 1.0/(1.0 - theta_n_time);
//Sa_solution -= Sa_solution_n;
//Sa_solution.compress(VectorOperation::add);
//Sa_solution *= (1.0 - theta_n_time)/theta_n_time;
////        	Sa_solution.add((theta_n_time-1.0)/theta_n_time, Sa_solution_n);
//}
//
//Sv_solution = Sv_solution_kplus1;
//Sv_solution.compress(VectorOperation::insert);
//if(theta_n_time < 1.0)
//{
//Sv_solution *= 1.0/(1.0 - theta_n_time);
//Sv_solution -= Sv_solution_n;
//Sv_solution.compress(VectorOperation::add);
//Sv_solution *= (1.0 - theta_n_time)/theta_n_time;
//}
//
////        pl_solution = pl_solution_kplus1;
////        Sa_solution = Sa_solution_kplus1;
////        Sv_solution = Sv_solution_kplus1;
//
//pl_solution_nminus1 = pl_solution_n;
//pl_solution_nminus1.compress(VectorOperation::insert);
//Sa_solution_nminus1 = Sa_solution_n;
//Sa_solution_nminus1.compress(VectorOperation::insert);
//Sv_solution_nminus1 = Sv_solution_n;
//Sv_solution_nminus1.compress(VectorOperation::insert);
//
//pl_solution_n = pl_solution;
//pl_solution_n.compress(VectorOperation::insert);
//Sa_solution_n = Sa_solution;
//Sa_solution_n.compress(VectorOperation::insert);
//Sv_solution_n = Sv_solution;
//Sv_solution_n.compress(VectorOperation::insert);
//
//timer.reset();
//timer.start();
//if(print_vtk && timestep_number % vtk_freq == 0)
//output_vtk();
//if(output_sol && timestep_number % output_sol_freq == 0)
//output_solution();
//timer.stop();
////        pcout << "Elapsed CPU time for output results: " << timer.cpu_time() << " seconds." << std::endl;
////        pcout << std::endl;
//
//totalDarcyvelocity_RT_Sa_n = totalDarcyvelocity_RT_Sa;
//totalDarcyvelocity_RT_Sa = 0.0;
//
//totalDarcyvelocity_RT_Sv_n = totalDarcyvelocity_RT_Sv;
//totalDarcyvelocity_RT_Sv = 0.0;
//
//QTrapezoid<1>     q_trapez;
//QIterated<dim> quadrature(q_trapez, degree + 2);
//PETScWrappers::MPI::Vector temp_Sa_solution;
//temp_Sa_solution.reinit(locally_owned_dofs,
//        locally_relevant_dofs,
//        mpi_communicator);
//temp_Sa_solution = Sa_solution;
//
//Vector<double> cellwise_errors_Sa(triangulation.n_active_cells());
//
//// With this, we can then let the library compute the errors and output
//// them to the screen:
//VectorTools::integrate_difference(dof_handler,
//        temp_Sa_solution,
//        Functions::ZeroFunction<dim>(1),
//        cellwise_errors_Sa,
//        quadrature,
//        VectorTools::Linfty_norm);
//
//max_Sa_per_time[index_time] = cellwise_errors_Sa.linfty_norm();
////        cellwise_errors_Sa *= -1.0;
//VectorTools::integrate_difference(dof_handler,
//        temp_Sa_solution,
//        Functions::ConstantFunction<dim>(100.0,1),
//        cellwise_errors_Sa,
//        quadrature,
//        VectorTools::Linfty_norm);
//
//min_Sa_per_time[index_time] = 100.0 - cellwise_errors_Sa.linfty_norm();
//
//double real_energy;
//
//double num_energy = calculate_energy(real_energy);
//energy_file << num_energy << " " << real_energy;
//energy_file << std::endl;
//
//iter_file << iterations_per_time[index_time] << std::endl;
////        iter_file << min_Sa_per_time[index_time];
////		iter_file << std::endl;
////		iter_file << max_Sa_per_time[index_time];
////		iter_file << std::endl;
////		iter_file << energy_per_time[index_time];
////		iter_file << std::endl;
////		iter_file << real_energy;
////		iter_file << std::endl;
//
////        for(unsigned int ii = 1; ii < cellwise_errors_Sa.size(); ii++)
////        {
////        	if(cellwise_errors_Sa[ii] < min_Sa_per_time[index_time])
////        		min_Sa_per_time[index_time] = cellwise_errors_Sa[ii];
//}
//}
#endif //THREEPHASE_MIDPOINT_METHOD_HH
