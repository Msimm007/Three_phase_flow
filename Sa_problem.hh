#ifndef SA_PROBLEM_HH
#define SA_PROBLEM_HH

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

namespace AqueousSaturation
{
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

    using namespace dealii;

    template <int dim>
    class AqueousSaturationProblem
    {
    public:
        AqueousSaturationProblem(Triangulation<dim, dim> &triangulation_,
                                 const unsigned int degree_, double time_step_, double theta_Sa_, double penalty_Sa_,
                                 double penalty_Sa_bdry_, std::vector<unsigned int> dirichlet_id_sa_, bool use_exact_pl_in_Sa_,
                                 bool use_exact_Sv_in_Sa_, double time_, unsigned int timestep_number_,
                                 bool second_order_time_derivative_, bool second_order_extrapolation_,
                                 bool use_direct_solver_,bool Stab_a_, bool incompressible_, bool project_Darcy_with_gravity_,
                                 bool artificial_visc_exp_, bool artificial_visc_imp_,
                                 double art_visc_multiple_Sa_,
                                 PETScWrappers::MPI::Vector pl_solution_, PETScWrappers::MPI::Vector pl_solution_n_,
                                 PETScWrappers::MPI::Vector pl_solution_nminus1_,
                                 PETScWrappers::MPI::Vector Sa_solution_n_, PETScWrappers::MPI::Vector Sa_solution_nminus1_,
                                 PETScWrappers::MPI::Vector Sv_solution_n_, PETScWrappers::MPI::Vector Sv_solution_nminus1_,
                                 PETScWrappers::MPI::Vector kappa_abs_vec_, PETScWrappers::MPI::Vector totalDarcyvelocity_RT_,
                                 const unsigned int degreeRT_, bool project_only_kappa_,
                                 MPI_Comm mpi_communicator_, const unsigned int n_mpi_processes_, const unsigned int this_mpi_process_);

        void assemble_system_matrix_aqueous_saturation();
        void assemble_rhs_aqueous_saturation();

        // modified so that solve function takes in an argument.
        void solve_aqueous_saturation(PETScWrappers::MPI::SparseMatrix &mat);


        // storing matrix for use in main file
        PETScWrappers::MPI::SparseMatrix stored_matrix;

        PETScWrappers::MPI::Vector Sa_solution;
    private:
        void setup_mat();
        void setup_rhs();

        parallel::shared::Triangulation<dim>   triangulation;
        const MappingQ1<dim> mapping;

        const QGauss<dim>     quadrature;
        const QGauss<dim - 1> face_quadrature;

        using ScratchData = MeshWorker::ScratchData<dim>;

        // We want to use DG elements
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

        PETScWrappers::MPI::SparseMatrix system_matrix_aqueous_saturation;
        PETScWrappers::MPI::Vector right_hand_side_aqueous_saturation;

        PETScWrappers::MPI::Vector pl_solution;
        PETScWrappers::MPI::Vector pl_solution_n;
        PETScWrappers::MPI::Vector pl_solution_nminus1;

        PETScWrappers::MPI::Vector Sa_solution_n;
        PETScWrappers::MPI::Vector Sa_solution_nminus1;

        PETScWrappers::MPI::Vector Sv_solution_n;
        PETScWrappers::MPI::Vector Sv_solution_nminus1;

        FE_DGQ<dim> fe_dg0;
        DoFHandler<dim> dof_handler_dg0;
        IndexSet locally_owned_dofs_dg0;
        IndexSet locally_relevant_dofs_dg0;
        PETScWrappers::MPI::Vector kappa_abs_vec;

        double 		 time_step;
        double       time;
        unsigned int timestep_number;

        double penalty_Sa;
        double penalty_Sa_bdry;

        double theta_Sa;

        std::vector<unsigned int> dirichlet_id_sa;

        bool Stab_a;
        bool incompressible;
        bool second_order_time_derivative;
        bool second_order_extrapolation;

        bool use_direct_solver;

        bool project_Darcy_with_gravity;
        bool project_only_kappa;

        bool use_exact_pl_in_Sa;
        bool use_exact_Sv_in_Sa;

        bool artificial_visc_exp;
        bool artificial_visc_imp;
        double art_visc_multiple_Sa;

        PETScWrappers::MPI::Vector totalDarcyvelocity_RT;

        const unsigned int degreeRT;
        FE_RaviartThomas<dim> fe_RT;
        DoFHandler<dim> dof_handler_RT;

        AffineConstraints<double> constraints;


    };


    template <int dim>
    AqueousSaturationProblem<dim>::AqueousSaturationProblem(

            Triangulation<dim, dim> &triangulation_,
            const unsigned int degree_, double time_step_, double theta_Sa_, double penalty_Sa_,
            double penalty_Sa_bdry_, std::vector<unsigned int> dirichlet_id_sa_, bool use_exact_pl_in_Sa_,
            bool use_exact_Sv_in_Sa_, double time_, unsigned int timestep_number_,
            bool second_order_time_derivative_, bool second_order_extrapolation_,
            bool use_direct_solver_, bool Stab_a_, bool incompressible_, bool project_Darcy_with_gravity_,
            bool artificial_visc_exp_, bool artificial_visc_imp_,
            double art_visc_multiple_Sa_,
            PETScWrappers::MPI::Vector pl_solution_, PETScWrappers::MPI::Vector pl_solution_n_,
            PETScWrappers::MPI::Vector pl_solution_nminus1_,
            PETScWrappers::MPI::Vector Sa_solution_n_, PETScWrappers::MPI::Vector Sa_solution_nminus1_,
            PETScWrappers::MPI::Vector Sv_solution_n_, PETScWrappers::MPI::Vector Sv_solution_nminus1_,
            PETScWrappers::MPI::Vector kappa_abs_vec_, PETScWrappers::MPI::Vector totalDarcyvelocity_RT_,
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
            , time_step(time_step_)
            , theta_Sa(theta_Sa_)
            , penalty_Sa(penalty_Sa_)
            , penalty_Sa_bdry(penalty_Sa_bdry_)
            , dirichlet_id_sa(dirichlet_id_sa_)
            , use_exact_pl_in_Sa(use_exact_pl_in_Sa_)
            , use_exact_Sv_in_Sa(use_exact_Sv_in_Sa_)
            , time(time_)
            , timestep_number(timestep_number_)
            , second_order_time_derivative(second_order_time_derivative_)
            , second_order_extrapolation(second_order_extrapolation_)
            , Stab_a(Stab_a_)
            , incompressible(incompressible_)
            , use_direct_solver(use_direct_solver_)
            , project_Darcy_with_gravity(project_Darcy_with_gravity_)
            , project_only_kappa(project_only_kappa_)
            , artificial_visc_exp(artificial_visc_exp_)
            , artificial_visc_imp(artificial_visc_imp_)
            , art_visc_multiple_Sa(art_visc_multiple_Sa_)
            , pl_solution(pl_solution_)
            , pl_solution_n(pl_solution_n_)
            , pl_solution_nminus1(pl_solution_nminus1_)
            , Sa_solution_n(Sa_solution_n_)
            , Sa_solution_nminus1(Sa_solution_nminus1_)
            , Sv_solution_n(Sv_solution_n_)
            , Sv_solution_nminus1(Sv_solution_nminus1_)
            , kappa_abs_vec(kappa_abs_vec_)
            , totalDarcyvelocity_RT(totalDarcyvelocity_RT_)
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
    void AqueousSaturationProblem<dim>::setup_mat()
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

        system_matrix_aqueous_saturation.reinit(locally_owned_dofs,
                                                locally_owned_dofs,
                                                sparsity_pattern,
                                                mpi_communicator);

        Sa_solution.reinit(locally_owned_dofs, mpi_communicator);

        const std::vector<IndexSet> locally_owned_dofs_per_proc_RT =
                DoFTools::locally_owned_dofs_per_subdomain(dof_handler_RT);
        locally_owned_dofs_RT = locally_owned_dofs_per_proc_RT[this_mpi_process];

        DoFTools::extract_locally_relevant_dofs(dof_handler_RT, locally_relevant_dofs_RT);

        dof_handler_dg0.distribute_dofs(fe_dg0);
        const std::vector<IndexSet> locally_owned_dofs_per_proc_dg0 =
                DoFTools::locally_owned_dofs_per_subdomain(dof_handler_dg0);
        locally_owned_dofs_dg0 = locally_owned_dofs_per_proc_dg0[this_mpi_process];

        DoFTools::extract_locally_relevant_dofs(dof_handler_dg0, locally_relevant_dofs_dg0);
    }

    template <int dim>
    void AqueousSaturationProblem<dim>::setup_rhs()
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

        // only reinit RHS as opposed to both matrix and rhs
        right_hand_side_aqueous_saturation.reinit(locally_owned_dofs, mpi_communicator);

        Sa_solution.reinit(locally_owned_dofs, mpi_communicator);

        const std::vector<IndexSet> locally_owned_dofs_per_proc_RT =
                DoFTools::locally_owned_dofs_per_subdomain(dof_handler_RT);
        locally_owned_dofs_RT = locally_owned_dofs_per_proc_RT[this_mpi_process];

        DoFTools::extract_locally_relevant_dofs(dof_handler_RT, locally_relevant_dofs_RT);

        dof_handler_dg0.distribute_dofs(fe_dg0);
        const std::vector<IndexSet> locally_owned_dofs_per_proc_dg0 =
                DoFTools::locally_owned_dofs_per_subdomain(dof_handler_dg0);
        locally_owned_dofs_dg0 = locally_owned_dofs_per_proc_dg0[this_mpi_process];

        DoFTools::extract_locally_relevant_dofs(dof_handler_dg0, locally_relevant_dofs_dg0);
    }

    template <int dim>
    void AqueousSaturationProblem<dim>::assemble_system_matrix_aqueous_saturation()
    {

        setup_mat();


      FEFaceValues<dim> fe_face_values_RT(fe_RT,
                                            face_quadrature,
                                            update_values);

        FEFaceValues<dim> fe_face_values_RT_neighbor(fe_RT,
                                                     face_quadrature,
                                                     update_values);

        const FEValuesExtractors::Vector velocities(0);

        using Iterator = typename DoFHandler<dim>::active_cell_iterator;


        BoundaryValuesAqueousSaturation<dim> boundary_function;
        BoundaryValuesLiquidPressure<dim> boundary_function_pl;
        RightHandSideAqueousSaturation<dim> right_hand_side_fcn;
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

        // Stabilization term. Declared and defined
        Kappa_tilde_a<dim> Kappa_tilde_a_fcn;
        double Kappa_tilde_a = Kappa_tilde_a_fcn.value();

        // Capillary pressures
        CapillaryPressurePca<dim> cap_p_pca_fcn;
        CapillaryPressurePcv<dim> cap_p_pcv_fcn;

        // Neumann term
        NeumannTermAqueousSaturation<dim> neumann_fcn;

        // Solutions on this processor
        PETScWrappers::MPI::Vector temp_pl_solution;
        PETScWrappers::MPI::Vector temp_pl_solution_n;
        PETScWrappers::MPI::Vector temp_pl_solution_nminus1;

        PETScWrappers::MPI::Vector temp_Sa_solution_n;
        PETScWrappers::MPI::Vector temp_Sa_solution_nminus1;

        PETScWrappers::MPI::Vector temp_Sv_solution_n;
        PETScWrappers::MPI::Vector temp_Sv_solution_nminus1;

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

        temp_Sa_solution_n.reinit(locally_owned_dofs,
                                  locally_relevant_dofs,
                                  mpi_communicator);

        temp_Sa_solution_nminus1.reinit(locally_owned_dofs,
                                        locally_relevant_dofs,
                                        mpi_communicator);

        temp_Sv_solution_n.reinit(locally_owned_dofs,
                                  locally_relevant_dofs,
                                  mpi_communicator);

        temp_Sv_solution_nminus1.reinit(locally_owned_dofs,
                                        locally_relevant_dofs,
                                        mpi_communicator);

        temp_totalDarcyVelocity_RT.reinit(locally_owned_dofs_RT,
                                          locally_relevant_dofs_RT,
                                          mpi_communicator);

        temp_kappa.reinit(locally_owned_dofs_dg0,
                          locally_relevant_dofs_dg0,
                          mpi_communicator);

        temp_pl_solution = pl_solution;
        temp_pl_solution_n = pl_solution_n;
        temp_pl_solution_nminus1 = pl_solution_nminus1;

        temp_Sa_solution_n = Sa_solution_n;
        temp_Sa_solution_nminus1 = Sa_solution_nminus1;

        temp_Sv_solution_n = Sv_solution_n;
        temp_Sv_solution_nminus1 = Sv_solution_nminus1;

        temp_totalDarcyVelocity_RT = totalDarcyvelocity_RT;

        temp_kappa = kappa_abs_vec;

        // Volume integrals
        const auto cell_worker = [&](const auto &cell,
                                     auto &scratch_data,
                                     auto &copy_data)
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
//
            std::vector<double>         rhs_values(n_qpoints);
            right_hand_side_fcn.set_time(time);
            right_hand_side_fcn.value_list(q_points, rhs_values);

            gravity_fcn.set_time(time);

            // Vectors containing discrete solutions at int points
            std::vector<double> pl_vals(n_qpoints);
            std::vector<double> old_pl_vals(n_qpoints);
            std::vector<double> old_pl_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> pl_grads(n_qpoints);

            std::vector<double> old_Sa_vals(n_qpoints);
            std::vector<double> old_Sa_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sa_grads(n_qpoints);

            std::vector<double> old_Sv_vals(n_qpoints);
            std::vector<double> old_Sv_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_nminus1(n_qpoints);

            fe_v.get_function_values(temp_pl_solution, pl_vals);
            fe_v.get_function_values(temp_pl_solution_n, old_pl_vals);
            fe_v.get_function_values(temp_pl_solution_nminus1, old_pl_vals_nminus1);
            fe_v.get_function_gradients(temp_pl_solution, pl_grads);

            fe_v.get_function_values(temp_Sa_solution_n, old_Sa_vals);
            fe_v.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1);
            fe_v.get_function_gradients(temp_Sa_solution_n, old_Sa_grads);

            fe_v.get_function_values(temp_Sv_solution_n, old_Sv_vals);
            fe_v.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1);
            fe_v.get_function_gradients(temp_Sv_solution_n, old_Sv_grads);
            fe_v.get_function_gradients(temp_Sv_solution_nminus1, old_Sv_grads_nminus1);

            std::vector<Tensor<1, dim>> DarcyVelocities(n_qpoints);
            fe_values_RT[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities);

            // get maximum of Darcy Velocity, This is for artificial viscosity stuff
            std::vector<double> linf_norm_Darcy_vel(n_qpoints);
            for(unsigned int kk = 0; kk < n_qpoints; kk++)
            {
                Vector<double> darcy_v(dim);
                for(unsigned int jj = 0; jj < dim; jj++)
                    darcy_v[jj] = DarcyVelocities[kk][jj];

                linf_norm_Darcy_vel[kk] = darcy_v.linfty_norm();
            }

            double maximum_Darcy = *std::max_element(linf_norm_Darcy_vel.begin(), linf_norm_Darcy_vel.end());
            double maximum_Sa = *std::max_element(old_Sa_vals.begin(), old_Sa_vals.end());

            double kappa = temp_kappa[cell->global_active_cell_index()];

            for (unsigned int point = 0; point < n_qpoints; ++point)
            {
                double pl_value = pl_vals[point];
                double pl_value_n = old_pl_vals[point];
                double pl_value_nminus1 = old_pl_vals_nminus1[point];
                Tensor<1,dim> pl_grad = pl_grads[point];

                if(use_exact_pl_in_Sa)
                {
                    pl_fcn.set_time(time);

                    pl_value = pl_fcn.value(q_points[point]);
                    pl_grad = pl_fcn.gradient(q_points[point]);

                    pl_fcn.set_time(time - time_step);

                    pl_value_n = pl_fcn.value(q_points[point]);

                    pl_fcn.set_time(time - 2.0*time_step);

                    pl_value_nminus1 = pl_fcn.value(q_points[point]);
                }

                double Sa_value_n = old_Sa_vals[point];
                double Sa_value_nminus1 = old_Sa_vals_nminus1[point];
                Tensor<1,dim> Sa_grad_n = old_Sa_grads[point];

                double Sv_value_n = old_Sv_vals[point];
                double Sv_value_nminus1 = old_Sv_vals_nminus1[point];
                Tensor<1,dim> Sv_grad_n = old_Sv_grads[point];
                Tensor<1,dim> Sv_grad_nminus1 = old_Sv_grads_nminus1[point];

                if(use_exact_Sv_in_Sa)
                {
                    Sv_fcn.set_time(time - time_step);

                    Sv_value_n = Sv_fcn.value(q_points[point]);
                    Sv_grad_n = Sv_fcn.gradient(q_points[point]);

                    if(timestep_number > 1)
                        Sv_fcn.set_time(time - 2.0*time_step);

                    Sv_value_nminus1 = Sv_fcn.value(q_points[point]);
                    Sv_grad_nminus1 = Sv_fcn.gradient(q_points[point]);
                }

                // Darcy velocity at current int point
                Tensor<1,dim> totalDarcyVelo = DarcyVelocities[point];

                // Second order extrapolations if needed
                double Sa_nplus1_extrapolation = Sa_value_n;
                double Sv_nplus1_extrapolation = Sv_value_n;
                Tensor<1,dim> Sv_grad_nplus1_extrapolation = Sv_grad_n;
                Tensor<1,dim> totalDarcyVelo_extrapolation = totalDarcyVelo;

                if(second_order_extrapolation)
                {
                    Sa_nplus1_extrapolation *= 2.0;
                    Sa_nplus1_extrapolation -= Sa_value_nminus1;

                    Sv_nplus1_extrapolation *= 2.0;
                    Sv_nplus1_extrapolation -= Sv_value_nminus1;

                    Sv_grad_nplus1_extrapolation *= 2.0;
                    Sv_grad_nplus1_extrapolation -= Sv_grad_nminus1;

                }
                // Coefficient values
                double phi_nplus1 = porosity_fcn.value(pl_value);
                double phi_n = porosity_fcn.value(pl_value_n);
                double phi_nminus1 = porosity_fcn.value(pl_value_nminus1);

                double rho_l = rho_l_fcn.value(pl_value);
                double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                double rho_a = rho_a_fcn.value(pl_value);

                double rho_a_n = rho_a_fcn.value(pl_value_n);
                double rho_a_nminus1 = rho_a_fcn.value(pl_value_nminus1);

                if(incompressible)
                {
                    rho_l = rho_v = rho_a = 1.0;
                    rho_a_n = 1.0;
                    rho_a_nminus1 = 1.0;
                }

                double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

                double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

                double dpca_dSa = cap_p_pca_fcn.derivative_wrt_Sa(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                double dpca_dSv = cap_p_pca_fcn.derivative_wrt_Sv(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

                // Artificial viscosity stuff. This is not fully tested/working
                double nu_h_artificial_visc = 0.0;
                if(artificial_visc_exp || artificial_visc_imp)
                    nu_h_artificial_visc = 0.5*sqrt(cell->measure())*art_visc_multiple_Sa*maximum_Darcy*2.0*maximum_Sa;

                // This is where the main formulation starts
                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {
                        if(timestep_number == 1 || !second_order_time_derivative)
                        {
                            // Time term
                            copy_data.cell_matrix(i,j) +=
                                    (1.0/time_step)
                                    * phi_nplus1
                                    * rho_a
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
                                    * rho_a
                                    * fe_v.shape_value(i, point)
                                    * fe_v.shape_value(j, point)
                                    * JxW[point];
                        }
                        if (Stab_a)
                        {
                                // Diffusion Term
                                copy_data.cell_matrix(i,j) -=
                                    -Kappa_tilde_a // also must be negative
                                    * kappa
                                    * fe_v.shape_grad(i, point)
                                    * fe_v.shape_grad(j, point)
                                    * JxW[point];
                        }
                        else
                        {
                            // Diffusion Term
                            copy_data.cell_matrix(i,j) -=
                                    rho_a
                                    * lambda_a
                                    * dpca_dSa // negative term
                                    * kappa
                                    * fe_v.shape_grad(i, point)
                                    * fe_v.shape_grad(j, point)
                                    * JxW[point];

                        }
                        if(artificial_visc_imp)
                        {
                            copy_data.cell_matrix(i,j) +=
                                    nu_h_artificial_visc
                                    * fe_v.shape_grad(i, point)
                                    * fe_v.shape_grad(j, point)
                                    * JxW[point];
                        }
                    }
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

            FEFaceValues<dim> fe_face_values_RT(fe_RT,
                                                face_quadrature,
                                                update_values);

            typename DoFHandler<dim>::cell_iterator cell_RT(&triangulation,
                                                            cell->level(), cell->index(), &dof_handler_RT);

            fe_face_values_RT.reinit(cell_RT, face_no);

            const unsigned int n_facet_dofs = fe_face.dofs_per_cell;
            const std::vector<double> &        JxW     = scratch_data.get_JxW_values();
            const std::vector<Tensor<1, dim>> &normals = scratch_data.get_normal_vectors();

            std::vector<double> g(n_qpoints);
            boundary_function.set_time(time);
            boundary_function.value_list(q_points, g);

            boundary_function_pl.set_time(time);
            std::vector<double> g_pl(n_qpoints);
            boundary_function_pl.value_list(q_points, g_pl);

            gravity_fcn.set_time(time);

            neumann_fcn.set_time(time);

            std::vector<double> pl_vals(n_qpoints);
            std::vector<Tensor<1, dim>> pl_grads(n_qpoints);

            std::vector<double> old_Sa_vals(n_qpoints);
            std::vector<double> old_Sa_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sa_grads(n_qpoints);

            std::vector<double> old_Sv_vals(n_qpoints);
            std::vector<double> old_Sv_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_nminus1(n_qpoints);

            fe_face.get_function_values(temp_pl_solution, pl_vals);
            fe_face.get_function_gradients(temp_pl_solution, pl_grads);

            fe_face.get_function_values(temp_Sa_solution_n, old_Sa_vals);
            fe_face.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1);
            fe_face.get_function_gradients(temp_Sa_solution_n, old_Sa_grads);

            fe_face.get_function_values(temp_Sv_solution_n, old_Sv_vals);
            fe_face.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1);
            fe_face.get_function_gradients(temp_Sv_solution_n, old_Sv_grads);
            fe_face.get_function_gradients(temp_Sv_solution_nminus1, old_Sv_grads_nminus1);

            std::vector<Tensor<1, dim>> DarcyVelocities(n_qpoints);
            fe_face_values_RT[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities);

            // get maximum of Darcy Velocity. Art visc stuff (not fully tested/working)
            std::vector<double> linf_norm_Darcy_vel(n_qpoints);
            for(unsigned int kk = 0; kk < n_qpoints; kk++)
            {
                Vector<double> darcy_v(dim);
                for(unsigned int jj = 0; jj < dim; jj++)
                    darcy_v[jj] = DarcyVelocities[kk][jj];

                linf_norm_Darcy_vel[kk] = darcy_v.linfty_norm();
            }

            double maximum_Darcy = *std::max_element(linf_norm_Darcy_vel.begin(), linf_norm_Darcy_vel.end());
            double maximum_Sa = *std::max_element(old_Sa_vals.begin(), old_Sa_vals.end());

            double kappa = temp_kappa[cell->global_active_cell_index()];

            // Figure out if this face is Dirichlet or Neumann
            bool dirichlet = false;

            for(unsigned int i = 0; i < dirichlet_id_sa.size(); i++)
            {
                if(cell->face(face_no)->boundary_id() == dirichlet_id_sa[i])
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

                    if(use_exact_pl_in_Sa)
                    {
                        pl_fcn.set_time(time);

                        pl_value = pl_fcn.value(q_points[point]);
                        pl_grad = pl_fcn.gradient(q_points[point]);

                    }

                    double Sa_value_n = old_Sa_vals[point];
                    double Sa_value_nminus1 = old_Sa_vals_nminus1[point];
                    Tensor<1,dim> Sa_grad_n = old_Sa_grads[point];

                    double Sv_value_n = old_Sv_vals[point];
                    double Sv_value_nminus1 = old_Sv_vals_nminus1[point];
                    Tensor<1,dim> Sv_grad_n = old_Sv_grads[point];
                    Tensor<1,dim> Sv_grad_nminus1 = old_Sv_grads_nminus1[point];

                    if(use_exact_Sv_in_Sa)
                    {
                        Sv_fcn.set_time(time - time_step);

                        Sv_value_n = Sv_fcn.value(q_points[point]);
                        Sv_grad_n = Sv_fcn.gradient(q_points[point]);
                        Sv_fcn.set_time(time - 2.0*time_step);

                        Sv_value_nminus1 = Sv_fcn.value(q_points[point]);
                        Sv_grad_nminus1 = Sv_fcn.gradient(q_points[point]);
                    }

                    Tensor<1,dim> totalDarcyVelo = DarcyVelocities[point];

                    // Second order extrapolations if needed
                    double Sa_nplus1_extrapolation = Sa_value_n;
                    double Sv_nplus1_extrapolation = Sv_value_n;
                    Tensor<1,dim> Sv_grad_nplus1_extrapolation = Sv_grad_n;
                    Tensor<1,dim> totalDarcyVelo_extrapolation = totalDarcyVelo;

                    if(second_order_extrapolation)
                    {
                        Sa_nplus1_extrapolation *= 2.0;
                        Sa_nplus1_extrapolation -= Sa_value_nminus1;

                        Sv_nplus1_extrapolation *= 2.0;
                        Sv_nplus1_extrapolation -= Sv_value_nminus1;

                        Sv_grad_nplus1_extrapolation *= 2.0;
                        Sv_grad_nplus1_extrapolation -= Sv_grad_nminus1;

                    }

                    // Coefficients
                    double rho_l = rho_l_fcn.value(pl_value);
                    double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double rho_a = rho_a_fcn.value(pl_value);

                    if(incompressible)
                    {
                        rho_l = rho_v = rho_a = 1.0;
                    }

                    double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

//				double lambda_l = lambda_l_fcn.value(g_pl[point], g[point], Sv_nplus1_extrapolation);
//				double lambda_v = lambda_v_fcn.value(g_pl[point], g[point], Sv_nplus1_extrapolation);
//				double lambda_a = lambda_a_fcn.value(g_pl[point], g[point], Sv_nplus1_extrapolation);

                    double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

                    double dpca_dSa = cap_p_pca_fcn.derivative_wrt_Sa(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
//				    double dpca_dSa = cap_p_pca_fcn.derivative_wrt_Sa(g[point], Sv_nplus1_extrapolation);
                    double dpca_dSv = cap_p_pca_fcn.derivative_wrt_Sv(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

                    double nu_h_artificial_visc = 0.0;
                    if(artificial_visc_exp || artificial_visc_imp)
                        nu_h_artificial_visc = 0.5*sqrt(cell->measure())*art_visc_multiple_Sa*maximum_Darcy*2.0*maximum_Sa;

                    double gamma_Sa_e = fabs(rho_a*lambda_a*kappa*dpca_dSa);

                    if(artificial_visc_imp)
                        gamma_Sa_e += nu_h_artificial_visc;

                    gamma_Sa_e += sqrt(totalDarcyVelo_extrapolation*totalDarcyVelo_extrapolation);

                    double h_e = cell->face(face_no)->measure();
                    double penalty_factor = (penalty_Sa_bdry/h_e) * gamma_Sa_e * degree*(degree + dim - 1);

                    // start of boundary terms
                    for (unsigned int i = 0; i < n_facet_dofs; ++i)
                    {

                        for (unsigned int j = 0; j < n_facet_dofs; ++j)
                        {
                            if(Stab_a)
                            {
                                // Diffusion term
                                copy_data.cell_matrix(i, j) -=
                                        Kappa_tilde_a
                                        * kappa
                                        * fe_face.shape_value(i, point)
                                        * fe_face.shape_grad(j, point)
                                        * normals[point]
                                        * JxW[point];
                                // Theta term
                                copy_data.cell_matrix(i, j) +=
                                        Kappa_tilde_a
                                        * kappa
                                        * theta_Sa
                                        * fe_face.shape_grad(i, point)
                                        * normals[point]
                                        * fe_face.shape_value(j, point)
                                        * JxW[point];
                            }
                            else
                            {
                                // Diffusion term
                                copy_data.cell_matrix(i, j) -=
                                        -rho_a
                                        * lambda_a
                                        * dpca_dSa
                                        * kappa
                                        * fe_face.shape_value(i, point)
                                        * fe_face.shape_grad(j, point)
                                        * normals[point]
                                        * JxW[point];
                                //Theta term
                                copy_data.cell_matrix(i, j) +=
                                         -rho_a
                                        * lambda_a
                                        * dpca_dSa
                                        * kappa
                                        * theta_Sa
                                        * fe_face.shape_grad(i, point)
                                        * normals[point]
                                        * fe_face.shape_value(j, point)
                                        * JxW[point];
                                //                             Boundary condition

                            }
                            // Not fully tested/working
                            if(artificial_visc_imp)
                            {
                                copy_data.cell_matrix(i, j) -=
                                        nu_h_artificial_visc
                                        * fe_face.shape_value(i, point)
                                        * fe_face.shape_grad(j, point)
                                        * normals[point]
                                        * JxW[point];

                                copy_data.cell_matrix(i, j) +=
                                        theta_Sa
                                        * nu_h_artificial_visc
                                        * fe_face.shape_grad(i, point)
                                        * normals[point]
                                        * fe_face.shape_value(j, point)
                                        * JxW[point];
                            }

//                             Boundary condition
                            copy_data.cell_matrix(i, j) +=
                                    penalty_factor
                                    * fe_face.shape_value(i, point)
                                    * fe_face.shape_value(j, point)
                                    * JxW[point];
                        }
                    }
                }
            }
            else // Neumann boundary
            {
                for (unsigned int point = 0; point < q_points.size(); ++point)
                {
                    double pl_value = pl_vals[point];
                    Tensor<1,dim> pl_grad = pl_grads[point];

                    if(use_exact_pl_in_Sa)
                    {
                        pl_fcn.set_time(time);

                        pl_value = pl_fcn.value(q_points[point]);
                        pl_grad = pl_fcn.gradient(q_points[point]);

                    }

                    double Sa_value_n = old_Sa_vals[point];
                    double Sa_value_nminus1 = old_Sa_vals_nminus1[point];
                    Tensor<1,dim> Sa_grad_n = old_Sa_grads[point];

                    double Sv_value_n = old_Sv_vals[point];
                    double Sv_value_nminus1 = old_Sv_vals_nminus1[point];

                    if(use_exact_Sv_in_Sa)
                    {
                        Sv_fcn.set_time(time - time_step);

                        Sv_value_n = Sv_fcn.value(q_points[point]);

                        Sv_fcn.set_time(time - 2.0*time_step);

                        Sv_value_nminus1 = Sv_fcn.value(q_points[point]);
                    }

                    // Second order extrapolations if needed
                    double Sa_nplus1_extrapolation = Sa_value_n;
                    double Sv_nplus1_extrapolation = Sv_value_n;

                    if(second_order_extrapolation)
                    {
                        Sa_nplus1_extrapolation *= 2.0;
                        Sa_nplus1_extrapolation -= Sa_value_nminus1;

                        Sv_nplus1_extrapolation *= 2.0;
                        Sv_nplus1_extrapolation -= Sv_value_nminus1;

                    }

                    Tensor<1,dim> neumann_term = neumann_fcn.vector_value(q_points[point]);
                    Tensor<1,dim> totalDarcyVelo = DarcyVelocities[point];

                    double rho_l = rho_l_fcn.value(pl_value);
                    double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double rho_a = rho_a_fcn.value(pl_value);

                    if(incompressible)
                    {
                        rho_l = rho_v = rho_a = 1.0;
                    }

                    double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

                    double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

//                    for (unsigned int i = 0; i < n_facet_dofs; ++i)
//                    {
////
////					if(cell->face(face_no)->boundary_id() == 5 || cell->face(face_no)->boundary_id() == 6)
////					{
////                        if(project_only_kappa)
////                            copy_data.cell_rhs(i) -= (rho_a*lambda_a)
////                                                     * totalDarcyVelo
////                                                     * normals[point]
////                                                     * fe_face.shape_value(i, point)
////                                                     * JxW[point];
////                        else
////                            copy_data.cell_rhs(i) -= (rho_a*lambda_a/rholambda_t)
////                                                     * totalDarcyVelo
////                                                     * normals[point]
////                                                     * fe_face.shape_value(i, point)
////                                                     * JxW[point];
////			}
////                        copy_data.cell_rhs(i) += neumann_term
////                                                 * normals[point]
////                                                 * fe_face.shape_value(i, point)
////                                                 * JxW[point];
//                    }
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
//            copy_data_face.cell_rhs.reinit(n_dofs);

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

            std::vector<double> old_Sa_vals(n_qpoints);
            std::vector<double> old_Sa_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sa_grads(n_qpoints);

            std::vector<double> old_Sa_vals_neighbor(n_qpoints);
            std::vector<double> old_Sa_vals_nminus1_neighbor(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sa_grads_neighbor(n_qpoints);

            std::vector<double> old_Sv_vals(n_qpoints);
            std::vector<double> old_Sv_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_nminus1(n_qpoints);

            std::vector<double> old_Sv_vals_neighbor(n_qpoints);
            std::vector<double> old_Sv_vals_nminus1_neighbor(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_neighbor(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_nminus1_neighbor(n_qpoints);

            fe_face.get_function_values(temp_pl_solution, pl_vals);
            fe_face_neighbor.get_function_values(temp_pl_solution, pl_vals_neighbor);

            fe_face.get_function_gradients(temp_pl_solution, pl_grads);
            fe_face_neighbor.get_function_gradients(temp_pl_solution, pl_grads_neighbor);

            fe_face.get_function_values(temp_Sa_solution_n, old_Sa_vals);
            fe_face.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1);
            fe_face.get_function_gradients(temp_Sa_solution_n, old_Sa_grads);

            fe_face_neighbor.get_function_values(temp_Sa_solution_n, old_Sa_vals_neighbor);
            fe_face_neighbor.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1_neighbor);
            fe_face_neighbor.get_function_gradients(temp_Sa_solution_n, old_Sa_grads_neighbor);

            fe_face.get_function_values(temp_Sv_solution_n, old_Sv_vals);
            fe_face.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1);
            fe_face.get_function_gradients(temp_Sv_solution_n, old_Sv_grads);
            fe_face.get_function_gradients(temp_Sv_solution_nminus1, old_Sv_grads_nminus1);

            fe_face_neighbor.get_function_values(temp_Sv_solution_n, old_Sv_vals_neighbor);
            fe_face_neighbor.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1_neighbor);
            fe_face_neighbor.get_function_gradients(temp_Sv_solution_n, old_Sv_grads_neighbor);
            fe_face_neighbor.get_function_gradients(temp_Sv_solution_nminus1, old_Sv_grads_nminus1_neighbor);

            std::vector<Tensor<1, dim>> DarcyVelocities(n_qpoints);
            fe_face_values_RT[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities);

            std::vector<Tensor<1, dim>> DarcyVelocities_neighbor(n_qpoints);
            fe_face_values_RT_neighbor[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities_neighbor);

            // get maximum of Darcy Velocity
            std::vector<double> linf_norm_Darcy_vel0(n_qpoints);
            std::vector<double> linf_norm_Darcy_vel1(n_qpoints);

            for(unsigned int kk = 0; kk < n_qpoints; kk++)
            {
                Vector<double> darcy_v0(dim), darcy_v1(dim);
                for(unsigned int jj = 0; jj < dim; jj++)
                {
                    darcy_v0[jj] = DarcyVelocities[kk][jj];
                    darcy_v1[jj] = DarcyVelocities_neighbor[kk][jj];
                }

                linf_norm_Darcy_vel0[kk] = darcy_v0.linfty_norm();
                linf_norm_Darcy_vel1[kk] = darcy_v1.linfty_norm();
            }

            double maximum_Darcy0 = *std::max_element(linf_norm_Darcy_vel0.begin(), linf_norm_Darcy_vel0.end());
            double maximum_Darcy1 = *std::max_element(linf_norm_Darcy_vel1.begin(), linf_norm_Darcy_vel1.end());

            double maximum_Sa0 = *std::max_element(old_Sa_vals.begin(), old_Sa_vals.end());
            double maximum_Sa1 = *std::max_element(old_Sa_vals_neighbor.begin(), old_Sa_vals_neighbor.end());

            double kappa0 = temp_kappa[cell->global_active_cell_index()];
            double kappa1 = temp_kappa[ncell->global_active_cell_index()];

            for (unsigned int point = 0; point < n_qpoints; ++point)
            {
                // Get pl, sa and sv values on current integration point.
                // The 0 indicates current element, and 1 indicates neighboring element.
                double pl_value0 = pl_vals[point];
                double pl_value1 = pl_vals_neighbor[point];

                Tensor<1,dim> pl_grad0 = pl_grads[point];
                Tensor<1,dim> pl_grad1 = pl_grads_neighbor[point];

                if(use_exact_pl_in_Sa)
                {
                    pl_fcn.set_time(time);

                    pl_value0 = pl_fcn.value(q_points[point]);
                    pl_value1 = pl_value0;

                    pl_grad0 = pl_fcn.gradient(q_points[point]);
                    pl_grad1 = pl_grad0;
                }

                double Sa_value0_n = old_Sa_vals[point];
                double Sa_value1_n = old_Sa_vals_neighbor[point];
                Tensor<1,dim> Sa_grad0_n = old_Sa_grads[point];
                Tensor<1,dim> Sa_grad1_n = old_Sa_grads_neighbor[point];

                double Sa_value0_nminus1 = old_Sa_vals_nminus1[point];
                double Sa_value1_nminus1 = old_Sa_vals_nminus1_neighbor[point];

                double Sv_value0_n = old_Sv_vals[point];
                double Sv_value1_n = old_Sv_vals_neighbor[point];
                double Sv_value0_nminus1 = old_Sv_vals_nminus1[point];
                double Sv_value1_nminus1 = old_Sv_vals_nminus1_neighbor[point];

                Tensor<1,dim> Sv_grad0_n = old_Sv_grads[point];
                Tensor<1,dim> Sv_grad1_n = old_Sv_grads_neighbor[point];
                Tensor<1,dim> Sv_grad0_nminus1 = old_Sv_grads_nminus1[point];
                Tensor<1,dim> Sv_grad1_nminus1 = old_Sv_grads_nminus1_neighbor[point];

                if(use_exact_Sv_in_Sa)
                {
                    Sv_fcn.set_time(time - time_step);

                    Sv_value0_n = Sv_fcn.value(q_points[point]);
                    Sv_value1_n = Sv_value0_n;
                    Sv_grad0_n = Sv_fcn.gradient(q_points[point]);
                    Sv_grad1_n = Sv_grad0_n;

                    Sv_fcn.set_time(time - 2.0*time_step);

                    Sv_value0_nminus1 = Sv_fcn.value(q_points[point]);
                    Sv_value1_nminus1 = Sv_value0_nminus1;
                    Sv_grad0_nminus1 = Sv_fcn.gradient(q_points[point]);
                    Sv_grad1_nminus1 = Sv_grad0_nminus1;
                }

                // Darcy velocities
                Tensor<1,dim> totalDarcyVelo0 = DarcyVelocities[point];
                Tensor<1,dim> totalDarcyVelo1 = DarcyVelocities_neighbor[point];

                // Second order extrapolations if needed
                double Sa_nplus1_extrapolation0 = Sa_value0_n;
                double Sa_nplus1_extrapolation1 = Sa_value1_n;
                double Sv_nplus1_extrapolation0 = Sv_value0_n;
                double Sv_nplus1_extrapolation1 = Sv_value1_n;
                Tensor<1,dim> Sv_grad_nplus1_extrapolation0 = Sv_grad0_n;
                Tensor<1,dim> Sv_grad_nplus1_extrapolation1 = Sv_grad1_n;
                Tensor<1,dim> totalDarcyVelo_extrapolation0 = totalDarcyVelo0;
                Tensor<1,dim> totalDarcyVelo_extrapolation1 = totalDarcyVelo1;

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

                    Sv_grad_nplus1_extrapolation0 *= 2.0;
                    Sv_grad_nplus1_extrapolation0 -= Sv_grad0_nminus1;

                    Sv_grad_nplus1_extrapolation1 *= 2.0;
                    Sv_grad_nplus1_extrapolation1 -= Sv_grad1_nminus1;

                }

                // Coefficients
                double rho_l0 = rho_l_fcn.value(pl_value0);
                double rho_l1 = rho_l_fcn.value(pl_value1);

                double rho_v0 = rho_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double rho_v1 = rho_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

                double rho_a0 = rho_a_fcn.value(pl_value0);
                double rho_a1 = rho_a_fcn.value(pl_value1);

                if(incompressible)
                {
                    rho_l0 = rho_v0 = rho_a0 = 1.0;
                    rho_l1 = rho_v1 = rho_a1 = 1.0;
                }

                double lambda_l0 = lambda_l_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double lambda_v0 = lambda_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double lambda_a0 = lambda_a_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);

                double lambda_l1 = lambda_l_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
                double lambda_v1 = lambda_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
                double lambda_a1 = lambda_a_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

                double rholambda_t0 = rho_l0*lambda_l0 + rho_v0*lambda_v0 + rho_a0*lambda_a0;
                double rholambda_t1 = rho_l1*lambda_l1 + rho_v1*lambda_v1 + rho_a1*lambda_a1;

                double dpca_dSa0 = cap_p_pca_fcn.derivative_wrt_Sa(Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double dpca_dSa1 = cap_p_pca_fcn.derivative_wrt_Sa(Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

                double dpca_dSv0 = cap_p_pca_fcn.derivative_wrt_Sv(Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double dpca_dSv1 = cap_p_pca_fcn.derivative_wrt_Sv(Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

                double nu_h_artificial_visc0 = 0.0;
                double nu_h_artificial_visc1 = 0.0;

                if(artificial_visc_exp || artificial_visc_imp)
                {
                    nu_h_artificial_visc0 = 0.5*sqrt(cell->measure())*art_visc_multiple_Sa*maximum_Darcy0*2.0*maximum_Sa0;
                    nu_h_artificial_visc1 = 0.5*sqrt(ncell->measure())*art_visc_multiple_Sa*maximum_Darcy1*2.0*maximum_Sa1;
                }

                // Diffusion coefficients and weights for stab method
                double coef0_diff = fabs(rho_a0*lambda_a0*kappa0*dpca_dSa0);
                double coef1_diff = fabs(rho_a1*lambda_a1*kappa1*dpca_dSa1);

                // for stabilization term
                double coef0_diff_stab = fabs(kappa0*Kappa_tilde_a);
                double coef1_diff_stab = fabs(kappa1*Kappa_tilde_a);

                if(artificial_visc_imp)
                {
                    coef0_diff += nu_h_artificial_visc0;
                    coef1_diff += nu_h_artificial_visc1;
                }

                double gamma_Sa_e = fabs(2.0*coef0_diff*coef1_diff/(coef0_diff + coef1_diff + 1.e-20));

                double h_e = cell->face(f)->measure();
                double penalty_factor = (penalty_Sa/h_e) * gamma_Sa_e * degree*(degree + dim - 1);

                double weight0_diff = coef1_diff/(coef0_diff + coef1_diff + 1.e-20);
                double weight1_diff = coef0_diff/(coef0_diff + coef1_diff + 1.e-20);

                // for stabilization term
                double weight0_diff_stab = coef1_diff_stab/(coef0_diff_stab + coef1_diff_stab + 1.e-20);
                double weight1_diff_stab = coef0_diff_stab/(coef0_diff_stab + coef1_diff_stab + 1.e-20);

                // Sv coefficients and weights
                double coef0_Sv = rho_a0*lambda_a0*dpca_dSv0*kappa0;
                double coef1_Sv = rho_a1*lambda_a1*dpca_dSv1*kappa1;

                double weight0_Sv = coef1_Sv/(coef0_Sv + coef1_Sv + 1.e-20);
                double weight1_Sv = coef0_Sv/(coef0_Sv + coef1_Sv + 1.e-20);

                //Sa coefficients and weights for stab method
                double coef0_Sa_stab = (rho_a0*lambda_a0*dpca_dSa0+Kappa_tilde_a)*kappa0;
                double coef1_Sa_stab = (rho_a1*lambda_a1*dpca_dSa1+Kappa_tilde_a)*kappa1;

                double weight0_Sa_stab = coef1_Sa_stab/(coef0_Sa_stab + coef1_Sa_stab + 1.e-20);
                double weight1_Sa_stab = coef0_Sa_stab/(coef0_Sa_stab + coef1_Sa_stab + 1.e-20);


                // start of interior face terms
                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {
//                         Interior face terms from diffusion
                        copy_data_face.cell_matrix(i, j) +=
                                penalty_factor
                                * fe_iv.jump(i, point)
                                * fe_iv.jump(j, point)
                                * JxW[point];
                        if (Stab_a) {
                            double weighted_aver_j_stab = AverageGradOperators::weighted_average_gradient<dim>(cell, f,
                                                                                                               sf,
                                                                                                               ncell,
                                                                                                               nf,
                                                                                                               nsf,
                                                                                                               fe_iv,
                                                                                                               normals[point],
                                                                                                               j, point,
                                                                                                               coef0_diff_stab,
                                                                                                               coef1_diff_stab,
                                                                                                               weight0_diff_stab,
                                                                                                               weight1_diff_stab);
                            copy_data_face.cell_matrix(i, j) -=
                                    fe_iv.jump(i, point)
                                    * weighted_aver_j_stab
                                    * JxW[point];
                            double weighted_aver_i_stab = AverageGradOperators::weighted_average_gradient<dim>(cell, f,
                                                                                                               sf,
                                                                                                               ncell,
                                                                                                               nf,
                                                                                                               nsf,
                                                                                                               fe_iv,
                                                                                                               normals[point],
                                                                                                               i, point,
                                                                                                               coef0_diff_stab,
                                                                                                               coef1_diff_stab,
                                                                                                               weight0_diff_stab,
                                                                                                               weight1_diff_stab);
                            copy_data_face.cell_matrix(i, j) +=
                                    theta_Sa
                                    * fe_iv.jump(j, point)
                                    * weighted_aver_i_stab
                                    * JxW[point];
                            //                         Interior face terms from diffusion
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
                                    theta_Sa
                                    * fe_iv.jump(j, point)
                                    * weighted_aver_i
                                    * JxW[point];
                        }
                    }

                }
            }

        };

            const auto copier = [&](const CopyData &c) {
                constraints.distribute_local_to_global(c.cell_matrix,
                                                       /*c.cell_rhs,*/
                                                       c.local_dof_indices,
                                                       system_matrix_aqueous_saturation
                                                      /* right_hand_side_aqueous_saturation*/);

                for (auto &cdf : c.face_data)
                {
                    constraints.distribute_local_to_global(cdf.cell_matrix,
                                                           /*cdf.cell_rhs,*/
                                                           cdf.joint_dof_indices,
                                                           system_matrix_aqueous_saturation
                                                           /*right_hand_side_aqueous_saturation*/);
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

            system_matrix_aqueous_saturation.compress(VectorOperation::add);

            stored_matrix.reinit(system_matrix_aqueous_saturation);
            stored_matrix.copy_from(system_matrix_aqueous_saturation);


    }

    template <int dim>
    void AqueousSaturationProblem<dim>::assemble_rhs_aqueous_saturation()
    {

        setup_rhs();

        FEFaceValues<dim> fe_face_values_RT(fe_RT,
                                            face_quadrature,
                                            update_values);

        FEFaceValues<dim> fe_face_values_RT_neighbor(fe_RT,
                                                     face_quadrature,
                                                     update_values);

        const FEValuesExtractors::Vector velocities(0);

        using Iterator = typename DoFHandler<dim>::active_cell_iterator;


        BoundaryValuesAqueousSaturation<dim> boundary_function;
        BoundaryValuesLiquidPressure<dim> boundary_function_pl;
        RightHandSideAqueousSaturation<dim> right_hand_side_fcn;
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

        // Stabilization term. Declared and defined
        Kappa_tilde_a<dim> Kappa_tilde_a_fcn;
        double Kappa_tilde_a = Kappa_tilde_a_fcn.value();

        // Capillary pressures
        CapillaryPressurePca<dim> cap_p_pca_fcn;
        CapillaryPressurePcv<dim> cap_p_pcv_fcn;

        // Neumann term
        NeumannTermAqueousSaturation<dim> neumann_fcn;

        // Solutions on this processor
        PETScWrappers::MPI::Vector temp_pl_solution;
        PETScWrappers::MPI::Vector temp_pl_solution_n;
        PETScWrappers::MPI::Vector temp_pl_solution_nminus1;

        PETScWrappers::MPI::Vector temp_Sa_solution_n;
        PETScWrappers::MPI::Vector temp_Sa_solution_nminus1;

        PETScWrappers::MPI::Vector temp_Sv_solution_n;
        PETScWrappers::MPI::Vector temp_Sv_solution_nminus1;

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

        temp_Sa_solution_n.reinit(locally_owned_dofs,
                                  locally_relevant_dofs,
                                  mpi_communicator);

        temp_Sa_solution_nminus1.reinit(locally_owned_dofs,
                                        locally_relevant_dofs,
                                        mpi_communicator);

        temp_Sv_solution_n.reinit(locally_owned_dofs,
                                  locally_relevant_dofs,
                                  mpi_communicator);

        temp_Sv_solution_nminus1.reinit(locally_owned_dofs,
                                        locally_relevant_dofs,
                                        mpi_communicator);

        temp_totalDarcyVelocity_RT.reinit(locally_owned_dofs_RT,
                                          locally_relevant_dofs_RT,
                                          mpi_communicator);

        temp_kappa.reinit(locally_owned_dofs_dg0,
                          locally_relevant_dofs_dg0,
                          mpi_communicator);

        temp_pl_solution = pl_solution;
        temp_pl_solution_n = pl_solution_n;
        temp_pl_solution_nminus1 = pl_solution_nminus1;

        temp_Sa_solution_n = Sa_solution_n;
        temp_Sa_solution_nminus1 = Sa_solution_nminus1;

        temp_Sv_solution_n = Sv_solution_n;
        temp_Sv_solution_nminus1 = Sv_solution_nminus1;

        temp_totalDarcyVelocity_RT = totalDarcyvelocity_RT;

        temp_kappa = kappa_abs_vec;

        // Volume integrals
        const auto cell_worker = [&](const auto &cell,
                                     auto &scratch_data,
                                     auto &copy_data)
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

            // Vectors containing discrete solutions at int points
            std::vector<double> pl_vals(n_qpoints);
            std::vector<double> old_pl_vals(n_qpoints);
            std::vector<double> old_pl_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> pl_grads(n_qpoints);

            std::vector<double> old_Sa_vals(n_qpoints);
            std::vector<double> old_Sa_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sa_grads(n_qpoints);

            std::vector<double> old_Sv_vals(n_qpoints);
            std::vector<double> old_Sv_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_nminus1(n_qpoints);

            fe_v.get_function_values(temp_pl_solution, pl_vals);
            fe_v.get_function_values(temp_pl_solution_n, old_pl_vals);
            fe_v.get_function_values(temp_pl_solution_nminus1, old_pl_vals_nminus1);
            fe_v.get_function_gradients(temp_pl_solution, pl_grads);

            fe_v.get_function_values(temp_Sa_solution_n, old_Sa_vals);
            fe_v.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1);
            fe_v.get_function_gradients(temp_Sa_solution_n, old_Sa_grads);

            fe_v.get_function_values(temp_Sv_solution_n, old_Sv_vals);
            fe_v.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1);
            fe_v.get_function_gradients(temp_Sv_solution_n, old_Sv_grads);
            fe_v.get_function_gradients(temp_Sv_solution_nminus1, old_Sv_grads_nminus1);

            std::vector<Tensor<1, dim>> DarcyVelocities(n_qpoints);
            fe_values_RT[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities);

            // get maximum of Darcy Velocity, This is for artificial viscosity stuff
            std::vector<double> linf_norm_Darcy_vel(n_qpoints);
            for(unsigned int kk = 0; kk < n_qpoints; kk++)
            {
                Vector<double> darcy_v(dim);
                for(unsigned int jj = 0; jj < dim; jj++)
                    darcy_v[jj] = DarcyVelocities[kk][jj];

                linf_norm_Darcy_vel[kk] = darcy_v.linfty_norm();
            }

            double maximum_Darcy = *std::max_element(linf_norm_Darcy_vel.begin(), linf_norm_Darcy_vel.end());
            double maximum_Sa = *std::max_element(old_Sa_vals.begin(), old_Sa_vals.end());

            double kappa = temp_kappa[cell->global_active_cell_index()];

            for (unsigned int point = 0; point < n_qpoints; ++point)
            {
                double pl_value = pl_vals[point];
                double pl_value_n = old_pl_vals[point];
                double pl_value_nminus1 = old_pl_vals_nminus1[point];
                Tensor<1,dim> pl_grad = pl_grads[point];

                if(use_exact_pl_in_Sa)
                {
                    pl_fcn.set_time(time);

                    pl_value = pl_fcn.value(q_points[point]);
                    pl_grad = pl_fcn.gradient(q_points[point]);

                    pl_fcn.set_time(time - time_step);

                    pl_value_n = pl_fcn.value(q_points[point]);

                    pl_fcn.set_time(time - 2.0*time_step);

                    pl_value_nminus1 = pl_fcn.value(q_points[point]);
                }

                double Sa_value_n = old_Sa_vals[point];
                double Sa_value_nminus1 = old_Sa_vals_nminus1[point];
                Tensor<1,dim> Sa_grad_n = old_Sa_grads[point];

                double Sv_value_n = old_Sv_vals[point];
                double Sv_value_nminus1 = old_Sv_vals_nminus1[point];
                Tensor<1,dim> Sv_grad_n = old_Sv_grads[point];
                Tensor<1,dim> Sv_grad_nminus1 = old_Sv_grads_nminus1[point];

                if(use_exact_Sv_in_Sa)
                {
                    Sv_fcn.set_time(time - time_step);

                    Sv_value_n = Sv_fcn.value(q_points[point]);
                    Sv_grad_n = Sv_fcn.gradient(q_points[point]);

                    if(timestep_number > 1)
                        Sv_fcn.set_time(time - 2.0*time_step);

                    Sv_value_nminus1 = Sv_fcn.value(q_points[point]);
                    Sv_grad_nminus1 = Sv_fcn.gradient(q_points[point]);
                }

                // Darcy velocity at current int point
                Tensor<1,dim> totalDarcyVelo = DarcyVelocities[point];

                // Second order extrapolations if needed
                double Sa_nplus1_extrapolation = Sa_value_n;
                double Sv_nplus1_extrapolation = Sv_value_n;
                Tensor<1,dim> Sv_grad_nplus1_extrapolation = Sv_grad_n;
                Tensor<1,dim> totalDarcyVelo_extrapolation = totalDarcyVelo;

                if(second_order_extrapolation)
                {
                    Sa_nplus1_extrapolation *= 2.0;
                    Sa_nplus1_extrapolation -= Sa_value_nminus1;

                    Sv_nplus1_extrapolation *= 2.0;
                    Sv_nplus1_extrapolation -= Sv_value_nminus1;

                    Sv_grad_nplus1_extrapolation *= 2.0;
                    Sv_grad_nplus1_extrapolation -= Sv_grad_nminus1;

                }
                // Coefficient values
                double phi_nplus1 = porosity_fcn.value(pl_value);
                double phi_n = porosity_fcn.value(pl_value_n);
                double phi_nminus1 = porosity_fcn.value(pl_value_nminus1);

                double rho_l = rho_l_fcn.value(pl_value);
                double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                double rho_a = rho_a_fcn.value(pl_value);

                double rho_a_n = rho_a_fcn.value(pl_value_n);
                double rho_a_nminus1 = rho_a_fcn.value(pl_value_nminus1);

                if(incompressible)
                {
                    rho_l = rho_v = rho_a = 1.0;
                    rho_a_n = 1.0;
                    rho_a_nminus1 = 1.0;
                }

                double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

                double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

                double dpca_dSa = cap_p_pca_fcn.derivative_wrt_Sa(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                double dpca_dSv = cap_p_pca_fcn.derivative_wrt_Sv(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

                // Artificial viscosity stuff. This is not fully tested/working
                double nu_h_artificial_visc = 0.0;
                if(artificial_visc_exp || artificial_visc_imp)
                    nu_h_artificial_visc = 0.5*sqrt(cell->measure())*art_visc_multiple_Sa*maximum_Darcy*2.0*maximum_Sa;

                // This is where the main formulation starts
                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    // Source term
                    copy_data.cell_rhs(i) += right_hand_side_fcn.value(q_points[point]) * fe_v.shape_value(i, point) * JxW[point];

//                     Time term
                    if(timestep_number == 1 || !second_order_time_derivative) // bdf1
                    {
                        copy_data.cell_rhs(i) += (1.0/time_step) * phi_n * rho_a_n * Sa_value_n
                                                 * fe_v.shape_value(i, point) * JxW[point];
                    }
                    else // bdf2
                    {
                        copy_data.cell_rhs(i) += (1.0/time_step)
                                                 * (2.0 * phi_n * rho_a_n * Sa_value_n
                                                    - 0.5 * phi_nminus1 * rho_a_nminus1 * Sa_value_nminus1)
                                                 * fe_v.shape_value(i, point) * JxW[point];
                    }


                     //this is the SV term
                    copy_data.cell_rhs(i) += rho_a * lambda_a * dpca_dSv * kappa * Sv_grad_nplus1_extrapolation
                                             * fe_v.shape_grad(i, point) * JxW[point];

                    // Diffusion term moved to RHS - stab method
                    if(Stab_a)
                    {
                        copy_data.cell_rhs(i) += (rho_a * lambda_a * dpca_dSa + Kappa_tilde_a) * kappa * Sa_grad_n
                                                 * fe_v.shape_grad(i, point) * JxW[point];

                    }

                    // Darcy term. Coefficient depends on what was projected
                    if(project_only_kappa)
                    {
                        copy_data.cell_rhs(i) += (rho_a*lambda_a) * totalDarcyVelo_extrapolation
                                                 * fe_v.shape_grad(i, point) * JxW[point];
                    }
                    else
                        copy_data.cell_rhs(i) += (rho_a*lambda_a/rholambda_t) * totalDarcyVelo_extrapolation
                                                 * fe_v.shape_grad(i, point) * JxW[point];

                    // Gravity term
                    if(!project_Darcy_with_gravity)
                        copy_data.cell_rhs(i) += kappa*rho_a*rho_a_fcn.value(pl_value)*lambda_a
                                                 * gravity_fcn.vector_value(q_points[point])
                                                 * fe_v.shape_grad(i, point)
                                                 * JxW[point];

                    // Artificial viscosity
                    if(artificial_visc_exp)
                        copy_data.cell_rhs(i) -= nu_h_artificial_visc
                                                 * Sa_grad_n
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

            FEFaceValues<dim> fe_face_values_RT(fe_RT,
                                                face_quadrature,
                                                update_values);

            typename DoFHandler<dim>::cell_iterator cell_RT(&triangulation,
                                                            cell->level(), cell->index(), &dof_handler_RT);

            fe_face_values_RT.reinit(cell_RT, face_no);

            const unsigned int n_facet_dofs = fe_face.dofs_per_cell;
            const std::vector<double> &        JxW     = scratch_data.get_JxW_values();
            const std::vector<Tensor<1, dim>> &normals = scratch_data.get_normal_vectors();

            std::vector<double> g(n_qpoints);
            boundary_function.set_time(time);
            boundary_function.value_list(q_points, g);

            boundary_function_pl.set_time(time);
            std::vector<double> g_pl(n_qpoints);
            boundary_function_pl.value_list(q_points, g_pl);

            gravity_fcn.set_time(time);

            neumann_fcn.set_time(time);

            std::vector<double> pl_vals(n_qpoints);
            std::vector<Tensor<1, dim>> pl_grads(n_qpoints);

            std::vector<double> old_Sa_vals(n_qpoints);
            std::vector<double> old_Sa_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sa_grads(n_qpoints);

            std::vector<double> old_Sv_vals(n_qpoints);
            std::vector<double> old_Sv_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_nminus1(n_qpoints);

            fe_face.get_function_values(temp_pl_solution, pl_vals);
            fe_face.get_function_gradients(temp_pl_solution, pl_grads);

            fe_face.get_function_values(temp_Sa_solution_n, old_Sa_vals);
            fe_face.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1);
            fe_face.get_function_gradients(temp_Sa_solution_n, old_Sa_grads);

            fe_face.get_function_values(temp_Sv_solution_n, old_Sv_vals);
            fe_face.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1);
            fe_face.get_function_gradients(temp_Sv_solution_n, old_Sv_grads);
            fe_face.get_function_gradients(temp_Sv_solution_nminus1, old_Sv_grads_nminus1);

            std::vector<Tensor<1, dim>> DarcyVelocities(n_qpoints);
            fe_face_values_RT[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities);

            // get maximum of Darcy Velocity. Art visc stuff (not fully tested/working)
            std::vector<double> linf_norm_Darcy_vel(n_qpoints);
            for(unsigned int kk = 0; kk < n_qpoints; kk++)
            {
                Vector<double> darcy_v(dim);
                for(unsigned int jj = 0; jj < dim; jj++)
                    darcy_v[jj] = DarcyVelocities[kk][jj];

                linf_norm_Darcy_vel[kk] = darcy_v.linfty_norm();
            }

            double maximum_Darcy = *std::max_element(linf_norm_Darcy_vel.begin(), linf_norm_Darcy_vel.end());
            double maximum_Sa = *std::max_element(old_Sa_vals.begin(), old_Sa_vals.end());

            double kappa = temp_kappa[cell->global_active_cell_index()];

            // Figure out if this face is Dirichlet or Neumann
            bool dirichlet = false;

            for(unsigned int i = 0; i < dirichlet_id_sa.size(); i++)
            {
                if(cell->face(face_no)->boundary_id() == dirichlet_id_sa[i])
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

                    if(use_exact_pl_in_Sa)
                    {
                        pl_fcn.set_time(time);

                        pl_value = pl_fcn.value(q_points[point]);
                        pl_grad = pl_fcn.gradient(q_points[point]);

                    }

                    double Sa_value_n = old_Sa_vals[point];
                    double Sa_value_nminus1 = old_Sa_vals_nminus1[point];
                    Tensor<1,dim> Sa_grad_n = old_Sa_grads[point];

                    double Sv_value_n = old_Sv_vals[point];
                    double Sv_value_nminus1 = old_Sv_vals_nminus1[point];
                    Tensor<1,dim> Sv_grad_n = old_Sv_grads[point];
                    Tensor<1,dim> Sv_grad_nminus1 = old_Sv_grads_nminus1[point];

                    if(use_exact_Sv_in_Sa)
                    {
                        Sv_fcn.set_time(time - time_step);

                        Sv_value_n = Sv_fcn.value(q_points[point]);
                        Sv_grad_n = Sv_fcn.gradient(q_points[point]);
                        Sv_fcn.set_time(time - 2.0*time_step);

                        Sv_value_nminus1 = Sv_fcn.value(q_points[point]);
                        Sv_grad_nminus1 = Sv_fcn.gradient(q_points[point]);
                    }

                    Tensor<1,dim> totalDarcyVelo = DarcyVelocities[point];

                    // Second order extrapolations if needed
                    double Sa_nplus1_extrapolation = Sa_value_n;
                    double Sv_nplus1_extrapolation = Sv_value_n;
                    Tensor<1,dim> Sv_grad_nplus1_extrapolation = Sv_grad_n;
                    Tensor<1,dim> totalDarcyVelo_extrapolation = totalDarcyVelo;

                    if(second_order_extrapolation)
                    {
                        Sa_nplus1_extrapolation *= 2.0;
                        Sa_nplus1_extrapolation -= Sa_value_nminus1;

                        Sv_nplus1_extrapolation *= 2.0;
                        Sv_nplus1_extrapolation -= Sv_value_nminus1;

                        Sv_grad_nplus1_extrapolation *= 2.0;
                        Sv_grad_nplus1_extrapolation -= Sv_grad_nminus1;

                    }

                    // Coefficients
                    double rho_l = rho_l_fcn.value(pl_value);
                    double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double rho_a = rho_a_fcn.value(pl_value);

                    if(incompressible)
                    {
                        rho_l = rho_v = rho_a = 1.0;
                    }

                    double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

//				double lambda_l = lambda_l_fcn.value(g_pl[point], g[point], Sv_nplus1_extrapolation);
//				double lambda_v = lambda_v_fcn.value(g_pl[point], g[point], Sv_nplus1_extrapolation);
//				double lambda_a = lambda_a_fcn.value(g_pl[point], g[point], Sv_nplus1_extrapolation);

                    double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

                    double dpca_dSa = cap_p_pca_fcn.derivative_wrt_Sa(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
//				    double dpca_dSa = cap_p_pca_fcn.derivative_wrt_Sa(g[point], Sv_nplus1_extrapolation);
                    double dpca_dSv = cap_p_pca_fcn.derivative_wrt_Sv(Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

                    double nu_h_artificial_visc = 0.0;
                    if(artificial_visc_exp || artificial_visc_imp)
                        nu_h_artificial_visc = 0.5*sqrt(cell->measure())*art_visc_multiple_Sa*maximum_Darcy*2.0*maximum_Sa;

                    double gamma_Sa_e = fabs(rho_a*lambda_a*kappa*dpca_dSa);

                    if(artificial_visc_imp)
                        gamma_Sa_e += nu_h_artificial_visc;

                    gamma_Sa_e += sqrt(totalDarcyVelo_extrapolation*totalDarcyVelo_extrapolation);

                    double h_e = cell->face(face_no)->measure();
                    double penalty_factor = (penalty_Sa_bdry/h_e) * gamma_Sa_e * degree*(degree + dim - 1);

                    // start of boundary terms
                    for (unsigned int i = 0; i < n_facet_dofs; ++i)
                    {
//                         //Boundary condition
                        copy_data.cell_rhs(i) += penalty_factor
                                                 * fe_face.shape_value(i, point)
                                                 * g[point]
                                                 * JxW[point];
                        // added to RHS - stab method
                        if (Stab_a)
                        {
                            copy_data.cell_rhs(i) -= (rho_a
                                                     * lambda_a
                                                     * dpca_dSa + Kappa_tilde_a)
                                                     * kappa
                                                     * Sa_grad_n
                                                     * normals[point]
                                                     * fe_face.shape_value(i, point)
                                                     * JxW[point];
                            copy_data.cell_rhs(i) -= theta_Sa
                                                     *(-Kappa_tilde_a)
                                                     * kappa
                                                     * fe_face.shape_grad(i, point)
                                                     * normals[point]
                                                     * g[point]
                                                     * JxW[point];
                        }
                        else
                        {
                            copy_data.cell_rhs(i) -= theta_Sa
                                                     * rho_a
                                                     * lambda_a
                                                     * dpca_dSa
                                                     * kappa
                                                     * fe_face.shape_grad(i, point)
                                                     * normals[point]
                                                     * g[point]
                                                     * JxW[point];
                        }
                        if(artificial_visc_imp) {
                            copy_data.cell_rhs(i) += theta_Sa
                                                     * nu_h_artificial_visc
                                                     * fe_face.shape_grad(i, point)
                                                     * normals[point]
                                                     * g[point]
                                                     * JxW[point];
                        }

                        // sv term for rhs
                        copy_data.cell_rhs(i) -= rho_a
                                                 * lambda_a
                                                 * dpca_dSv
                                                 * kappa
                                                 * Sv_grad_nplus1_extrapolation
                                                 * normals[point]
                                                 * fe_face.shape_value(i, point)
                                                 * JxW[point];

                        // Darcy velocity
                        if(project_only_kappa)
                        {
                            copy_data.cell_rhs(i) -= (rho_a*lambda_a)
                                                     * totalDarcyVelo_extrapolation
                                                     * normals[point]
                                                     * fe_face.shape_value(i, point)
                                                     * JxW[point];
                        }
                        else
                            copy_data.cell_rhs(i) -= (rho_a*lambda_a/rholambda_t)
                                                     * totalDarcyVelo_extrapolation
                                                     * normals[point]
                                                     * fe_face.shape_value(i, point)
                                                     * JxW[point];

                        // Gravity
                        if(!project_Darcy_with_gravity)
                            copy_data.cell_rhs(i) -= kappa*rho_a_fcn.value(pl_value)*rho_a*lambda_a
                                                     * gravity_fcn.vector_value(q_points[point])
                                                     * normals[point]
                                                     * fe_face.shape_value(i, point)
                                                     * JxW[point];

                        // Artificial viscosity term
                        if(artificial_visc_exp)
                            copy_data.cell_rhs(i) += nu_h_artificial_visc
                                                     * Sa_grad_n
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

                    if(use_exact_pl_in_Sa)
                    {
                        pl_fcn.set_time(time);

                        pl_value = pl_fcn.value(q_points[point]);
                        pl_grad = pl_fcn.gradient(q_points[point]);

                    }

                    double Sa_value_n = old_Sa_vals[point];
                    double Sa_value_nminus1 = old_Sa_vals_nminus1[point];
                    Tensor<1,dim> Sa_grad_n = old_Sa_grads[point];

                    double Sv_value_n = old_Sv_vals[point];
                    double Sv_value_nminus1 = old_Sv_vals_nminus1[point];

                    if(use_exact_Sv_in_Sa)
                    {
                        Sv_fcn.set_time(time - time_step);

                        Sv_value_n = Sv_fcn.value(q_points[point]);

                        Sv_fcn.set_time(time - 2.0*time_step);

                        Sv_value_nminus1 = Sv_fcn.value(q_points[point]);
                    }

                    // Second order extrapolations if needed
                    double Sa_nplus1_extrapolation = Sa_value_n;
                    double Sv_nplus1_extrapolation = Sv_value_n;

                    if(second_order_extrapolation)
                    {
                        Sa_nplus1_extrapolation *= 2.0;
                        Sa_nplus1_extrapolation -= Sa_value_nminus1;

                        Sv_nplus1_extrapolation *= 2.0;
                        Sv_nplus1_extrapolation -= Sv_value_nminus1;

                    }

                    Tensor<1,dim> neumann_term = neumann_fcn.vector_value(q_points[point]);
                    Tensor<1,dim> totalDarcyVelo = DarcyVelocities[point];

                    double rho_l = rho_l_fcn.value(pl_value);
                    double rho_v = rho_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double rho_a = rho_a_fcn.value(pl_value);

                    if(incompressible)
                    {
                        rho_l = rho_v = rho_a = 1.0;
                    }

                    double lambda_l = lambda_l_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double lambda_v = lambda_v_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);
                    double lambda_a = lambda_a_fcn.value(pl_value, Sa_nplus1_extrapolation, Sv_nplus1_extrapolation);

                    double rholambda_t = rho_l*lambda_l + rho_v*lambda_v + rho_a*lambda_a;

                    for (unsigned int i = 0; i < n_facet_dofs; ++i)
                    {
                     //   if(cell->face(face_no)->boundary_id() == 5 || cell->face(face_no)->boundary_id() == 6)

                            if(project_only_kappa)
                            {
                                copy_data.cell_rhs(i) -= (rho_a*lambda_a)
                                                         * totalDarcyVelo
                                                         * normals[point]
                                                         * fe_face.shape_value(i, point)
                                                         * JxW[point];
                            }
                            else

                            {
                                copy_data.cell_rhs(i) -= (rho_a*lambda_a/rholambda_t)
                                                         * totalDarcyVelo
                                                         * normals[point]
                                                         * fe_face.shape_value(i, point)
                                                         * JxW[point];
                            }
                            copy_data.cell_rhs(i) += neumann_term
                                                     * normals[point]
                                                     *fe_face.shape_value(i,point)
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

            //
            //copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);
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

            std::vector<double> old_Sa_vals(n_qpoints);
            std::vector<double> old_Sa_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sa_grads(n_qpoints);

            std::vector<double> old_Sa_vals_neighbor(n_qpoints);
            std::vector<double> old_Sa_vals_nminus1_neighbor(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sa_grads_neighbor(n_qpoints);

            std::vector<double> old_Sv_vals(n_qpoints);
            std::vector<double> old_Sv_vals_nminus1(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_nminus1(n_qpoints);

            std::vector<double> old_Sv_vals_neighbor(n_qpoints);
            std::vector<double> old_Sv_vals_nminus1_neighbor(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_neighbor(n_qpoints);
            std::vector<Tensor<1, dim>> old_Sv_grads_nminus1_neighbor(n_qpoints);

            fe_face.get_function_values(temp_pl_solution, pl_vals);
            fe_face_neighbor.get_function_values(temp_pl_solution, pl_vals_neighbor);

            fe_face.get_function_gradients(temp_pl_solution, pl_grads);
            fe_face_neighbor.get_function_gradients(temp_pl_solution, pl_grads_neighbor);

            fe_face.get_function_values(temp_Sa_solution_n, old_Sa_vals);
            fe_face.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1);
            fe_face.get_function_gradients(temp_Sa_solution_n, old_Sa_grads);

            fe_face_neighbor.get_function_values(temp_Sa_solution_n, old_Sa_vals_neighbor);
            fe_face_neighbor.get_function_values(temp_Sa_solution_nminus1, old_Sa_vals_nminus1_neighbor);
            fe_face_neighbor.get_function_gradients(temp_Sa_solution_n, old_Sa_grads_neighbor);

            fe_face.get_function_values(temp_Sv_solution_n, old_Sv_vals);
            fe_face.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1);
            fe_face.get_function_gradients(temp_Sv_solution_n, old_Sv_grads);
            fe_face.get_function_gradients(temp_Sv_solution_nminus1, old_Sv_grads_nminus1);

            fe_face_neighbor.get_function_values(temp_Sv_solution_n, old_Sv_vals_neighbor);
            fe_face_neighbor.get_function_values(temp_Sv_solution_nminus1, old_Sv_vals_nminus1_neighbor);
            fe_face_neighbor.get_function_gradients(temp_Sv_solution_n, old_Sv_grads_neighbor);
            fe_face_neighbor.get_function_gradients(temp_Sv_solution_nminus1, old_Sv_grads_nminus1_neighbor);

            std::vector<Tensor<1, dim>> DarcyVelocities(n_qpoints);
            fe_face_values_RT[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities);

            std::vector<Tensor<1, dim>> DarcyVelocities_neighbor(n_qpoints);
            fe_face_values_RT_neighbor[velocities].get_function_values(temp_totalDarcyVelocity_RT, DarcyVelocities_neighbor);

            // get maximum of Darcy Velocity
            std::vector<double> linf_norm_Darcy_vel0(n_qpoints);
            std::vector<double> linf_norm_Darcy_vel1(n_qpoints);

            for(unsigned int kk = 0; kk < n_qpoints; kk++)
            {
                Vector<double> darcy_v0(dim), darcy_v1(dim);
                for(unsigned int jj = 0; jj < dim; jj++)
                {
                    darcy_v0[jj] = DarcyVelocities[kk][jj];
                    darcy_v1[jj] = DarcyVelocities_neighbor[kk][jj];
                }

                linf_norm_Darcy_vel0[kk] = darcy_v0.linfty_norm();
                linf_norm_Darcy_vel1[kk] = darcy_v1.linfty_norm();
            }

            double maximum_Darcy0 = *std::max_element(linf_norm_Darcy_vel0.begin(), linf_norm_Darcy_vel0.end());
            double maximum_Darcy1 = *std::max_element(linf_norm_Darcy_vel1.begin(), linf_norm_Darcy_vel1.end());

            double maximum_Sa0 = *std::max_element(old_Sa_vals.begin(), old_Sa_vals.end());
            double maximum_Sa1 = *std::max_element(old_Sa_vals_neighbor.begin(), old_Sa_vals_neighbor.end());

            double kappa0 = temp_kappa[cell->global_active_cell_index()];
            double kappa1 = temp_kappa[ncell->global_active_cell_index()];

            for (unsigned int point = 0; point < n_qpoints; ++point)
            {
                // Get pl, sa and sv values on current integration point.
                // The 0 indicates current element, and 1 indicates neighboring element.
                double pl_value0 = pl_vals[point];
                double pl_value1 = pl_vals_neighbor[point];

                Tensor<1,dim> pl_grad0 = pl_grads[point];
                Tensor<1,dim> pl_grad1 = pl_grads_neighbor[point];

                if(use_exact_pl_in_Sa)
                {
                    pl_fcn.set_time(time);

                    pl_value0 = pl_fcn.value(q_points[point]);
                    pl_value1 = pl_value0;

                    pl_grad0 = pl_fcn.gradient(q_points[point]);
                    pl_grad1 = pl_grad0;
                }

                double Sa_value0_n = old_Sa_vals[point];
                double Sa_value1_n = old_Sa_vals_neighbor[point];
                Tensor<1,dim> Sa_grad0_n = old_Sa_grads[point];
                Tensor<1,dim> Sa_grad1_n = old_Sa_grads_neighbor[point];

                double Sa_value0_nminus1 = old_Sa_vals_nminus1[point];
                double Sa_value1_nminus1 = old_Sa_vals_nminus1_neighbor[point];

                double Sv_value0_n = old_Sv_vals[point];
                double Sv_value1_n = old_Sv_vals_neighbor[point];
                double Sv_value0_nminus1 = old_Sv_vals_nminus1[point];
                double Sv_value1_nminus1 = old_Sv_vals_nminus1_neighbor[point];

                Tensor<1,dim> Sv_grad0_n = old_Sv_grads[point];
                Tensor<1,dim> Sv_grad1_n = old_Sv_grads_neighbor[point];
                Tensor<1,dim> Sv_grad0_nminus1 = old_Sv_grads_nminus1[point];
                Tensor<1,dim> Sv_grad1_nminus1 = old_Sv_grads_nminus1_neighbor[point];

                if(use_exact_Sv_in_Sa)
                {
                    Sv_fcn.set_time(time - time_step);

                    Sv_value0_n = Sv_fcn.value(q_points[point]);
                    Sv_value1_n = Sv_value0_n;
                    Sv_grad0_n = Sv_fcn.gradient(q_points[point]);
                    Sv_grad1_n = Sv_grad0_n;

                    Sv_fcn.set_time(time - 2.0*time_step);

                    Sv_value0_nminus1 = Sv_fcn.value(q_points[point]);
                    Sv_value1_nminus1 = Sv_value0_nminus1;
                    Sv_grad0_nminus1 = Sv_fcn.gradient(q_points[point]);
                    Sv_grad1_nminus1 = Sv_grad0_nminus1;
                }

                // Darcy velocities
                Tensor<1,dim> totalDarcyVelo0 = DarcyVelocities[point];
                Tensor<1,dim> totalDarcyVelo1 = DarcyVelocities_neighbor[point];

                // Second order extrapolations if needed
                double Sa_nplus1_extrapolation0 = Sa_value0_n;
                double Sa_nplus1_extrapolation1 = Sa_value1_n;
                double Sv_nplus1_extrapolation0 = Sv_value0_n;
                double Sv_nplus1_extrapolation1 = Sv_value1_n;
                Tensor<1,dim> Sv_grad_nplus1_extrapolation0 = Sv_grad0_n;
                Tensor<1,dim> Sv_grad_nplus1_extrapolation1 = Sv_grad1_n;
                Tensor<1,dim> totalDarcyVelo_extrapolation0 = totalDarcyVelo0;
                Tensor<1,dim> totalDarcyVelo_extrapolation1 = totalDarcyVelo1;

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

                    Sv_grad_nplus1_extrapolation0 *= 2.0;
                    Sv_grad_nplus1_extrapolation0 -= Sv_grad0_nminus1;

                    Sv_grad_nplus1_extrapolation1 *= 2.0;
                    Sv_grad_nplus1_extrapolation1 -= Sv_grad1_nminus1;

                }

                // Coefficients
                double rho_l0 = rho_l_fcn.value(pl_value0);
                double rho_l1 = rho_l_fcn.value(pl_value1);

                double rho_v0 = rho_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double rho_v1 = rho_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

                double rho_a0 = rho_a_fcn.value(pl_value0);
                double rho_a1 = rho_a_fcn.value(pl_value1);

                if(incompressible)
                {
                    rho_l0 = rho_v0 = rho_a0 = 1.0;
                    rho_l1 = rho_v1 = rho_a1 = 1.0;
                }

                double lambda_l0 = lambda_l_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double lambda_v0 = lambda_v_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double lambda_a0 = lambda_a_fcn.value(pl_value0, Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);

                double lambda_l1 = lambda_l_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
                double lambda_v1 = lambda_v_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);
                double lambda_a1 = lambda_a_fcn.value(pl_value1, Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

                double rholambda_t0 = rho_l0*lambda_l0 + rho_v0*lambda_v0 + rho_a0*lambda_a0;
                double rholambda_t1 = rho_l1*lambda_l1 + rho_v1*lambda_v1 + rho_a1*lambda_a1;

                double dpca_dSa0 = cap_p_pca_fcn.derivative_wrt_Sa(Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double dpca_dSa1 = cap_p_pca_fcn.derivative_wrt_Sa(Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

                double dpca_dSv0 = cap_p_pca_fcn.derivative_wrt_Sv(Sa_nplus1_extrapolation0, Sv_nplus1_extrapolation0);
                double dpca_dSv1 = cap_p_pca_fcn.derivative_wrt_Sv(Sa_nplus1_extrapolation1, Sv_nplus1_extrapolation1);

                double nu_h_artificial_visc0 = 0.0;
                double nu_h_artificial_visc1 = 0.0;

                if(artificial_visc_exp || artificial_visc_imp)
                {
                    nu_h_artificial_visc0 = 0.5*sqrt(cell->measure())*art_visc_multiple_Sa*maximum_Darcy0*2.0*maximum_Sa0;
                    nu_h_artificial_visc1 = 0.5*sqrt(ncell->measure())*art_visc_multiple_Sa*maximum_Darcy1*2.0*maximum_Sa1;
                }

                // Diffusion coefficients and weights for stab method
                double coef0_diff = fabs(rho_a0*lambda_a0*kappa0*dpca_dSa0);
                double coef1_diff = fabs(rho_a1*lambda_a1*kappa1*dpca_dSa1);

                // for stabilization term
                double coef0_diff_stab = fabs(kappa0*Kappa_tilde_a);
                double coef1_diff_stab = fabs(kappa1*Kappa_tilde_a);

                if(artificial_visc_imp)
                {
                    coef0_diff += nu_h_artificial_visc0;
                    coef1_diff += nu_h_artificial_visc1;
                }

                double gamma_Sa_e = fabs(2.0*coef0_diff*coef1_diff/(coef0_diff + coef1_diff + 1.e-20));

                double h_e = cell->face(f)->measure();
                double penalty_factor = (penalty_Sa/h_e) * gamma_Sa_e * degree*(degree + dim - 1);

                double weight0_diff = coef1_diff/(coef0_diff + coef1_diff + 1.e-20);
                double weight1_diff = coef0_diff/(coef0_diff + coef1_diff + 1.e-20);

                // for stabilization term
                double weight0_diff_stab = coef1_diff_stab/(coef0_diff_stab + coef1_diff_stab + 1.e-20);
                double weight1_diff_stab = coef0_diff_stab/(coef0_diff_stab + coef1_diff_stab + 1.e-20);

                // Sv coefficients and weights
                double coef0_Sv = rho_a0*lambda_a0*dpca_dSv0*kappa0;
                double coef1_Sv = rho_a1*lambda_a1*dpca_dSv1*kappa1;

                double weight0_Sv = coef1_Sv/(coef0_Sv + coef1_Sv + 1.e-20);
                double weight1_Sv = coef0_Sv/(coef0_Sv + coef1_Sv + 1.e-20);

                //Sa coefficients and weights for stab method
                double coef0_Sa_stab = (rho_a0*lambda_a0*dpca_dSa0+Kappa_tilde_a)*kappa0;
                double coef1_Sa_stab = (rho_a1*lambda_a1*dpca_dSa1+Kappa_tilde_a)*kappa1;
                //TEST DEBUG LC --> BAD
                //double coef0_Sa_stab = fabs(rho_a0*lambda_a0*dpca_dSa0+Kappa_tilde_a)*kappa0;
                //double coef1_Sa_stab = fabs(rho_a1*lambda_a1*dpca_dSa1+Kappa_tilde_a)*kappa1;

                double weight0_Sa_stab = coef1_Sa_stab/(coef0_Sa_stab + coef1_Sa_stab + 1.e-20);
                double weight1_Sa_stab = coef0_Sa_stab/(coef0_Sa_stab + coef1_Sa_stab + 1.e-20);

                // start of interior face terms
                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    if(Stab_a)
                    {
                        // Sa term added to the RHS
                        double weighted_aver_rhs0_stab = AverageGradOperators::weighted_average_rhs<dim>(normals[point],
                                                                                                         Sa_grad0_n, Sa_grad1_n,
                                                                                                         coef0_Sa_stab, coef1_Sa_stab,
                                                                                                         weight0_Sa_stab, weight1_Sa_stab);

                        copy_data_face.cell_rhs(i) -=
                                weighted_aver_rhs0_stab
                                * fe_iv.jump(i, point)
                                * JxW[point];
                    }
                    // Sv term added to RHS
                    double weighted_aver_rhs1 = AverageGradOperators::weighted_average_rhs<dim>(normals[point],
                                                                                                Sv_grad_nplus1_extrapolation0, Sv_grad_nplus1_extrapolation1,
                                                                                                coef0_Sv, coef1_Sv,
                                                                                                weight0_Sv, weight1_Sv);
                    copy_data_face.cell_rhs(i) -=
                            weighted_aver_rhs1
                            * fe_iv.jump(i, point)
                            * JxW[point];

                    // Darcy velocity and upwind stuff
                    Tensor<1,dim> g_val = gravity_fcn.vector_value(q_points[point]);
                    double coef0_darcy, coef1_darcy;

                    if(project_only_kappa)
                    {
                        coef0_darcy = rho_a0*lambda_a0;
                        coef1_darcy = rho_a1*lambda_a1;
                    }
                    else
                    {
                        coef0_darcy = rho_a0*lambda_a0/rholambda_t0;
                        coef1_darcy = rho_a1*lambda_a1/rholambda_t1;
                    }

                    double Dg0 = kappa0*rho_a_fcn.value(pl_value0)*rho_a0*lambda_a0;
                    double Dg1 = kappa1*rho_a_fcn.value(pl_value1)*rho_a1*lambda_a1;

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
                        coef0_g = kappa0*rho_a_fcn.value(pl_value0)*rho_a0*lambda_a0;
                        coef1_g = kappa1*rho_a_fcn.value(pl_value1)*rho_a1*lambda_a1;

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

                    // Artificial viscosity term
                    if(artificial_visc_exp)
                    {
                        double coef0_nu = nu_h_artificial_visc0;
                        double coef1_nu = nu_h_artificial_visc1;

                        double weight0_nu = coef1_nu/(coef0_nu + coef1_nu + 1.e-20);
                        double weight1_nu = coef0_nu/(coef0_nu + coef1_nu + 1.e-20);

                        double weighted_aver_rhs_nu = AverageGradOperators::weighted_average_rhs(normals[point],
                                                                                                 Sa_grad0_n, Sa_grad1_n,
                                                                                                 coef0_nu, coef1_nu,
                                                                                                 weight0_nu, weight1_nu);

                        copy_data_face.cell_rhs(i) += weighted_aver_rhs_nu
                                                      * fe_iv.jump(i, point)
                                                      * JxW[point];
                  }
                }
            }
        };

        const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(/*c.cell_matrix,*/
                    c.cell_rhs,
                    c.local_dof_indices,
                    /*system_matrix_aqueous_saturation*/
                     right_hand_side_aqueous_saturation);

            for (auto &cdf : c.face_data)
            {
                constraints.distribute_local_to_global(/*cdf.cell_matrix,*/
                        cdf.cell_rhs,
                                                       cdf.joint_dof_indices,
                                                       /*system_matrix_aqueous_saturation*/
                        right_hand_side_aqueous_saturation);
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

        right_hand_side_aqueous_saturation.compress(VectorOperation::add);
    }



    template <int dim>
    void AqueousSaturationProblem<dim>::solve_aqueous_saturation(PETScWrappers::MPI::SparseMatrix &mat)
    {
//	std::map<types::global_dof_index, double> boundary_values;
//	        VectorTools::interpolate_boundary_values(dof_handler,
//	                                                 1,
//													 BoundaryValuesAqueousSaturation<dim>(),
//	                                                 boundary_values);
//		MatrixTools::apply_boundary_values(
//	          boundary_values, system_matrix_aqueous_saturation, Sa_solution, right_hand_side_aqueous_saturation, false);


        if(use_direct_solver)
        {
            SolverControl cn;
            PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
            //	solver.set_symmetric_mode(true);
            solver.solve( mat, Sa_solution, right_hand_side_aqueous_saturation);


        }
        else
        {
            SolverControl solver_control(pl_solution.size(), 1.e-7 * right_hand_side_aqueous_saturation.l2_norm());

            PETScWrappers::SolverGMRES gmres(solver_control, mpi_communicator);
            PETScWrappers::PreconditionBoomerAMG preconditioner(mat);

            gmres.solve(/*mat*/ mat,Sa_solution, right_hand_side_aqueous_saturation, preconditioner);

            Vector<double> localized_solution(Sa_solution);
            constraints.distribute(localized_solution);

            Sa_solution = localized_solution;
        }
    }
} // namespace AqueousSaturation

#endif //SA_PROBLEM_HH
