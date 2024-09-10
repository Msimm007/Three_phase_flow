
#ifndef THREEPHASE_SA_RHS_HH
#define THREEPHASE_SA_RHS_HH

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


// The following regards the stability method.
namespace AqueousSaturationRHS
{
    struct CopyDataFace
    {
        FullMatrix<double>                   cell_matrix;
        Vector<double>                       cell_rhs;
        std::vector<types::global_dof_index> joint_dof_indices;
    };

    struct CopyData
    {
        FullMatrix<double>                   cell_matrix;
        Vector<double>                       cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        std::vector<CopyDataFace>            face_data;

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
    class AqueousSaturationRHS{
    public:
        AqueousSaturationRHS(Triangulation<dim, dim> &triangulation_,
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

        void assemble_rhs_aqueous_saturation();
    private:

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
    AqueousSaturationRHS<dim>::AqueousSaturationRHS(Triangulation<dim, dim> &triangulation_, const unsigned int degree_,
    double time_step_, double theta_Sa_, double penalty_Sa_,
    double penalty_Sa_bdry_, std::vector<unsigned int> dirichlet_id_sa_,
    bool use_exact_pl_in_Sa_, bool use_exact_Sv_in_Sa_, double time_,
    unsigned int timestep_number_, bool second_order_time_derivative_,
    bool second_order_extrapolation_, bool use_direct_solver_,
    bool Stab_a_, bool incompressible_,
    bool project_Darcy_with_gravity_, bool artificial_visc_exp_,
    bool artificial_visc_imp_, double art_visc_multiple_Sa_,
    PETScWrappers::MPI::Vector pl_solution_,
    PETScWrappers::MPI::Vector pl_solution_n_,
    PETScWrappers::MPI::Vector pl_solution_nminus1_,
    PETScWrappers::MPI::Vector Sa_solution_n_,
    PETScWrappers::MPI::Vector Sa_solution_nminus1_,
    PETScWrappers::MPI::Vector Sv_solution_n_,
    PETScWrappers::MPI::Vector Sv_solution_nminus1_,
    PETScWrappers::MPI::Vector kappa_abs_vec_,
    PETScWrappers::MPI::Vector totalDarcyvelocity_RT_,
    const unsigned int degreeRT_, bool project_only_kappa_,
    MPI_Comm mpi_communicator_, const unsigned int n_mpi_processes_,
    const unsigned int this_mpi_process_)

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
    void AqueousSaturationRHS<dim>::setup_rhs()
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

}



#endif //THREEPHASE_SA_RHS_HH
