#ifndef UTILITIES_HH
#define UTILITIES_HH

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
        prm.declare_entry("Final time", "1.0", Patterns::Double(0.0));
        prm.declare_entry("Initial time step", "1.0", Patterns::Double(0.0));

        prm.declare_entry("Second order time derivative", "false", Patterns::Bool());
        prm.declare_entry("Second order extrapolation", "false", Patterns::Bool());

        prm.declare_entry("Use implicit time term in pl", "false", Patterns::Bool());
        prm.declare_entry("Create initial perturbation", "false", Patterns::Bool());

        prm.declare_entry("Midpoint method", "false", Patterns::Bool());
        prm.declare_entry("Theta_n", "0.5", Patterns::Double(0.0));
        prm.declare_entry("delta_t div 4", "false", Patterns::Bool());
    }
    prm.leave_subsection();

    prm.enter_subsection("Spatial discretization parameters");
    {
        prm.declare_entry("Dimension", "2", Patterns::Integer(2));
        prm.declare_entry("Degree", "1", Patterns::Integer(1));
        prm.declare_entry("Initial level of refinement", "0", Patterns::Integer(0));
        prm.declare_entry("Final level of refinement", "7", Patterns::Integer(0));

        prm.declare_entry("Incompressible", "true", Patterns::Bool());
        prm.declare_entry("Stab_a", "true", Patterns::Bool());
        prm.declare_entry("Stab_v", "true", Patterns::Bool());

        prm.declare_entry("Theta pl","1.0",Patterns::Double());
		prm.declare_entry("Theta Sa","1.0",Patterns::Double());
		prm.declare_entry("Theta Sv","1.0",Patterns::Double());

        prm.declare_entry("Penalty pl","1.0",Patterns::Double(0.0));
		prm.declare_entry("Penalty Sa","1.0",Patterns::Double(0.0));
    	prm.declare_entry("Penalty Sv","1.0",Patterns::Double(0.0));

		prm.declare_entry("Penalty pl boundary","1.0",Patterns::Double(0.0));
		prm.declare_entry("Penalty Sa boundary","1.0",Patterns::Double(0.0));
		prm.declare_entry("Penalty Sv boundary","1.0",Patterns::Double(0.0));

		prm.declare_entry("Project to RT0","true",Patterns::Bool());
		prm.declare_entry("Project to RT0 with kappa only","true",Patterns::Bool());
		prm.declare_entry("Use direct solver for linear systems","true",Patterns::Bool());

		prm.declare_entry("Use exact pl in Sa","false",Patterns::Bool());
		prm.declare_entry("Use exact pl in Sv","false",Patterns::Bool());
		prm.declare_entry("Use exact pl in RT","false",Patterns::Bool());

		prm.declare_entry("Use exact Sa in pl","false",Patterns::Bool());
		prm.declare_entry("Use exact Sa in Sv","false",Patterns::Bool());
		prm.declare_entry("Use exact Sa in RT","false",Patterns::Bool());

		prm.declare_entry("Use exact Sv in pl","false",Patterns::Bool());
		prm.declare_entry("Use exact Sv in Sa","false",Patterns::Bool());
		prm.declare_entry("Use exact Sv in RT","false",Patterns::Bool());

		prm.declare_entry("Project Darcy with gravity","false",Patterns::Bool());
		prm.declare_entry("Add explicit artificial viscosity","false",Patterns::Bool());
		prm.declare_entry("Add implicit artificial viscosity","false",Patterns::Bool());
		prm.declare_entry("Artificial viscosity multiple for Sa","1.0",Patterns::Double(0.0));
		prm.declare_entry("Two phase problem","false",Patterns::Bool());

    }
    prm.leave_subsection();

    prm.enter_subsection("Output parameters");
    {
        prm.declare_entry("print vtk", "false", Patterns::Bool());
        prm.declare_entry("Frequency for vtk printing", "1", Patterns::Integer(1));
        prm.declare_entry("Compute errors", "false", Patterns::Bool());
        prm.declare_entry("Compute energy", "false", Patterns::Bool());
    }
    prm.leave_subsection();

    prm.enter_subsection("Continuation from previous solution");
    {
        prm.declare_entry("Output solution", "false", Patterns::Bool());
        prm.declare_entry("Frequency for solution output", "1", Patterns::Integer(1));
        prm.declare_entry("Continue solution", "false", Patterns::Bool());
        prm.declare_entry("Time step number", "1", Patterns::Integer(1));
    }
    prm.leave_subsection();
}

void ParameterReader::read_parameters(const std::string &parameter_file)
{
    declare_parameters();
    prm.parse_input(parameter_file);

}


} // namespace CouplingPressureSaturation



#endif // UTILITIES_HH
