
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


}



#endif //THREEPHASE_SA_RHS_HH
