#ifndef AVERAGEGRADIENTOPERATORS_HH
#define AVERAGEGRADIENTOPERATORS_HH

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/fe_face.h>

namespace AverageGradOperators
{
using namespace dealii;


template <int dim>
double weighted_average_gradient(const typename DoFHandler<dim>::active_cell_iterator &cell,
    		const unsigned int &f,
            const unsigned int &sf,
			const typename DoFHandler<dim>::active_cell_iterator &    ncell,
            const unsigned int &nf,
            const unsigned int &nsf,
//			ScratchData<dim> &  scratch_data,
			const FEInterfaceValues<dim> &fe_iv,
			Tensor<1,dim> normal_v,
			unsigned int index, unsigned int point,
			double coef0, double coef1,
			double weight0, double weight1)
{
	//fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

	const FEFaceValuesBase<dim> &fe_face = fe_iv.get_fe_face_values(0);
	const FEFaceValuesBase<dim> &fe_face_neighbor = fe_iv.get_fe_face_values(1);

	const auto dof_pair = fe_iv.interface_dof_to_dof_indices(index);

	double weighted_aver = 0.0;

	if (dof_pair[0] != numbers::invalid_unsigned_int)
		weighted_aver += weight0*coef0*fe_face.shape_grad(dof_pair[0], point)*normal_v;
	if (dof_pair[1] != numbers::invalid_unsigned_int)
		weighted_aver += weight1*coef1*fe_face_neighbor.shape_grad(dof_pair[1], point)*normal_v;

	return weighted_aver;
}

template <int dim>
double weighted_average_rhs(Tensor<1,dim> normal_v, Tensor<1,dim> grad0, Tensor<1,dim> grad1,
		double coef0, double coef1, double weight0, double weight1)
{
	double weighted_aver = 0.0;

	weighted_aver += weight0*coef0*grad0*normal_v;
	weighted_aver += weight1*coef1*grad1*normal_v;

	return weighted_aver;
}

};

#endif // AVERAGEGRADIENTOPERATORS_HH
