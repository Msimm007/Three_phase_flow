#ifndef AUX_PRIMARY_HH
#define AUX_PRIMARY_HH

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/full_matrix.h>

#include <cmath>
#include <iostream>
#include <fstream>

using namespace dealii;
extern double amp_factor_cap_pressure;

extern double stab_sa_data;
extern double stab_sv_data;

extern bool hetero;


// Mesh creator
template <int dim>
void create_mesh(Triangulation<dim, dim> &triangulation, unsigned int ref_level,
		std::vector<unsigned int> &dirichlet_id_pl,
		std::vector<unsigned int> &dirichlet_id_sa,
		std::vector<unsigned int> &dirichlet_id_sv)
{

	
	Triangulation<dim> triangulation1;

    Point<dim> v0;
    Point<dim> v1;

    v0[0] = 0.0;
    v0[1] = 0.0;

    if(dim == 3)
        v0[2] = 0.0;

    v1[0] = 50.0;
    v1[1] = 100.0;

    if(dim == 3)
        v1[2] = 100.0;

    std::vector<unsigned int> repetitions(dim);
    repetitions[0] = 10;
    repetitions[1] = 20;

    if (dim == 3)
        repetitions[2] = 20;

    std::vector<int> n_cells_to_remove(dim);
    n_cells_to_remove[0] = 1;
    n_cells_to_remove[1] = 1;

    if (dim == 3)
        n_cells_to_remove[2] = -1;

    GridGenerator::subdivided_hyper_L(triangulation1, repetitions, v0, v1, n_cells_to_remove);

    //*---------------------------------
    Triangulation<dim> triangulation2;

    Point<dim> v0_2;
    Point<dim> v1_2;

    v0_2[0] = 50.0;
    v0_2[1] = 0.0;

    if (dim == 3)
        v0_2[2] = 0.0;

    v1_2[0] = 100.0;
    v1_2[1] = 100.0;

    if (dim == 3)
        v1_2[2] = 100.0;

    std::vector<unsigned int> repetitions_2(dim);
    repetitions_2[0] = 10;
    repetitions_2[1] = 20;

    if (dim == 3)
        repetitions_2[2] = 20;

    std::vector<int> n_cells_to_remove_2(dim);

    n_cells_to_remove_2[0] = -1;
    n_cells_to_remove_2[1] = -1;

    if (dim == 3)
        n_cells_to_remove_2[2] = 1;

    GridGenerator::subdivided_hyper_L(triangulation2, repetitions_2, v0_2, v1_2, n_cells_to_remove_2);

    GridGenerator::merge_triangulations({&triangulation1,&triangulation2},triangulation);

	triangulation.refine_global(ref_level);
 
	// Boundary classification
	// 1: 0 < x < 5, y = 5
	// 2: x = 5, 0 < y < 5
	// 3: 5 < x < 100, y = 0
	// 4: x = 100, 0 < y < 95
	// 5: 95 < x < 100, y = 95
	// 6: x = 95, 95 < y < 100
	// 7: 0 < x < 95, y = 100
	// 8: x = 0, 5 < y < 100

	typename Triangulation<dim>::active_cell_iterator
		cell = triangulation.begin_active(),
		endc = triangulation.end();

	for (; cell != endc; cell++)
	{
		for (unsigned int face_no=0; face_no < GeometryInfo<dim>::faces_per_cell; face_no++)
		{
//    		pcout << "face_no = " << face_no << std::endl;
			if(cell->face(face_no)->at_boundary())
			{
				bool bdr_1, bdr_2, bdr_3, bdr_4, bdr_5, bdr_6, bdr_7, bdr_8;
				bdr_1 = bdr_2 = bdr_3 = bdr_4 = bdr_5 = bdr_6 = bdr_7 = bdr_8 = false;

				 for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
				 {
					 Point<dim> &v = cell->face(face_no)->vertex(i);

					 // 0 < x < 5, y = 5
					 if(0.0 < v[0] + 1.e-12 && v[0] - 1.e-12 < 5.0 && fabs(v[1] - 5.0) < 1.e-12)
						 bdr_1 = true;
					 else
					 {
						 bdr_1 = false;
						 break;
					 }

				 }

				 for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
				 {
					 Point<dim> &v = cell->face(face_no)->vertex(i);

					 // x = 5, 0 < y < 5
					 if(fabs(v[0] - 5.0) < 1.e-12 && 0.0 < v[1] + 1.e-12 && v[1] - 1.e-12 < 5.0)
						 bdr_2 = true;
					 else
					 {
						 bdr_2 = false;
						 break;
					 }

				 }

				 for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
				 {
					 Point<dim> &v = cell->face(face_no)->vertex(i);

//					 // 5 < x < 100, y = 0
					 if(fabs(v[1] - 0.0) < 1.e-12 && 5.0 < v[0] + 1.e-12 && v[0] - 1.e-12 < 100.0)
						 bdr_3 = true;
					 else
					 {
						 bdr_3 = false;
						 break;
					 }

				 }

				 for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
				 {
					 Point<dim> &v = cell->face(face_no)->vertex(i);

					 // x = 100, 0 < y < 95
					 if(fabs(v[0] - 100.0) < 1.e-12 && 0.0 < v[1] + 1.e-12 && v[1] - 1.e-12 < 95.0)
						 bdr_4 = true;
					 else
					 {
						 bdr_4 = false;
						 break;
					 }

				 }

				 for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
				 {
					 Point<dim> &v = cell->face(face_no)->vertex(i);

					 // 95 < x < 100, y = 95
					 if(fabs(v[1] - 95.0) < 1.e-12 && 95.0 < v[0] + 1.e-12 && v[0] - 1.e-12 < 100.0)
						 bdr_5 = true;
					 else
					 {
						 bdr_5 = false;
						 break;
					 }

				 }

				 for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
				 {
					 Point<dim> &v = cell->face(face_no)->vertex(i);

					 // x = 95, 95 < y < 100
					 if(fabs(v[0] - 95.0) < 1.e-12 && 95.0 < v[1] + 1.e-12 && v[1] - 1.e-12 < 100.0)
						 bdr_6 = true;
					 else
					 {
						 bdr_6 = false;
						 break;
					 }

				 }

				 for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
				 {
					 Point<dim> &v = cell->face(face_no)->vertex(i);

//					 // 0 < x < 95, y = 100
					 if(fabs(v[1] - 100.0) < 1.e-12 && 0.0 < v[0] + 1.e-12 && v[0] - 1.e-12 < 95.0)
						 bdr_7 = true;
					 else
					 {
						 bdr_7 = false;
						 break;
					 }

				 }

				 for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
				 {
					 Point<dim> &v = cell->face(face_no)->vertex(i);

//					 // x = 0, 5 < y < 100
					 if(fabs(v[0] - 0.0) < 1.e-12 && 5.0 < v[1] + 1.e-12 && v[1] - 1.e-12 < 100.0)
						 bdr_8 = true;
					 else
					 {
						 bdr_8 = false;
						 break;
					 }

				 }

				 if(bdr_1)
					 cell->face(face_no)->set_boundary_id(1);
				 else if(bdr_2)
					 cell->face(face_no)->set_boundary_id(2);
				 else if(bdr_3)
					 cell->face(face_no)->set_boundary_id(3);
				 else if(bdr_4)
					 cell->face(face_no)->set_boundary_id(4);
				 else if(bdr_5)
					 cell->face(face_no)->set_boundary_id(5);
				 else if(bdr_6)
					 cell->face(face_no)->set_boundary_id(6);
				 else if(bdr_7)
					 cell->face(face_no)->set_boundary_id(7);
				 else if(bdr_8)
					 cell->face(face_no)->set_boundary_id(8);
//				 if(bdr_1 || bdr_2 || bdr_5 || bdr_6)
//					 cell->face(face_no)->set_boundary_id(1);
//				 else
//					 cell->face(face_no)->set_boundary_id(2);
			}
		}
	}
	// Dirichlet boundary attributes for each problem
	dirichlet_id_pl.resize(4);
	dirichlet_id_pl[0] = 1;
	dirichlet_id_pl[1] = 2;
	dirichlet_id_pl[2] = 5;
	dirichlet_id_pl[3] = 6;

    dirichlet_id_sa.resize(2);
    dirichlet_id_sa[0] = 1;
    dirichlet_id_sa[1] = 2;
//    dirichlet_id_sa[2] = 5;
//    dirichlet_id_sa[3] = 6;

//	dirichlet_id_pl.resize(2);
//	dirichlet_id_pl[0] = 4;
//	dirichlet_id_pl[1] = 5;
//
//    dirichlet_id_sa.resize(4);
//    dirichlet_id_sa[0] = 4;
//    dirichlet_id_sa[1] = 5;

    dirichlet_id_sv.resize(1);
    dirichlet_id_sv[0] = 0;
}

// This function create and prints the initial perturbation to a file. It can be empty if not used
// NOTE: It only works when the code is run with only one processor.
// TODO: figure out_NIPDG_SvSa_both_sides_Kappa_1_delta_4 how to print the file in parallel
template <int dim>
void create_initial_Sa_vector(Triangulation<dim, dim> &triangulation, MPI_Comm mpi_communicator,
		const unsigned int n_mpi_processes, const unsigned int this_mpi_process)
{
	
}

// This function creates the value of kappa according to the cell
template <int dim>
double compute_kappa_value(const typename DoFHandler<dim>::active_cell_iterator &cell)
{
//	double kappa_abs = 3.72e-13;
	double kappa_abs = 5.e-8;

	if (hetero)
	{	
		double xx = cell->center()[0];
		double yy = cell->center()[1];


		double zz;
		if(dim == 3){
			zz = cell->center()[2];
		}


		if(xx >= 25.0 && xx <= 50.0)
			if((yy >= 25.0 && yy <= 50.0))
				if((zz >= 25.0 && zz <= 50.0))
					kappa_abs /= 1000.0;
	}
	return kappa_abs;
};



template <int dim>
class StabAqueousSaturation : public Function<dim>
{
public:
    StabAqueousSaturation()
            : Function<dim>(1)
    {}

    virtual double value() const;
};
template <int dim>
double StabAqueousSaturation<dim>::value()const
{

    return stab_sa_data;
}

template <int dim>
class StabVaporSaturation : public Function<dim>
{
public:
    StabVaporSaturation()
            : Function<dim>(1)
    {}

    virtual double value() const;
};
template <int dim>
double StabVaporSaturation<dim>::value()const
{
    return stab_sv_data;
}

template <int dim>
class ExactLiquidPressure : public Function<dim>
{
public:
	ExactLiquidPressure()
        : Function<dim>(1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
						 
};

template <int dim>
double
ExactLiquidPressure<dim>::value(const Point<dim> &p,
                          const unsigned int /*component*/) const
{
	if(this->get_time() == 0.0)
	{
//		if(0.0 + 1.e-12 < p[0] && p[0] - 1.e-12 < 10.0 && 0.0 + 1.e-12 < p[1] && p[1] - 1.e-12 < 10.0)
//			return 3.e5;
//		else
			return 1.0e5;
	}

	if(0.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 5.0 && fabs(p[1] - 5.0) < 1.e-12)
		return 3.e5;
	else if(fabs(p[0] - 5.0) < 1.e-12 && 0.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 5.0)
		return 3.e5;
	else if(fabs(p[1] - 95.0) < 1.e-12 && 95.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 100.0)
		return 1.0e5;
	else if(fabs(p[0] - 95.0) < 1.e-12 && 95.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 100.0)
		return 1.0e5;

//	if(fabs(p[1] - 100.0) < 1.e-12 && 95.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 100.0)
//		return 1.5e5;
//	else if(fabs(p[0] - 100.0) < 1.e-12 && 95.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 100.0)
//		return 1.5e5;

	return 1.0e5;

}

template <int dim>
class ExactVaporSaturation : public Function<dim>
{
public:
	ExactVaporSaturation()
        : Function<dim>(1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
						 
};

template <int dim>
double
ExactVaporSaturation<dim>::value(const Point<dim> &p,
                          const unsigned int /*component*/) const
{
	return 0.0;
}

template <int dim>
class ExactAqueousSaturation : public Function<dim>
{
public:
	ExactAqueousSaturation()
        : Function<dim>(1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
						 
};

template <int dim>
double
ExactAqueousSaturation<dim>::value(const Point<dim> &p,
                          const unsigned int /*component*/) const
{
	if(this->get_time() == 0)
		return 0.2;

	if(0.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 5.0 && fabs(p[1] - 5.0) < 1.e-12)
		return std::min(0.7,0.2 + 1.e-2*this->get_time()*0.5);
	else if(fabs(p[0] - 5.0) < 1.e-12 && 0.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 5.0)
		return std::min(0.7,0.2 + 1.e-2*this->get_time()*0.5);

//	if(fabs(p[1] - 100.0) < 1.e-12 && 95.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 100.0)
//		return 0.1;
//	else if(fabs(p[0] - 100.0) < 1.e-12 && 95.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 100.0)
//		return 0.1;

	return 0.1;
}

// pl at t=0
template <int dim>
class InitialValuesLiquidPressure : public Function<dim>
{
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        ExactLiquidPressure<dim> exact_pressure;
        exact_pressure.set_time(0.0);

        return exact_pressure.value(p);
    }
};

// pl at t=tau
template <int dim>
class InitialValuesLiquidPressure_dt : public Function<dim>
{
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        ExactLiquidPressure<dim> exact_pressure;
        exact_pressure.set_time(0.0);

        return exact_pressure.value(p);
    }
};

// sv at t=0
template <int dim>
class InitialValuesVaporSaturation : public Function<dim>
{
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        ExactVaporSaturation<dim> exact_vapor_sat;
        exact_vapor_sat.set_time(0.0);

        return exact_vapor_sat.value(p);
    }
};

// sv at t=tau
template <int dim>
class InitialValuesVaporSaturation_dt : public Function<dim>
{
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        ExactVaporSaturation<dim> exact_vapor_sat;
        exact_vapor_sat.set_time(0.0);

        return exact_vapor_sat.value(p);
    }
};

// sa at t=0
template <int dim>
class InitialValuesAqueousSaturation : public Function<dim>
{
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        ExactAqueousSaturation<dim> exact_aqueous_sat;
        exact_aqueous_sat.set_time(0.0);

        return exact_aqueous_sat.value(p);
    }
};

// sa at t=tau
template <int dim>
class InitialValuesAqueousSaturation_dt : public Function<dim>
{
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        ExactAqueousSaturation<dim> exact_aqueous_sat;
        exact_aqueous_sat.set_time(0.0);

        return exact_aqueous_sat.value(p);
    }
};


// Auxiliary functions
template<int dim>
double ComputeSl(const double pl, const double Sa, const double Sv)
{
	return 1.0 - Sa - Sv;
}

// Capillary pressures
template <int dim>
class CapillaryPressurePcv : public Function<dim>
{
public:
	CapillaryPressurePcv()
        : Function<dim>(1)
    {}

    virtual double value(double Sv,
                         const unsigned int component = 0) const;

    virtual Tensor<1,dim> num_gradient(double Sv,
    							   	   const Tensor<1,dim> grad_Sv,
									   const unsigned int component = 0) const;

    virtual double derivative_wrt_Sv(double Sv,
            						 const unsigned int component = 0) const;
};

template <int dim>
double
CapillaryPressurePcv<dim>::value(double Sv,
                          const unsigned int /*component*/) const
{
	Sv = std::min(1.0, std::max(Sv, 0.0));

	return (3.9/log(0.01))*log(1.0 - Sv + 0.01);
//	return (10.3/log(0.01))*log(1.0 - Sv + 0.01);
}

template <int dim>
Tensor<1,dim>
CapillaryPressurePcv<dim>::num_gradient(double Sv,
									const Tensor<1,dim> grad_Sv,
								    const unsigned int /*component*/) const
{
	Sv = std::min(1.0, std::max(Sv, 0.0));

	double dpcv_dSv = derivative_wrt_Sv(Sv);

	Tensor<1,dim> grad_pcv = grad_Sv;
	grad_pcv *= dpcv_dSv;

	return grad_pcv;
}

template <int dim>
double
CapillaryPressurePcv<dim>::derivative_wrt_Sv(double Sv,
                          	  	  	  	  	 const unsigned int /*component*/) const
{
	Sv = std::min(1.0, std::max(Sv, 0.0));

	return -(3.9/log(0.01))*(1.0/(1.0 - Sv + 0.01));
//	return -(10.3/log(0.01))*(1.0/(1.0 - Sv + 0.01));
}

template <int dim>
class CapillaryPressurePca : public Function<dim>
{
public:
	CapillaryPressurePca()
        : Function<dim>(1)
    {}

    virtual double value(double Sa, double Sv,
                         const unsigned int component = 0) const;

    virtual Tensor<1,dim> num_gradient(double Sa, double Sv,
    							   const Tensor<1,dim> grad_Sa,
								   const Tensor<1,dim> grad_Sv,
								   const unsigned int component = 0) const;

    virtual double derivative_wrt_Sa(double Sa, double Sv,
    								 const unsigned int component = 0) const;

    virtual double derivative_wrt_Sv(double Sa, double Sv,
    								 const unsigned int component = 0) const;
};

template <int dim>
double
CapillaryPressurePca<dim>::value(double Sa, double Sv,
                          const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

//	return 1.e-2*sqrt(0.205/3.72e-13)*(1.0 - Sa)*(1.0 - Sa);
//	if(Sa > 0.05)
//		return amp_factor_cap_pressure/sqrt(Sa);
//	else
//		return amp_factor_cap_pressure*(1.5 - 10.0*Sa)/sqrt(0.05);

	return 5.e3*pow(Sa,-1/3);
}

template <int dim>
Tensor<1,dim>
CapillaryPressurePca<dim>::num_gradient(double Sa, double Sv,
									const Tensor<1,dim> grad_Sa,
									const Tensor<1,dim> grad_Sv,
								    const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

	double dpca_dSv = derivative_wrt_Sv(Sa, Sv);
	double dpca_dSa = derivative_wrt_Sa(Sa, Sv);

	Tensor<1,dim> grad_pca = grad_Sv;
	grad_pca *= dpca_dSv;

	Tensor<1,dim> Sa_term = grad_Sa;
	Sa_term *= dpca_dSa;

	grad_pca += Sa_term;

	return grad_pca;
}

template <int dim>
double
CapillaryPressurePca<dim>::derivative_wrt_Sa(double Sa, double Sv,
                          const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

//	return -2.0*1.e-2*sqrt(0.205/3.72e-13)*(1.0 - Sa);
//	if(Sa > 0.05)
//		return -amp_factor_cap_pressure*0.5*pow(Sa, -1.5);
//	else
//		return -amp_factor_cap_pressure*10.0/sqrt(0.05);
	return -5.e3/3*pow(Sa,-4/3);
}

template <int dim>
double
CapillaryPressurePca<dim>::derivative_wrt_Sv(double Sa, double Sv,
                          const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

	return 0.0;
}

template <int dim>
class VaporPressure : public Function<dim>
{
public:
	VaporPressure()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, double Sa, double Sv,
                         const unsigned int component = 0) const;

};

template <int dim>
double
VaporPressure<dim>::value(const double pl, double Sa, double Sv,
                          const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

	CapillaryPressurePcv<dim> pcv;

	return pl + pcv.value(Sv);
}

template <int dim>
class AqueousPressure : public Function<dim>
{
public:
	AqueousPressure()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, double Sa, double Sv,
                         const unsigned int component = 0) const;

    virtual Tensor<1,dim> num_gradient(double Sa, double Sv,
    								   const Tensor<1,dim> grad_pl,
        							   const Tensor<1,dim> grad_Sa,
    								   const Tensor<1,dim> grad_Sv,
    								   const unsigned int component = 0) const;

};

template <int dim>
double
AqueousPressure<dim>::value(const double pl, double Sa, double Sv,
                          const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

	CapillaryPressurePca<dim> pca;

	return pl - pca.value(Sa, 0.0);
}

template <int dim>
Tensor<1,dim>
AqueousPressure<dim>::num_gradient(double Sa, double Sv, const Tensor<1,dim> grad_pl,
		   	   	   	      const Tensor<1,dim> grad_Sa, const Tensor<1,dim> grad_Sv,
                          const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

	CapillaryPressurePca<dim> pca;

	Tensor<1,dim> result;

	result = grad_pl;

	Tensor<1,dim> grad_pca = pca.num_gradient(Sa, Sv, grad_Sa, grad_Sv);

	result -= grad_pca;

	return result;
}

template <int dim>
class porosity : public Function<dim>
{
public:
	porosity()
        : Function<dim>(1)
    {}

    virtual double value(const double pl,
                         const unsigned int component = 0) const;

    virtual double derivative_wrt_pl(const double pl, const unsigned int component = 0) const;
};

template <int dim>
double
porosity<dim>::value(const double pl,
                     const unsigned int /*component*/) const
{
	return 0.2;
}

template <int dim>
double porosity<dim>::derivative_wrt_pl(const double pl,
                         	 	 	 	const unsigned int /*component*/) const
{
	return 0.0;
}

// Densities
template <int dim>
class rho_l : public Function<dim>
{
public:
	rho_l()
        : Function<dim>(1)
    {}

    virtual double value(const double pl,
                         const unsigned int component = 0) const;

    virtual double derivative_wrt_pl(const double pl, const unsigned int component = 0) const;
};

template <int dim>
double rho_l<dim>::value(const double pl,
                         const unsigned int /*component*/) const
{
	return 1000.0;
}

template <int dim>
double rho_l<dim>::derivative_wrt_pl(const double pl,
                         	 	 	 const unsigned int /*component*/) const
{
	return 0.0;
}

template <int dim>
class rho_v : public Function<dim>
{
public:
	rho_v()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, const double Sa, const double Sv,
                         const unsigned int component = 0) const;

    virtual double derivative_wrt_pl(const double pl, const double Sa, const double Sv, const unsigned int component = 0) const;
    virtual double derivative_wrt_Sv(const double pl, const double Sa, const double Sv, const unsigned int component = 0) const;
};

template <int dim>
double rho_v<dim>::value(const double pl, const double Sa, const double Sv,
                         const unsigned int /*component*/) const
{
	VaporPressure<dim> pv;

//	return 1.0 + 0.01*pv.value(pl, Sa, Sv);
	return 1.0;
}

template <int dim>
double rho_v<dim>::derivative_wrt_pl(const double pl, const double Sa, const double Sv,
                         const unsigned int /*component*/) const
{
	return 0.0;
//	return 0.01;
}

template <int dim>
double rho_v<dim>::derivative_wrt_Sv(const double pl, const double Sa, const double Sv,
                         const unsigned int /*component*/) const
{
	CapillaryPressurePcv<dim> pcv;

//	return 0.01*pcv.derivative_wrt_Sv(Sv);
	return 0.0;
}

template <int dim>
class rho_a : public Function<dim>
{
public:
	rho_a()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, const double Sa = 0.0, const double Sv = 0.0,
                         const unsigned int component = 0) const;

    virtual double derivative_wrt_pl(const double pl, const unsigned int component = 0) const;
};

template <int dim>
double rho_a<dim>::value(const double pl, const double Sa, const double Sv,
                         const unsigned int /*component*/) const
{
	return 1000.0;
}

template <int dim>
double rho_a<dim>::derivative_wrt_pl(const double pl,
                         const unsigned int /*component*/) const
{
	return 0.0;
}

template <int dim>
class Kappa_l : public Function<dim>
{
public:
	Kappa_l()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, const double Sa, const double Sv,
                         const unsigned int component = 0) const;
						 
};

template <int dim>
double Kappa_l<dim>::value(const double pl, const double Sa, const double Sv,
                                 const unsigned int /*component*/) const
{
	double Sl;

	Sl = ComputeSl<dim>(pl, Sa, Sv);


	return Sl*Sl*(1.0 - pow(Sa,5/3));
}

template <int dim>
class Kappa_v : public Function<dim>
{
public:
	Kappa_v()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, const double Sa, const double Sv,
                         const unsigned int component = 0) const;
};

template <int dim>
double Kappa_v<dim>::value(const double pl, const double Sa, const double Sv,
                                 const unsigned int /*component*/) const
{
	return Sv*Sv;
}

template <int dim>
class Kappa_a : public Function<dim>
{
public:
	Kappa_a()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, const double Sa, const double Sv,
                         const unsigned int component = 0) const;
};

template <int dim>
double Kappa_a<dim>::value(const double pl, const double Sa, const double Sv,
                                 const unsigned int /*component*/) const
{
//	return Sa*Sa;
	return pow(Sa,11/3);
}

// Viscosities
template <int dim>
class viscosity_l : public Function<dim>
{
public:
	viscosity_l()
        : Function<dim>(1)
    {}

    virtual double value(const double pl,
                         const unsigned int component = 0) const;
};

template <int dim>
double viscosity_l<dim>::value(const double pl,
                               const unsigned int /*component*/) const
{
	return 2.e-3;
}

template <int dim>
class viscosity_v : public Function<dim>
{
public:
	viscosity_v()
        : Function<dim>(1)
    {}

    virtual double value(const double pl,
                         const unsigned int component = 0) const;
};

template <int dim>
double viscosity_v<dim>::value(const double pl,
                               const unsigned int /*component*/) const
{
	return 0.25;
}

template <int dim>
class viscosity_a : public Function<dim>
{
public:
	viscosity_a()
        : Function<dim>(1)
    {}

    virtual double value(const double pl,
                         const unsigned int component = 0) const;
};

template <int dim>
double viscosity_a<dim>::value(const double pl,
                               const unsigned int /*component*/) const
{
	return 5.e-4;
}

// Mobilities
template <int dim>
class lambda_l : public Function<dim>
{
public:
	lambda_l()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, double Sa, double Sv,
                         const unsigned int component = 0) const;
						 
};

template <int dim>
double lambda_l<dim>::value(const double pl, double Sa, double Sv,
                                 const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

	Kappa_l<dim> exact_k_l;
//	exact_k_l.set_time(this->get_time());

	viscosity_l<dim> exact_mu_l;
//	exact_mu_l.set_time(this->get_time());

	return exact_k_l.value(pl, Sa, Sv)/exact_mu_l.value(pl);
}

template <int dim>
class lambda_v : public Function<dim>
{
public:
	lambda_v()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, double Sa, double Sv,
                         const unsigned int component = 0) const;
};

template <int dim>
double lambda_v<dim>::value(const double pl, double Sa, double Sv,
                            const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

	Kappa_v<dim> exact_k_v;

	viscosity_v<dim> exact_mu_v;

	return exact_k_v.value(pl, Sa, Sv)/exact_mu_v.value(pl);

}

template <int dim>
class lambda_a : public Function<dim>
{
public:
	lambda_a()
        : Function<dim>(1)
    {}

    virtual double value(const double pl, double Sa, double Sv,
                         const unsigned int component = 0) const;

};

template <int dim>
double lambda_a<dim>::value(const double pl, double Sa, double Sv,
                                 const unsigned int /*component*/) const
{
	Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));

	Kappa_a<dim> exact_k_a;

	viscosity_a<dim> exact_mu_a;

	return exact_k_a.value(pl, Sa, Sv)/exact_mu_a.value(pl);
}

template <int dim>
class GravitySourceTerm : public Function<dim>
{
public:
	GravitySourceTerm()
        : Function<dim>(1)
    {}

    virtual Tensor<1,dim> vector_value(const Point<dim> & p,
    								const unsigned int component = 0) const;
};

template <int dim>
Tensor<1,dim>
GravitySourceTerm<dim>::vector_value(const Point<dim> & p,
						const unsigned int /*component*/) const
{
	Tensor<1,dim> g;
	g = 0.0;

//	g[1] = -0.1;

	return g;
}

template <int dim>
class RightHandSideLiquidPressure : public Function<dim>
{
public:
	RightHandSideLiquidPressure()
        : Function<dim>(1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSideLiquidPressure<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
{
    double value = 0.0;

//    if(95.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 100.0 && 95.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 100.0)
//    	value = 0.0;//-0.5;///1000.0;
//    else if(0.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 5.0 && 0.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 5.0)
//		value = 0.0;//0.5;///1000.0;
//
//    return value;

//    if(90.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 95.0 && 90.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 95.0)
//    	value = 0.0;//-0.5;///1000.0;
//    else if(5.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 10.0 && 5.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 10.0)
//		value = 0.5;///1000.0;

    return value;
}

template <int dim>
class RightHandSideAqueousSaturation : public Function<dim>
{
public:
	RightHandSideAqueousSaturation()
        : Function<dim>(1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSideAqueousSaturation<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
{
    double value = 0.0;

//    if(0.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 5.0 && 0.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 5.0)
//    	value = 0.0;//0.5;///1000.0;
//    if(5.0 < p[0] + 1.e-12 && p[0] - 1.e-12 < 10.0 && 5.0 < p[1] + 1.e-12 && p[1] - 1.e-12 < 10.0)
//		value = 0.5;///1000.0;

    return value;
}

template <int dim>
class RightHandSideVaporSaturation : public Function<dim>
{
public:
	RightHandSideVaporSaturation()
        : Function<dim>(1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSideVaporSaturation<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
{

    return 0.0;
}

template <int dim>
class NeumannTermLiquidPressure : public Function<dim>
{
public:
	NeumannTermLiquidPressure()
        : Function<dim>(1)
    {}

    virtual Tensor<1,dim> vector_value(const Point<dim> & p,
    								const unsigned int component = 0) const;
};

template <int dim>
Tensor<1,dim>
NeumannTermLiquidPressure<dim>::vector_value(const Point<dim> & p,
						const unsigned int /*component*/) const
{
    Tensor<1,dim> result;

    result = 0.0;

    return result;
}

template <int dim>
class NeumannTermAqueousSaturation : public Function<dim>
{
public:
	NeumannTermAqueousSaturation()
        : Function<dim>(1)
    {}

    virtual Tensor<1,dim> vector_value(const Point<dim> & p,
    								const unsigned int component = 0) const;
};

template <int dim>
Tensor<1,dim>
NeumannTermAqueousSaturation<dim>::vector_value(const Point<dim> & p,
						const unsigned int /*component*/) const
{
    Tensor<1,dim> result;

    result = 0.0;

    return result;
}

template <int dim>
class NeumannTermVaporSaturation : public Function<dim>
{
public:
	NeumannTermVaporSaturation()
        : Function<dim>(1)
    {}

    virtual Tensor<1,dim> vector_value(const Point<dim> & p,
    								const unsigned int component = 0) const;
};

template <int dim>
Tensor<1,dim>
NeumannTermVaporSaturation<dim>::vector_value(const Point<dim> & p,
						const unsigned int /*component*/) const
{
    Tensor<1,dim> result;

    result = 0.0;

    return result;
}

template <int dim>
class BoundaryValuesLiquidPressure : public Function<dim>
{
public:
    BoundaryValuesLiquidPressure() = default;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double> &          values,
                            const unsigned int component = 0) const override;
};

template <int dim>
void BoundaryValuesLiquidPressure<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double> &          values,
                                     const unsigned int component) const
{
    (void)component;
    AssertIndexRange(component, 1);
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));

    ExactLiquidPressure<dim> exact_pressure;
	exact_pressure.set_time(this->get_time());

    for (unsigned int i = 0; i < values.size(); ++i)
    {
    	values[i] = exact_pressure.value(points[i]);
    }
}

template <int dim>
class BoundaryValuesAqueousSaturation : public Function<dim>
{
public:
	BoundaryValuesAqueousSaturation() = default;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double> &          values,
                            const unsigned int component = 0) const override;
};

template <int dim>
void BoundaryValuesAqueousSaturation<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double> &          values,
                                     const unsigned int component) const
{
    (void)component;
    AssertIndexRange(component, 1);
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));

    ExactAqueousSaturation<dim> exact_sat;
	exact_sat.set_time(this->get_time());

    for (unsigned int i = 0; i < values.size(); ++i)
    {
    	values[i] = exact_sat.value(points[i]);
    }
}

template <int dim>
class BoundaryValuesVaporSaturation : public Function<dim>
{
public:
	BoundaryValuesVaporSaturation() = default;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double> &          values,
                            const unsigned int component = 0) const override;
};

template <int dim>
void BoundaryValuesVaporSaturation<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double> &          values,
                                     const unsigned int component) const
{
    (void)component;
    AssertIndexRange(component, 1);
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));

    ExactVaporSaturation<dim> exact_vapor_sat;
    exact_vapor_sat.set_time(this->get_time());

    for (unsigned int i = 0; i < values.size(); ++i)
    {
    	values[i] = exact_vapor_sat.value(points[i]);
    }
}




#endif // AUX_PRIMARY_HH
