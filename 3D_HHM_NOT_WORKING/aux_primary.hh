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
// ------------------------------- 3D HHM -------------------------
extern double M;
extern double amp_factor_cap_pressure;

// stability terms
extern double stab_sa_data;
extern double stab_sv_data;



// Mesh creator
template <int dim>
void create_mesh(Triangulation<dim, dim> &triangulation, unsigned int ref_level,
                 std::vector<unsigned int> &dirichlet_id_pl,
                 std::vector<unsigned int> &dirichlet_id_sa,
                 std::vector<unsigned int> &dirichlet_id_sv)
{

    Point<dim> v0;
    Point<dim> v1;

    // bottom left corner of the domain
    v0[0] = 0.0;
    v0[1] = 0.0;

    if(dim == 3)
        v0[2] = 0.0;

    // top right corner of the domain
    v1[0] = 100.0;
    v1[1] = 10.0;

    if(dim == 3)
        v1[2] = 10.0;

    std::vector<unsigned int> repetitions(dim);

    repetitions[0] = 100;
    repetitions[1] = 10;

    if(dim == 3)
        repetitions[2] = 10;

    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, v0, v1);

//	GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(ref_level);

    typename Triangulation<dim>::active_cell_iterator
            cell = triangulation.begin_active(),
            endc = triangulation.end();

    for (; cell != endc; cell++)
    {

        for (unsigned int face_no=0; face_no < GeometryInfo<dim>::faces_per_cell; face_no++)
        {
            if(cell->face(face_no)->at_boundary())
            {
                bool at_x_left = false; // at x = 0

                for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
                {
                    Point<dim> &v = cell->face(face_no)->vertex(i);

                    if(fabs(v[0] - v0[0]) < 1.e-12)
                        at_x_left = true;
                    else
                    {
                        at_x_left = false;
                        break;
                    }

                }

                bool at_x_right = false;

                for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
                {
                    Point<dim> &v = cell->face(face_no)->vertex(i);

                    if(fabs(v[0] - v1[0]) < 1.e-12)
                        at_x_right = true;
                    else
                    {
                        at_x_right = false;
                        break;
                    }

                }

                bool at_y_bottom = false;

                for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
                {
                    Point<dim> &v = cell->face(face_no)->vertex(i);

                    if(fabs(v[1] - v0[1]) < 1.e-12)
                        at_y_bottom = true;
                    else
                    {
                        at_y_bottom = false;
                        break;
                    }

                }

                bool at_y_top = false;

                for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
                {
                    Point<dim> &v = cell->face(face_no)->vertex(i);

                    if(fabs(v[1] - v1[1]) < 1.e-12)
                        at_y_top = true;
                    else
                    {
                        at_y_top = false;
                        break;
                    }

                }

                bool at_z_left = false;
                bool at_z_right = false;

                if(dim == 3)
                {
                    for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
                    {
                        Point<dim> &v = cell->face(face_no)->vertex(i);

                        if(fabs(v[2] - v0[2]) < 1.e-12)
                            at_z_left = true;
                        else
                        {
                            at_z_left = false;
                            break;
                        }
                    }
                    for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
                    {
                        Point<dim> &v = cell->face(face_no)->vertex(i);

                        if(fabs(v[2] - v1[2]) < 1.e-12)
                            at_z_right = true;
                        else
                        {
                            at_z_right = false;
                            break;
                        }

                    }
                }


                if(at_x_left) // x = 0 (left part of domain)
                    cell->face(face_no)->set_boundary_id(3);
                if(at_x_right) // x = 100 (right part of domain)
                    cell->face(face_no)->set_boundary_id(1);
                if(at_y_bottom || at_y_top) // y = 0 or y = 10 (bottom and top parts of domain)
                    cell->face(face_no)->set_boundary_id(0);

                if(dim == 3)
                    if(at_z_left || at_z_right)
                        cell->face(face_no)->set_boundary_id(0);
            }
        }
    }

    // Dirichlet boundary attributes for each problem
    dirichlet_id_pl.resize(2);
    dirichlet_id_pl[0] = 1;
    dirichlet_id_pl[1] = 3;

    dirichlet_id_sa.resize(1);
    dirichlet_id_sa[0] = 3;

    dirichlet_id_sv.resize(1);
    dirichlet_id_sv[0] = 3;
}

// This function create and prints the initial perturbation to a file. It can be empty if not used
// NOTE: It only works when the code is run with only one processor.
// TODO: figure out how to print the file in parallel
template <int dim>
void create_initial_Sa_vector(Triangulation<dim, dim> &triangulation, MPI_Comm mpi_communicator,
                              const unsigned int n_mpi_processes, const unsigned int this_mpi_process)
{
    
}

template <int dim>
double compute_initial_sa_value(const typename DoFHandler<dim>::active_cell_iterator &cell)
{
	double sa0 = 0.2;

	double coord_x = cell->center()[0];

	double sa_rand = 1.0;

	if(coord_x <= 4.0)
	{
		double mean_val = 0.0;
		double sigma = 0.1;
		double random_number = -1.0;

		while(random_number < -0.1 || random_number > 0.1)
			random_number = Utilities::generate_normal_random_number(mean_val,sigma);

		sa0 += random_number;
	}

	return sa0;
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


// This function creates the value of kappa according to the cell
template <int dim>
double compute_kappa_value(const typename DoFHandler<dim>::active_cell_iterator &cell)
{
    double kappa_abs = 3.12e-14;
	double kappa = kappa_abs; // tem
	double coord;

	coord = cell -> center()[dim -1];

	double kappa_rand = 1.0;
	if (coord <= 2.0)
	{
		
		double mean_val = 5.0;
		double sigma = mean_val/3.0;
		double random_number = -1.0;
		while (random_number < 1.0 || random_number > 10.0)
			random_number = Utilities::generate_normal_random_number(mean_val,sigma);
		
		kappa_rand = random_number;
	}
	else if(coord > 2.0 && coord <= 8.0)
	{
		double mean_val = 55.0;
		double sigma = mean_val/3.0;
		double random_number = -1.0;
		while(random_number < 10.0 || random_number > 100.0)
			random_number = Utilities::generate_normal_random_number(mean_val,sigma);
		kappa_rand = random_number;
	}
	else
	{
		double mean_val = 550.0;
		double sigma = mean_val/3.0;
		double random_number = -1.0;

		while(random_number < 100.0 || random_number > 1000.0)
			random_number = Utilities::generate_normal_random_number(mean_val,sigma);
		kappa_rand = random_number;
	}
	kappa *=kappa_rand;

    return kappa;
};

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
    // return 2.e7 + (3.e4 - 2.e5)*p[0];
    return 2.e8 - 1.e5*p[0];
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
    return 0.1;
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
    if(p[0] <= 3.0)
        {
            double mean_val = 0.0;
		    double sigma = 0.1;//mean_val/3.0;
		    double random_number = -1.0;

		    while(random_number < -0.1 || random_number > 0.1)
			    random_number = Utilities::generate_normal_random_number(mean_val,sigma);
		    return 0.2 + random_number;
        }
    return 0.2;
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
    Sa = std::min(1.0, std::max(Sa, 0.0));
	Sv = std::min(1.0, std::max(Sv, 0.0));
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

    virtual double value(const double Sv,
                         const unsigned int component = 0) const;

    virtual Tensor<1,dim> num_gradient(const double Sv,
                                       const Tensor<1,dim> grad_Sv,
                                       const unsigned int component = 0) const;

    virtual double derivative_wrt_Sv(const double Sv,
                                     const unsigned int component = 0) const;
};

template <int dim>
double
CapillaryPressurePcv<dim>::value(double Sv,
                                 const unsigned int /*component*/) const
{
	Sv = std::min(1.0, std::max(Sv, 0.0));

	if(Sv > 0.05)
		return -amp_factor_cap_pressure/sqrt(Sv);
	else
		return amp_factor_cap_pressure*(-1.5 + 10.0*Sv)/sqrt(0.05);   
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

	if(Sv > 0.05)
		return amp_factor_cap_pressure*0.5*pow(Sv, -1.5);
	else
		return amp_factor_cap_pressure*10.0/sqrt(0.05);
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

	if(Sa > 0.05)
		return amp_factor_cap_pressure/sqrt(Sa);
	else
		return amp_factor_cap_pressure*(1.5 - 10.0*Sa)/sqrt(0.05);
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

	if(Sa > 0.05)
		return -amp_factor_cap_pressure*0.5*pow(Sa, -1.5);
	else
		return -amp_factor_cap_pressure*10.0/sqrt(0.05);
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
    return 800.0;
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
    return 400.0;
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


    return Sl*Sl;
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
    // return 0.05*Sv*Sv;
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
    return Sa*Sa;
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
    // return M*1.e-3;
    return 1.e-3;
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
    return 1.e-4;
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
    return 1.e-3;
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
    return 0.0;
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
    return 0.0;
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
        // values[i] = exact_pressure.value(points[i]);
        values[i] = 2.e8 - 1.e5*points[i](0);
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
    double time = this->get_time();

    for (unsigned int i = 0; i < values.size(); ++i)
    {
        values[i] = 0.2 + 0.1*pow(time,6)/(1.e15 + pow(time,6));
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
        // values[i] = exact_vapor_sat.value(points[i]);
        values[i] = 0.1 + 0.6*pow(time,6)/(1.e15 + pow(time,6));

    }
}
#endif // AUX_PRIMARY_HH