#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
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


bool inc = false; // MUST MATCH PARAMETER FILE.

double porosity_data = 0.2;

double kappa = 1.0; // kappa absolute permeability

double rho_l_data = 3.0;
double rho_a_data = 5.0;
double rho_v_data = 1.0;

double mu_l_data = 0.75; // default 0.75
double mu_a_data = 1.0;  // default 1.0
double mu_v_data = 0.25; // default 0.25


double stab_sa_data = 5.0;
double stab_sv_data = 5.0;

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

    // top right corner of the domain
    v1[0] = 1.0;
    v1[1] = 1.0;

    std::vector<unsigned int> repetitions(dim);

    repetitions[0] = 1;
    repetitions[1] = 1;

    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, v0, v1);

    //GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(ref_level);

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
                bool at_x_equals_1 = false;

                // for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
                // {
                // Point<dim> &v = cell->face(face_no)->vertex(i);

                // if(fabs(v[0] - 1.0) < 1.e-12)
                // at_x_equals_1 = true;
                // else
                // {
                // at_x_equals_1 = false;
                // break;
                // }

                // }

//				if(at_x_equals_1)
//					cell->face(face_no)->set_boundary_id(1);

//				pcout << "boundary id = " << cell->face(face_no)->boundary_id() << std::endl;
            }
        }
    }
    // Dirichlet boundary attributes for each problem
    dirichlet_id_pl.resize(1);
    dirichlet_id_pl[0] = 0;

    dirichlet_id_sa.resize(1);
    dirichlet_id_sa[0] = 0;

    dirichlet_id_sv.resize(1);
    dirichlet_id_sv[0] = 0;
}

// This function create and prints the initial perturbation to a file. It can be empty if not used
// NOTE: It only works when the code is run with only one processor.
// TODO: figure out how to print the file in parallel
template <int dim>
void create_initial_Sa_vector(Triangulation<dim, dim> &triangulation, MPI_Comm mpi_communicator,
                              const unsigned int n_mpi_processes, const unsigned int this_mpi_process)
{

}

// This function creates the value of kappa according to the cell
template <int dim>
double compute_kappa_value(const typename DoFHandler<dim>::active_cell_iterator &cell)
{
    return kappa;
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

    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                                   const unsigned int component = 0) const override;

    virtual double laplacian(const Point<dim> & p,
                             const unsigned int component = 0) const override;

    virtual double time_derivative(const Point<dim> & p,
                                   const unsigned int component = 0) const;
};

template <int dim>
double
ExactLiquidPressure<dim>::value(const Point<dim> &p,
                                const unsigned int /*component*/) const
{
    return 2.0 + p[0]*p[1]*p[1] + p[0]*p[0]*sin(this->get_time() + p[1]);
}

template <int dim>
Tensor<1,dim>
ExactLiquidPressure<dim>::gradient(const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
    Tensor<1,dim> grad_pl;

    grad_pl[0] = p[1]*p[1] + 2.0*p[0]*sin(this->get_time() + p[1]);
    grad_pl[1] = 2.0*p[0]*p[1] + p[0]*p[0]*cos(this->get_time() + p[1]);

    return grad_pl;
}

template <int dim>
double
ExactLiquidPressure<dim>::laplacian(const Point<dim> &p,
                                    const unsigned int /*component*/) const
{
    return 2.0*p[0] + sin(this->get_time() + p[1])*(2.0 - p[0]*p[0]);
}

template <int dim>
double
ExactLiquidPressure<dim>::time_derivative(const Point<dim> &p,
                                          const unsigned int /*component*/) const
{
    return p[0]*p[0]*cos(this->get_time() + p[1]);
}

template <int dim>
class ExactLiquidSaturation : public Function<dim>
{
public:
    ExactLiquidSaturation()
            : Function<dim>(1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                                   const unsigned int component = 0) const override;

    virtual double time_derivative(const Point<dim> & p,
                                   const unsigned int component = 0) const;
};

template <int dim>
double
ExactLiquidSaturation<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
    return (2.0 - p[0]*p[0]*p[1]*p[1])/4.0;
}

template <int dim>
Tensor<1,dim>
ExactLiquidSaturation<dim>::gradient(const Point<dim> &p,
                                     const unsigned int /*component*/) const
{

    Tensor<1,dim> grad_Sl;
//	Vector<double> grad_Sl(dim);

    grad_Sl[0] = -p[0]*p[1]*p[1]/2.0;
    grad_Sl[1] = -p[0]*p[0]*p[1]/2.0;

    return grad_Sl;
}

template <int dim>
double
ExactLiquidSaturation<dim>::time_derivative(const Point<dim> &p,
                                            const unsigned int /*component*/) const
{
    return 0.0;
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

    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                                   const unsigned int component = 0) const override;

    virtual double laplacian(const Point<dim> & p,
                             const unsigned int component = 0) const override;

    virtual double time_derivative(const Point<dim> & p,
                                   const unsigned int component = 0) const;
};

template <int dim>
double
ExactVaporSaturation<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
    return (3.0 - cos(this->get_time() + p[0]))/8.0;
}

template <int dim>
Tensor<1,dim>
ExactVaporSaturation<dim>::gradient(const Point<dim> &p,
                                    const unsigned int /*component*/) const
{
    Tensor<1,dim> grad_Sv;

    grad_Sv[0] = sin(this->get_time() + p[0])/8.0;
    grad_Sv[1] = 0.0;

    return grad_Sv;
}

template <int dim>
double
ExactVaporSaturation<dim>::laplacian(const Point<dim> &p,
                                     const unsigned int /*component*/) const
{
    return cos(this->get_time() + p[0])/8.0;
}

template <int dim>
double
ExactVaporSaturation<dim>::time_derivative(const Point<dim> &p,
                                           const unsigned int /*component*/) const
{
    return sin(this->get_time() + p[0])/8.0;
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

    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                                   const unsigned int component = 0) const override;

    virtual double laplacian(const Point<dim> & p,
                             const unsigned int component = 0) const override;

    virtual double time_derivative(const Point<dim> & p,
                                   const unsigned int component = 0) const;
};

template <int dim>
double
ExactAqueousSaturation<dim>::value(const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
    return (1.0 + 2.0*p[0]*p[0]*p[1]*p[1] + cos(this->get_time() + p[0]))/8.0;
}

template <int dim>
Tensor<1,dim>
ExactAqueousSaturation<dim>::gradient(const Point<dim> &p,
                                      const unsigned int /*component*/) const
{
    Tensor<1,dim> grad_Sa;

    grad_Sa[0] = (4.0*p[0]*p[1]*p[1] - sin(this->get_time() + p[0]))/8.0;
    grad_Sa[1] = p[0]*p[0]*p[1]/2.0;

    return grad_Sa;
}

template <int dim>
double
ExactAqueousSaturation<dim>::laplacian(const Point<dim> &p,
                                       const unsigned int /*component*/) const
{
    return (4.0*p[1]*p[1] - cos(this->get_time() + p[0]))/8.0 + p[0]*p[0]/2.0;;
}

template <int dim>
double
ExactAqueousSaturation<dim>::time_derivative(const Point<dim> &p,
                                             const unsigned int /*component*/) const
{
    return -sin(this->get_time() + p[0])/8.0;
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
        exact_pressure.set_time(this->get_time());

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
        exact_vapor_sat.set_time(this->get_time());

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
        exact_aqueous_sat.set_time(this->get_time());

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

    virtual double value(const double Sv,
                         const unsigned int component = 0) const;

    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                                   const unsigned int component = 0) const override;

    virtual Tensor<1,dim> num_gradient(const double Sv,
                                       const Tensor<1,dim> grad_Sv,
                                       const unsigned int component = 0) const;

    virtual double laplacian(const Point<dim> & p,
                             const unsigned int component = 0) const override;

    virtual double derivative_wrt_Sv(const double Sv,
                                     const unsigned int component = 0) const;

    virtual double second_derivative_wrt_Sv(const double Sv,
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
CapillaryPressurePcv<dim>::gradient(const Point<dim> &p,
                                    const unsigned int /*component*/) const
{
    ExactVaporSaturation<dim> exact_Sv;
    exact_Sv.set_time(this->get_time());

    Tensor<1,dim> grad_Sv;
    double Sv;

    grad_Sv = exact_Sv.gradient(p);
    Sv = exact_Sv.value(p);

    double dpcv_dSv = derivative_wrt_Sv(Sv);

    Tensor<1,dim> grad_Pcv;

    grad_Pcv = grad_Sv;
    grad_Pcv *= dpcv_dSv;

    return grad_Pcv;
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
CapillaryPressurePcv<dim>::laplacian(const Point<dim> &p,
                                     const unsigned int /*component*/) const
{
    ExactVaporSaturation<dim> exact_Sv;
    exact_Sv.set_time(this->get_time());

    double Sv;
    Tensor<1,dim> grad_Sv;
    double lap_Sv;

    Sv = exact_Sv.value(p);
    grad_Sv = exact_Sv.gradient(p);
    lap_Sv = exact_Sv.laplacian(p);

    return -(3.9/log(0.01))*(1.0/((1.0 - Sv + 0.01)*(1.0 - Sv + 0.01))
                             *(grad_Sv[0]*grad_Sv[0] + grad_Sv[1]*grad_Sv[1]) + 1.0/(1.0 - Sv + 0.01)*lap_Sv);
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
double
CapillaryPressurePcv<dim>::second_derivative_wrt_Sv(double Sv,
                                                    const unsigned int /*component*/) const
{
    Sv = std::min(1.0, std::max(Sv, 0.0));

    return -(3.9/log(0.01))*(1.0/((1.0 - Sv + 0.01)*(1.0 - Sv + 0.01)));
//	return -(10.3/log(0.01))*(1.0/((1.0 - Sv + 0.01)*(1.0 - Sv + 0.01)));
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

    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                                   const unsigned int component = 0) const override;

    virtual Tensor<1,dim> num_gradient(double Sa, double Sv,
                                       const Tensor<1,dim> grad_Sa,
                                       const Tensor<1,dim> grad_Sv,
                                       const unsigned int component = 0) const;

    virtual double laplacian(const Point<dim> & p,
                             const unsigned int component = 0) const override;

    virtual double derivative_wrt_Sa(double Sa, double Sv,
                                     const unsigned int component = 0) const;

    virtual double derivative_wrt_Sv(double Sa, double Sv,
                                     const unsigned int component = 0) const;

    virtual double second_derivative_wrt_Sa(double Sa, double Sv,
                                            const unsigned int component = 0) const;
};

template <int dim>
double
CapillaryPressurePca<dim>::value(double Sa, double Sv,
                                 const unsigned int /*component*/) const
{
    Sa = std::min(1.0, std::max(Sa, 0.0));
    Sv = std::min(1.0, std::max(Sv, 0.0));

    return (6.3/log(0.01))*log(Sa + 0.01);
}

template <int dim>
Tensor<1,dim>
CapillaryPressurePca<dim>::gradient(const Point<dim> &p,
                                    const unsigned int /*component*/) const
{
    ExactAqueousSaturation<dim> exact_Sa;
    exact_Sa.set_time(this->get_time());

    Tensor<1,dim> grad_Sa;
    double Sa;

    grad_Sa = exact_Sa.gradient(p);
    Sa = exact_Sa.value(p);

    double dpca_dSa = (6.3/log(0.01))*(1.0/(Sa + 0.01));

    Tensor<1,dim> grad_Pca;

    grad_Pca = grad_Sa;
    grad_Pca *= dpca_dSa;

    return grad_Pca;
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
CapillaryPressurePca<dim>::laplacian(const Point<dim> &p,
                                     const unsigned int /*component*/) const
{
    ExactAqueousSaturation<dim> exact_Sa;
    exact_Sa.set_time(this->get_time());

    double Sa;
    Tensor<1,dim> grad_Sa;
    double lap_Sa;

    Sa = exact_Sa.value(p);
    grad_Sa = exact_Sa.gradient(p);
    lap_Sa = exact_Sa.laplacian(p);

    return (6.3/log(0.01))*(-1.0/((Sa + 0.01)*(Sa + 0.01))
                            *(grad_Sa[0]*grad_Sa[0] + grad_Sa[1]*grad_Sa[1]) + 1.0/(Sa + 0.01)*lap_Sa);
}

template <int dim>
double
CapillaryPressurePca<dim>::derivative_wrt_Sa(double Sa, double Sv,
                                             const unsigned int /*component*/) const
{
    Sa = std::min(1.0, std::max(Sa, 0.0));
    Sv = std::min(1.0, std::max(Sv, 0.0));

    return (6.3/log(0.01))*(1.0/(Sa + 0.01));
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
double
CapillaryPressurePca<dim>::second_derivative_wrt_Sa(double Sa, double Sv,
                                                    const unsigned int /*component*/) const
{
    Sa = std::min(1.0, std::max(Sa, 0.0));
    Sv = std::min(1.0, std::max(Sv, 0.0));

    return (-6.3/log(0.01))*(1.0/((Sa + 0.01)*(Sa + 0.01)));
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
    return porosity_data;
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
    return rho_l_data;
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
    return rho_v_data;
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
    return rho_a_data;
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

    virtual double derivative_wrt_Sa(const double pl, const double Sa, const double Sv,
                                     const unsigned int component = 0) const;

    virtual double derivative_wrt_Sv(const double pl, const double Sa, const double Sv,
                                     const unsigned int component = 0) const;
};

template <int dim>
double Kappa_l<dim>::value(const double pl, const double Sa, const double Sv,
                           const unsigned int /*component*/) const
{
    double Sl;

    Sl = ComputeSl<dim>(pl, Sa, Sv);


    return Sl*(Sl + Sa)*(1.0 - Sa);
}

template <int dim>
double Kappa_l<dim>::derivative_wrt_Sa(const double pl, const double Sa, const double Sv,
                                       const unsigned int /*component*/) const
{
    return (1.0 - Sv)*(-2.0 + 2.0*Sa + Sv);
}

template <int dim>
double Kappa_l<dim>::derivative_wrt_Sv(const double pl, const double Sa, const double Sv,
                                       const unsigned int /*component*/) const
{
    return (1.0 - Sa)*(-2.0 + 2.0*Sv + Sa);
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

    virtual double derivative_wrt_Sv(const double pl, const double Sa, const double Sv,
                                     const unsigned int component = 0) const;
};

template <int dim>
double Kappa_v<dim>::value(const double pl, const double Sa, const double Sv,
                           const unsigned int /*component*/) const
{
    return Sv*Sv;
}

template <int dim>
double Kappa_v<dim>::derivative_wrt_Sv(const double pl, const double Sa, const double Sv,
                                       const unsigned int /*component*/) const
{
    return 2.0*Sv;
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

    virtual double derivative_wrt_Sa(const double pl, const double Sa, const double Sv,
                                     const unsigned int component = 0) const;
};

template <int dim>
double Kappa_a<dim>::value(const double pl, const double Sa, const double Sv,
                           const unsigned int /*component*/) const
{
    return Sa*Sa;
}

template <int dim>
double Kappa_a<dim>::derivative_wrt_Sa(const double pl, const double Sa, const double Sv,
                                       const unsigned int /*component*/) const
{
    return 2.0*Sa;
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
    return mu_l_data;
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
    return mu_v_data;
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
    //return 0.5; 
    return mu_a_data; // Multiplied by 2 to divide diffusion term by 2
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

    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                                   const unsigned int component = 0) const override;
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
Tensor<1,dim>
lambda_l<dim>::gradient(const Point<dim> &p,
                        const unsigned int /*component*/) const
{
    ExactVaporSaturation<dim> exact_Sv;
    exact_Sv.set_time(this->get_time());

    ExactAqueousSaturation<dim> exact_Sa;
    exact_Sa.set_time(this->get_time());

    Kappa_l<dim> exact_kappa_l;
    exact_kappa_l.set_time(this->get_time());

    viscosity_l<dim> exact_mu_l;

    ExactLiquidPressure<dim> exact_pl;
    exact_pl.set_time(this->get_time());

    double pl = exact_pl.value(p);

    Tensor<1,dim> grad_Sv;
    double Sv;

    Tensor<1,dim> grad_Sa;
    double Sa;

    grad_Sv = exact_Sv.gradient(p);
    Sv = exact_Sv.value(p);

    grad_Sa = exact_Sa.gradient(p);
    Sa = exact_Sa.value(p);

    double dkl_dSa = exact_kappa_l.derivative_wrt_Sa(pl, Sa, Sv);
    double dkl_dSv = exact_kappa_l.derivative_wrt_Sv(pl, Sa, Sv);

    Tensor<1,dim> grad_lambda_l;

    grad_lambda_l = grad_Sv;
    grad_lambda_l *= dkl_dSv;

    Tensor<1,dim> aux1;

    aux1 = grad_Sa;
    aux1 *= dkl_dSa;

    grad_lambda_l += aux1;

    grad_lambda_l /= exact_mu_l.value(pl);

    return grad_lambda_l;
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

    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                                   const unsigned int component = 0) const override;
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
Tensor<1,dim>
lambda_v<dim>::gradient(const Point<dim> &p,
                        const unsigned int /*component*/) const
{
    ExactAqueousSaturation<dim> exact_Sa;
    exact_Sa.set_time(this->get_time());

    ExactVaporSaturation<dim> exact_Sv;
    exact_Sv.set_time(this->get_time());

    Kappa_v<dim> exact_kappa_v;
    exact_kappa_v.set_time(this->get_time());

    viscosity_v<dim> exact_mu_v;

    ExactLiquidPressure<dim> exact_pl;

    exact_pl.set_time(this->get_time());
    double pl = exact_pl.value(p);

    double Sa;
    Sa = exact_Sa.value(p);

    Tensor<1,dim> grad_Sv;
    double Sv;

    grad_Sv = exact_Sv.gradient(p);
    Sv = exact_Sv.value(p);

    double dkv_dSv = exact_kappa_v.derivative_wrt_Sv(pl, Sa, Sv);

    Tensor<1,dim> grad_lambda_v;

    grad_lambda_v = grad_Sv;
    grad_lambda_v *= dkv_dSv;

    grad_lambda_v /= exact_mu_v.value(pl);

    return grad_lambda_v;
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

    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                                   const unsigned int component = 0) const override;

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
Tensor<1,dim>
lambda_a<dim>::gradient(const Point<dim> &p,
                        const unsigned int /*component*/) const
{
    ExactAqueousSaturation<dim> exact_Sa;
    exact_Sa.set_time(this->get_time());

    ExactVaporSaturation<dim> exact_Sv;
    exact_Sv.set_time(this->get_time());

    Kappa_a<dim> exact_kappa_a;
    exact_kappa_a.set_time(this->get_time());

    viscosity_a<dim> exact_mu_a;

    ExactLiquidPressure<dim> exact_pl;

    exact_pl.set_time(this->get_time());
    double pl = exact_pl.value(p);

    Tensor<1,dim> grad_Sa;
    double Sa;

    grad_Sa = exact_Sa.gradient(p);
    Sa = exact_Sa.value(p);

    double Sv;
    Sv = exact_Sv.value(p);

    double dka_dSa = exact_kappa_a.derivative_wrt_Sa(pl, Sa, Sv);

    Tensor<1,dim> grad_lambda_a;

    grad_lambda_a = grad_Sa;
    grad_lambda_a *= dka_dSa;

    grad_lambda_a /= exact_mu_a.value(pl);

    return grad_lambda_a;
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

	g[1] = -0.1;

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
    ExactLiquidPressure<dim> ex_liquid_pressure;
    ExactAqueousSaturation<dim> ex_aqueous_saturation;
    ExactVaporSaturation<dim> ex_vapor_saturation;
    ExactLiquidSaturation<dim> ex_liquid_saturation;

    ex_liquid_pressure.set_time(this->get_time());
    ex_aqueous_saturation.set_time(this->get_time());
    ex_vapor_saturation.set_time(this->get_time());
    ex_liquid_saturation.set_time(this->get_time());

    lambda_l<dim> ex_lambda_l;
    lambda_v<dim> ex_lambda_v;
    lambda_a<dim> ex_lambda_a;

    rho_l<dim> ex_rho_l;
    rho_v<dim> ex_rho_v;
    rho_a<dim> ex_rho_a;

    CapillaryPressurePcv<dim> ex_cap_p_pcv;
    CapillaryPressurePca<dim> ex_cap_p_pca;

    GravitySourceTerm<dim> gravity_fcn;

    gravity_fcn.set_time(this->get_time());

    double pl = ex_liquid_pressure.value(p);
    double Sa = ex_aqueous_saturation.value(p);
    double Sv = ex_vapor_saturation.value(p);

    ex_lambda_l.set_time(this->get_time());
    ex_lambda_v.set_time(this->get_time());
    ex_lambda_a.set_time(this->get_time());

    ex_cap_p_pcv.set_time(this->get_time());
    ex_cap_p_pca.set_time(this->get_time());

    double lambda_l = ex_lambda_l.value(pl, Sa, Sv);
    Tensor<1,dim> grad_lambda_l = ex_lambda_l.gradient(p);

    double lambda_v = ex_lambda_v.value(pl, Sa, Sv);
    Tensor<1,dim> grad_lambda_v = ex_lambda_v.gradient(p);

    double lambda_a = ex_lambda_a.value(pl, Sa, Sv);
    Tensor<1,dim> grad_lambda_a = ex_lambda_a.gradient(p);

    double rho_l_val = ex_rho_l.value(pl);
    double rho_v_val = ex_rho_v.value(pl, Sa, Sv);
    double rho_a_val = ex_rho_a.value(pl);
    double rho_l_val_sq = ex_rho_l.value(pl)*ex_rho_l.value(pl);
    double rho_v_val_sq = ex_rho_v.value(pl, Sa, Sv)*ex_rho_v.value(pl, Sa, Sv);
    double rho_a_val_sq = ex_rho_a.value(pl)*ex_rho_a.value(pl);

    if (inc){
        rho_l_val = 1.0;
        rho_v_val = 1.0;
        rho_a_val = 1.0;
        rho_l_val_sq = 1.0;
        rho_v_val_sq = 1.0;
        rho_a_val_sq = 1.0;
    }


    Tensor<1,dim> grad_pl = ex_liquid_pressure.gradient(p);
    double lap_pl = ex_liquid_pressure.laplacian(p);

    Tensor<1,dim> grad_Sv = ex_vapor_saturation.gradient(p);

    Tensor<1,dim> grad_pcv = ex_cap_p_pcv.gradient(p);
    double lap_pcv = ex_cap_p_pcv.laplacian(p);

    Tensor<1,dim> grad_pca = ex_cap_p_pca.gradient(p);
    double lap_pca = ex_cap_p_pca.laplacian(p);

    porosity<dim> ex_porosity;
    double phi_porosity = ex_porosity.value(pl);

//    double time_der_term = phi_porosity*sin(this->get_time() + p[0])/8.0*(rho_v_val - rho_a_val);

    double liquid_time_term = rho_l_val*ex_liquid_saturation.time_derivative(p);
    double aqueous_time_term = rho_a_val*ex_aqueous_saturation.time_derivative(p);
    double vapor_time_term = rho_v_val*ex_vapor_saturation.time_derivative(p)
                             + Sv*(ex_rho_v.derivative_wrt_pl(pl, Sa, Sv)*ex_liquid_pressure.time_derivative(p)
                                   + ex_rho_v.derivative_wrt_Sv(pl, Sa, Sv)*ex_vapor_saturation.time_derivative(p));
    double time_der_term = phi_porosity*(liquid_time_term + aqueous_time_term + vapor_time_term);
    if (inc){
        time_der_term = 0.0;
    }

//    double time_der_term = phi_porosity*(ex_aqueous_saturation.time_derivative(p)*(rho_a_val - rho_l_val)
//    		+ ex_vapor_saturation.time_derivative(p)*(rho_v_val - rho_l_val)
//			+ Sv*(ex_rho_v.derivative_wrt_pl(pl, Sa, Sv)*ex_liquid_pressure.time_derivative(p)
//					+ ex_rho_v.derivative_wrt_Sv(pl, Sa, Sv)*ex_vapor_saturation.time_derivative(p)));

    Tensor<1,dim> grad_rho_v = grad_pl;
    grad_rho_v *= ex_rho_v.derivative_wrt_pl(pl, Sa, Sv);

    Tensor<1,dim> aux_rho_v = grad_Sv;
    aux_rho_v *= ex_rho_v.derivative_wrt_Sv(pl, Sa, Sv);

    grad_rho_v += aux_rho_v;

    double pl_term = rho_l_val*grad_lambda_l*grad_pl;
    pl_term += rho_v_val*grad_lambda_v*grad_pl;
    pl_term += lambda_v*grad_rho_v*grad_pl;
    pl_term += rho_a_val*grad_lambda_a*grad_pl;
    pl_term += (rho_l_val*lambda_l + rho_v_val*lambda_v + rho_a_val*lambda_a)*lap_pl;
//    double pl_term = (grad_lambda_l + grad_lambda_v + grad_lambda_a)*grad_pl
//    		+ (lambda_l + lambda_v + lambda_a)*lap_pl;

    double pcv_term = grad_lambda_v*grad_pcv + lambda_v*lap_pcv;
    pcv_term *= rho_v_val;

    Tensor<1,dim> aux_pcv = grad_rho_v;
    aux_pcv *= lambda_v;

    pcv_term += aux_pcv*grad_pcv;

    double pca_term = grad_lambda_a*grad_pca + lambda_a*lap_pca;
    pca_term *= rho_a_val;

    Tensor<1,dim> g_val = gravity_fcn.vector_value(p);

    double gravity_term = grad_lambda_l*g_val;
   // gravity_term *= rho_l_val*rho_l_val;
    gravity_term *= rho_l_val_sq;

    double aux_g1 = grad_lambda_v*g_val;
    //aux_g1 *= rho_v_val*rho_v_val;
    aux_g1 *= rho_v_val_sq;

    double aux_g2 = grad_pl*g_val;
    aux_g2 *= 2.0*lambda_v*rho_v_val*ex_rho_v.derivative_wrt_pl(pl, Sa, Sv);

    double aux_g3 = grad_Sv*g_val;
    aux_g3 *= 2.0*lambda_v*rho_v_val*ex_rho_v.derivative_wrt_Sv(pl, Sa, Sv);

    double aux_g4 = grad_lambda_a*g_val;
   // aux_g4 *= rho_a_val*rho_a_val;
    aux_g4 *= rho_a_val_sq;

    gravity_term += aux_g1 + aux_g2 + aux_g3 + aux_g4;
    //double RHS_Liquid_pressure;
      return time_der_term - kappa*pl_term - kappa*pcv_term + kappa*pca_term + kappa*gravity_term;
//    return - kappa*pl_term - kappa*pcv_term + kappa*pca_term + kappa*gravity_term;
//    return - kappa*pl_term - kappa*pcv_term + kappa*pca_term;
//    return -pl_term;
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
    ExactLiquidPressure<dim> ex_liquid_pressure;
    ExactAqueousSaturation<dim> ex_aqueous_saturation;
    ExactVaporSaturation<dim> ex_vapor_saturation;

    rho_a<dim> density_a;

    ex_liquid_pressure.set_time(this->get_time());
    ex_aqueous_saturation.set_time(this->get_time());
    ex_vapor_saturation.set_time(this->get_time());

    lambda_l<dim> ex_lambda_l;
    lambda_v<dim> ex_lambda_v;
    lambda_a<dim> ex_lambda_a;

    ex_lambda_l.set_time(this->get_time());
    ex_lambda_v.set_time(this->get_time());
    ex_lambda_a.set_time(this->get_time());

    CapillaryPressurePcv<dim> ex_cap_p_pcv;
    CapillaryPressurePca<dim> ex_cap_p_pca;

    ex_cap_p_pcv.set_time(this->get_time());
    ex_cap_p_pca.set_time(this->get_time());

    GravitySourceTerm<dim> gravity_fcn;

    gravity_fcn.set_time(this->get_time());

    porosity<dim> ex_porosity;

    double pl = ex_liquid_pressure.value(p);
    double Sa = ex_aqueous_saturation.value(p);
    double Sv = ex_vapor_saturation.value(p);

    double rho_a_val = density_a.value(pl);
    double rho_a_val_sq = density_a.value(pl)*density_a.value(pl);
    if (inc){
        rho_a_val = 1.0;
        rho_a_val_sq = density_a.value(pl);
    }

    double lambda_l = ex_lambda_l.value(pl, Sa, Sv);
    Tensor<1,dim> grad_lambda_l = ex_lambda_l.gradient(p);

    double lambda_v = ex_lambda_v.value(pl, Sa, Sv);
    Tensor<1,dim> grad_lambda_v = ex_lambda_v.gradient(p);

    double lambda_a = ex_lambda_a.value(pl, Sa, Sv);
    Tensor<1,dim> grad_lambda_a = ex_lambda_a.gradient(p);

    Tensor<1,dim> grad_pl = ex_liquid_pressure.gradient(p);
    double lap_pl = ex_liquid_pressure.laplacian(p);

    Tensor<1,dim> grad_Sa = ex_aqueous_saturation.gradient(p);
    double lap_Sa = ex_aqueous_saturation.laplacian(p);

    double dpca_dSa = ex_cap_p_pca.derivative_wrt_Sa(Sa, Sv);
    double d2pca_dSa2 = ex_cap_p_pca.second_derivative_wrt_Sa(Sa, Sv);

    double phi_porosity = ex_porosity.value(pl);

//    double time_der_term = -rho_a_val*phi_porosity*sin(this->get_time() + p[0])/8.0;
    double time_der_term = rho_a_val*phi_porosity*ex_aqueous_saturation.time_derivative(p);

    double pca_Sa_term = grad_lambda_a*grad_Sa;
    pca_Sa_term *= dpca_dSa;

    double aux1 = grad_Sa*grad_Sa;
    aux1 *= lambda_a*d2pca_dSa2;

    pca_Sa_term += aux1;

    pca_Sa_term += lambda_a*dpca_dSa*lap_Sa;

    pca_Sa_term *= rho_a_val;

    double pca_Sv_term = 0.0;

    double pl_term = grad_lambda_a*grad_pl;
    pl_term += lambda_a*lap_pl;
    pl_term *= rho_a_val;

    Tensor<1,dim> g_val = gravity_fcn.vector_value(p);

    double gravity_term = grad_lambda_a*g_val;
   // gravity_term *= rho_a_val*rho_a_val;
    gravity_term *= rho_a_val_sq;

    return time_der_term + kappa*pca_Sa_term + kappa*pca_Sv_term - kappa*pl_term + kappa*gravity_term;
//    return kappa*pca_Sa_term;
//    return pca_Sa_term + pca_Sv_term - pl_term;
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
    ExactLiquidPressure<dim> ex_liquid_pressure;
    ExactAqueousSaturation<dim> ex_aqueous_saturation;
    ExactVaporSaturation<dim> ex_vapor_saturation;

    rho_v<dim> density_v;

    ex_liquid_pressure.set_time(this->get_time());
    ex_aqueous_saturation.set_time(this->get_time());
    ex_vapor_saturation.set_time(this->get_time());

    lambda_v<dim> ex_lambda_v;

    ex_lambda_v.set_time(this->get_time());

    CapillaryPressurePcv<dim> ex_cap_p_pcv;

    ex_cap_p_pcv.set_time(this->get_time());

    GravitySourceTerm<dim> gravity_fcn;

    gravity_fcn.set_time(this->get_time());

    porosity<dim> ex_porosity;

    double pl = ex_liquid_pressure.value(p);
    double Sa = ex_aqueous_saturation.value(p);
    double Sv = ex_vapor_saturation.value(p);

    double rho_v_val = density_v.value(pl, Sa, Sv);
    double rho_v_val_sq = density_v.value(pl, Sa, Sv)*density_v.value(pl, Sa, Sv);
    if (inc){
        rho_v_val = 1.0;
        rho_v_val_sq = density_v.value(pl, Sa, Sv);
    }

    double lambda_v = ex_lambda_v.value(pl, Sa, Sv);
    Tensor<1,dim> grad_lambda_v = ex_lambda_v.gradient(p);

    Tensor<1,dim> grad_pl = ex_liquid_pressure.gradient(p);
    double lap_pl = ex_liquid_pressure.laplacian(p);

    Tensor<1,dim> grad_Sv = ex_vapor_saturation.gradient(p);
    double lap_Sv = ex_vapor_saturation.laplacian(p);

    double dpcv_dSv = ex_cap_p_pcv.derivative_wrt_Sv(Sv);

    double phi_porosity = ex_porosity.value(pl);

//    double time_der_term = phi_porosity*rho_v_val*sin(this->get_time() + p[0])/8.0;
    double time_der_term = phi_porosity*rho_v_val*ex_vapor_saturation.time_derivative(p)
                           + phi_porosity*Sv*(density_v.derivative_wrt_pl(pl, Sa, Sv)*ex_liquid_pressure.time_derivative(p)
                                              + density_v.derivative_wrt_Sv(pl, Sa, Sv)*ex_vapor_saturation.time_derivative(p));


    double pcv_Sv_term = rho_v_val*lambda_v*dpcv_dSv*lap_Sv;

    double Sv_Sv_term = grad_Sv*grad_Sv;
    Sv_Sv_term *= lambda_v*(dpcv_dSv*density_v.derivative_wrt_Sv(pl, Sa, Sv)
                            + rho_v_val*ex_cap_p_pcv.second_derivative_wrt_Sv(Sv));

    pcv_Sv_term += Sv_Sv_term;

    double lambda_v_Sv_term = grad_lambda_v*grad_Sv;
    lambda_v_Sv_term *= rho_v_val*dpcv_dSv;

    pcv_Sv_term += lambda_v_Sv_term;

    double pl_Sv_term = grad_pl*grad_Sv;
    pl_Sv_term *= lambda_v*dpcv_dSv*density_v.derivative_wrt_pl(pl, Sa, Sv);

    pcv_Sv_term += pl_Sv_term;

    double pl_term = rho_v_val*lambda_v*lap_pl;

    double lambda_v_pl_term = grad_lambda_v*grad_pl;
    lambda_v_pl_term *= rho_v_val;

    pl_term += lambda_v_pl_term;

    double pl_pl_term = grad_pl*grad_pl;
    pl_pl_term *= lambda_v*density_v.derivative_wrt_pl(pl, Sa, Sv);

    pl_term += pl_pl_term;

    double Sv_pl_term = grad_Sv*grad_pl;
    Sv_pl_term *= lambda_v*density_v.derivative_wrt_Sv(pl, Sa, Sv);

    pl_term += Sv_pl_term;

//    double pl_term = grad_lambda_v*grad_pl;
//    pl_term += lambda_v*lap_pl;
//    pl_term *= rho_v_val;

    Tensor<1,dim> g_val = gravity_fcn.vector_value(p);

    double gravity_term = grad_lambda_v*g_val;
   // gravity_term *= rho_v_val*rho_v_val;
    gravity_term *= rho_v_val_sq;

    double aux_g1 = grad_pl*g_val;
    aux_g1 *= 2.0*lambda_v*rho_v_val*density_v.derivative_wrt_pl(pl, Sa, Sv);

    double aux_g2 = grad_Sv*g_val;
    aux_g2 *= 2.0*lambda_v*rho_v_val*density_v.derivative_wrt_Sv(pl, Sa, Sv);

    gravity_term += aux_g1 + aux_g2;

//    std::cout << "pl_term = " << pl_term << std::endl;
    return time_der_term - kappa*pcv_Sv_term - kappa*pl_term + kappa*gravity_term;
//    return time_der_term - kappa*pcv_Sv_term + kappa*gravity_term;
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
    lambda_l<dim> ex_lambda_l;
    lambda_v<dim> ex_lambda_v;
    lambda_a<dim> ex_lambda_a;

    rho_l<dim> ex_rho_l;
    rho_v<dim> ex_rho_v;
    rho_a<dim> ex_rho_a;

    ExactLiquidPressure<dim> ex_liquid_pressure;
    ExactAqueousSaturation<dim> ex_aqueous_sat;
    ExactVaporSaturation<dim> ex_vapor_sat;

    CapillaryPressurePcv<dim> ex_cap_p_pcv;
    CapillaryPressurePca<dim> ex_cap_p_pca;

    ex_lambda_l.set_time(this->get_time());
    ex_lambda_v.set_time(this->get_time());
    ex_lambda_a.set_time(this->get_time());

    ex_liquid_pressure.set_time(this->get_time());
    ex_aqueous_sat.set_time(this->get_time());
    ex_vapor_sat.set_time(this->get_time());

    ex_cap_p_pcv.set_time(this->get_time());
    ex_cap_p_pca.set_time(this->get_time());

    double pl_value = ex_liquid_pressure.value(p);
    double Sa_value = ex_aqueous_sat.value(p);
    double Sv_value = ex_vapor_sat.value(p);

    double lambda_l = ex_lambda_l.value(pl_value, Sa_value, Sv_value);
    double lambda_v = ex_lambda_v.value(pl_value, Sa_value, Sv_value);
    double lambda_a = ex_lambda_a.value(pl_value, Sa_value, Sv_value);

    double rho_l_val = ex_rho_l.value(pl_value);
    double rho_v_val = ex_rho_v.value(pl_value, Sa_value, Sv_value);
    double rho_a_val = ex_rho_a.value(pl_value);

    if (inc){
        rho_l_val = 1.0;
        rho_v_val = 1.0;
        rho_a_val = 1.0;
    }

    double rho_lambda_t = rho_l_val*lambda_l + rho_v_val*lambda_v + rho_a_val*lambda_a;

    Tensor<1,dim> grad_pl = ex_liquid_pressure.gradient(p);

    Tensor<1,dim> grad_pcv = ex_cap_p_pcv.gradient(p);

    Tensor<1,dim> grad_pca = ex_cap_p_pca.gradient(p);

    Tensor<1,dim> pl_term;
//    pl_term = (lambda_l + lambda_v + lambda_a)*grad_pl;
    pl_term = grad_pl;
    pl_term *= rho_lambda_t;
//
    Tensor<1,dim> pcv_term;
    pcv_term = grad_pcv;
    pcv_term *= rho_v_val*lambda_v;

    Tensor<1,dim> pca_term;
    pca_term = grad_pca;
    pca_term *= -rho_a_val*lambda_a;

    Tensor<1,dim> result;
//    result = 0.0;

    result = pl_term;
//    result += pcv_term;
//    result += pca_term;

    result *= kappa;

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
    lambda_a<dim> ex_lambda_a;

    rho_a<dim> ex_rho_a;

    ExactLiquidPressure<dim> ex_liquid_pressure;
    ExactAqueousSaturation<dim> ex_aqueous_sat;
    ExactVaporSaturation<dim> ex_vapor_sat;

    CapillaryPressurePca<dim> ex_cap_p_pca;

    ex_lambda_a.set_time(this->get_time());

    ex_liquid_pressure.set_time(this->get_time());
    ex_aqueous_sat.set_time(this->get_time());
    ex_vapor_sat.set_time(this->get_time());

    double pl_value = ex_liquid_pressure.value(p);
    double Sa_value = ex_aqueous_sat.value(p);
    double Sv_value = ex_vapor_sat.value(p);

    ex_cap_p_pca.set_time(this->get_time());

    double lambda_a = ex_lambda_a.value(pl_value, Sa_value, Sv_value);

    double rho_a_val = ex_rho_a.value(pl_value);

    if(inc){
        rho_a_val = 1.0;
    }

    Tensor<1,dim> grad_pl = ex_liquid_pressure.gradient(p);
    Tensor<1,dim> grad_Sa = ex_aqueous_sat.gradient(p);
    Tensor<1,dim> grad_Sv = ex_vapor_sat.gradient(p);

    double dpca_dSa = ex_cap_p_pca.derivative_wrt_Sa(Sa_value, Sv_value);
    double dpca_dSv = ex_cap_p_pca.derivative_wrt_Sv(Sa_value, Sv_value);

    Tensor<1,dim> Sa_term;
    Sa_term = grad_Sa;
    Sa_term *= -rho_a_val*lambda_a*dpca_dSa;

    Tensor<1,dim> Sv_term;
    Sv_term = grad_Sv;
    Sv_term *= -rho_a_val*lambda_a*dpca_dSv;

    Tensor<1,dim> pl_term;
    pl_term = grad_pl;
    pl_term *= rho_a_val*lambda_a;

    Tensor<1,dim> result;

    result = Sa_term;
//    result += Sv_term;
//    result += pl_term;

    result *= kappa;

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
    lambda_v<dim> ex_lambda_v;

    rho_v<dim> ex_rho_v;

    ExactLiquidPressure<dim> ex_liquid_pressure;
    ExactAqueousSaturation<dim> ex_aqueous_sat;
    ExactVaporSaturation<dim> ex_vapor_sat;

    CapillaryPressurePcv<dim> ex_cap_p_pcv;

    ex_lambda_v.set_time(this->get_time());

    ex_liquid_pressure.set_time(this->get_time());
    ex_aqueous_sat.set_time(this->get_time());
    ex_vapor_sat.set_time(this->get_time());

    double pl_value = ex_liquid_pressure.value(p);
    double Sa_value = ex_aqueous_sat.value(p);
    double Sv_value = ex_vapor_sat.value(p);

    ex_cap_p_pcv.set_time(this->get_time());

    double lambda_v = ex_lambda_v.value(pl_value, Sa_value, Sv_value);

    double rho_v_val = ex_rho_v.value(pl_value, Sa_value, Sv_value);

    if (inc){
        rho_v_val = 1.0;
    }

    Tensor<1,dim> grad_pl = ex_liquid_pressure.gradient(p);
    Tensor<1,dim> grad_Sa = ex_aqueous_sat.gradient(p);
    Tensor<1,dim> grad_Sv = ex_vapor_sat.gradient(p);

    double dpcv_dSv = ex_cap_p_pcv.derivative_wrt_Sv(Sv_value);

    Tensor<1,dim> Sv_term;
    Sv_term = grad_Sv;
    Sv_term *= rho_v_val*lambda_v*dpcv_dSv;

    Tensor<1,dim> pl_term;
    pl_term = grad_pl;
    pl_term *= rho_v_val*lambda_v;

//    std::cout << "exact rho_v_val = " << rho_v_val << " lambda_v = " << lambda_v<< std::endl;
    Tensor<1,dim> result;

    result = Sv_term;
    result += pl_term;

    result *= kappa;

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
