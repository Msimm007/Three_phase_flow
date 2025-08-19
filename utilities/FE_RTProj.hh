#ifndef FE_RTPROJ_HH
#define FE_RTPROJ_HH

#include <deal.II/base/config.h>

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomials_bdm.h>
#include <deal.II/base/polynomials_nedelec.h>
#include <deal.II/base/polynomials_raviart_thomas.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/fe/fe_dg_vector.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_poly_tensor.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "LocalPolynomialsRT.hh"

// The class FE_DGVector_UH is a copy of the deal.II FE_DGVector class.
// This copy is needed to create the local spaces for the RT projection in 3D
// Otherwise, the deal.II source files would have to be modified.
// In the future, if the FE_DGVector class is modified by deal.II,
// the class FE_DGVector_UH needs to be updated accordingly.

template <class PolynomialType, int dim, int spacedim = dim>
class FE_DGVector_UH : public FE_PolyTensor<dim, spacedim>
{
public:
  FE_DGVector_UH(const unsigned int p, MappingKind m);

  virtual std::string
  get_name() const override;

  virtual std::unique_ptr<FiniteElement<dim, spacedim>>
  clone() const override;

  virtual bool
  has_support_on_face(const unsigned int shape_index,
                      const unsigned int face_index) const override;

  virtual std::size_t
  memory_consumption() const override;

private:
  static std::vector<unsigned int>
  get_dpo_vector(const unsigned int degree);

  class InternalData : public FiniteElement<dim>::InternalDataBase
  {
  public:
    std::vector<std::vector<Tensor<1, dim>>> shape_values;

    std::vector<std::vector<Tensor<2, dim>>> shape_gradients;
  };
  Table<3, double> interior_weights;
};





//namespace FE_RTProj
//{
using namespace dealii;
//using namespace LocalPolynomialsRT;

template <int dim, int spacedim = dim>
class FE_RTProj
  : public FE_DGVector_UH<LocalPolynomialsRT<dim>, dim, spacedim>
{
//template <int dim, int spacedim = dim>
//class FE_RTProj
//  : public FE_DGVector<PolynomialsRaviartThomas<dim>, dim, spacedim>
//{
public:
  /**
   * Constructor for the Raviart-Thomas element of degree @p p.
   */
	FE_RTProj(const unsigned int p);

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_DGRaviartThomas<dim>(degree)</tt>, with @p dim and @p
   * degree replaced by appropriate values.
   */
	virtual std::string get_name() const override;

};

//}; // namespace FE_RTProj

#endif //FE_RTPROJ_HH
