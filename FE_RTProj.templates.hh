#ifndef FE_RTProj_templates_h
#define FE_RTProj_templates_h


#include <deal.II/base/config.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_tools.h>

#include <memory>

#include "FE_RTProj.hh"

using namespace dealii;

// This file is a copy of fe_dg_vector.templates.h from deal.II

template <class PolynomialType, int dim, int spacedim>
FE_DGVector_UH<PolynomialType, dim, spacedim>::FE_DGVector_UH(const unsigned int deg,
                                                        MappingKind        map)
  : FE_PolyTensor<dim, spacedim>(
      PolynomialType(deg),
      FiniteElementData<dim>(get_dpo_vector(deg),
                             dim,
                             deg + 1,
                             FiniteElementData<dim>::L2),
      std::vector<bool>(PolynomialType::n_polynomials(deg), true),
      std::vector<ComponentMask>(PolynomialType::n_polynomials(deg),
                                 ComponentMask(dim, true)))
{
  this->mapping_kind                   = {map};
  const unsigned int polynomial_degree = this->tensor_degree();

  QGauss<dim> quadrature(polynomial_degree + 1);
  this->generalized_support_points = quadrature.get_points();

  this->reinit_restriction_and_prolongation_matrices(true, true);
  FETools::compute_projection_matrices(*this, this->restriction, true);
  FETools::compute_embedding_matrices(*this, this->prolongation, true);
}


template <class PolynomialType, int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_DGVector_UH<PolynomialType, dim, spacedim>::clone() const
{
  return std::make_unique<FE_DGVector_UH<PolynomialType, dim, spacedim>>(*this);
}


template <class PolynomialType, int dim, int spacedim>
std::string
FE_DGVector_UH<PolynomialType, dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_DGVector_UH_" << this->poly_space->name() << "<" << dim << ">("
          << this->degree - 1 << ")";
  return namebuf.str();
}


template <class PolynomialType, int dim, int spacedim>
std::vector<unsigned int>
FE_DGVector_UH<PolynomialType, dim, spacedim>::get_dpo_vector(
  const unsigned int deg)
{
  std::vector<unsigned int> dpo(dim + 1);
  dpo[dim] = PolynomialType::n_polynomials(deg);

  return dpo;
}


template <class PolynomialType, int dim, int spacedim>
bool
FE_DGVector_UH<PolynomialType, dim, spacedim>::has_support_on_face(
  const unsigned int,
  const unsigned int) const
{
  return true;
}


template <class PolynomialType, int dim, int spacedim>
std::size_t
FE_DGVector_UH<PolynomialType, dim, spacedim>::memory_consumption() const
{
  Assert(false, ExcNotImplemented());
  return 0;
}

#endif
