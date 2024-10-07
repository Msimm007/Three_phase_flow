#include "FE_RTProj.hh"
#include "FE_RTProj.templates.hh"


using namespace dealii;

template <int dim, int spacedim>
FE_RTProj<dim, spacedim>::FE_RTProj(const unsigned int p)
  : FE_DGVector_UH<LocalPolynomialsRT<dim>, dim, spacedim>(p, {mapping_raviart_thomas})
{}

template <int dim, int spacedim>
std::string FE_RTProj<dim, spacedim>::get_name() const
{
  // note that the
  // FETools::get_fe_by_name
  // function depends on the
  // particular format of the string
  // this function returns, so they
  // have to be kept in synch

  std::ostringstream namebuf;
  namebuf << "FE_RTProj<" << Utilities::dim_string(dim, spacedim)
          << ">(" << this->degree - 1 << ")";

  return namebuf.str();
}

template class FE_DGVector_UH<LocalPolynomialsRT<1>, 1, 1>;
template class FE_RTProj<1, 1>;
template class FE_DGVector_UH<LocalPolynomialsRT<2>, 2, 2>;
template class FE_RTProj<2, 2>;
template class FE_DGVector_UH<LocalPolynomialsRT<3>, 3, 3>;
template class FE_RTProj<3, 3>;
