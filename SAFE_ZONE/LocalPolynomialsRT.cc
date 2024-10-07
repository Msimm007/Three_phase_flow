#include "LocalPolynomialsRT.hh"

// This class contains the polynomials in the local space for the RT projection on quads in 3D
// It is based on the deal.II class 'polynomials_raviart_thomas.h', with the difference that the
// degree of the polynomials in the current coordinate is one less, i.e., k-1 x k x k, k x k-1 x k, k x k x k-1
// instead of k+1 x k x k, k x k+1 x k, k x k x k+1

using namespace dealii;

template <int dim>
LocalPolynomialsRT<dim>::LocalPolynomialsRT(const unsigned int k)
  : TensorPolynomialsBase<dim>(k, n_polynomials(k))
  , polynomial_space(create_polynomials(k))
{}

template <int dim>
std::vector<std::vector<Polynomials::Polynomial<double>>>
LocalPolynomialsRT<dim>::create_polynomials(const unsigned int k)
{
  // Create a vector of polynomial spaces where the first element
  // has degree k+1 and the rest has degree k. This corresponds to
  // the space of single-variable polynomials from which we will create the
  // space for the first component of the RT element by way of tensor
  // product.
  //
  // The other components of the RT space can be created by rotating
  // this vector of single-variable polynomials.
  //
  std::vector<std::vector<Polynomials::Polynomial<double>>> pols(dim);
  if (k == 0)
    {
      // k = 0 is not defined, so this does not matter.
	  // For future work: create an error for this case
      pols[0] = Polynomials::LagrangeEquidistant::generate_complete_basis(0);
      for (unsigned int d = 1; d < dim; ++d)
        pols[d] = Polynomials::Legendre::generate_complete_basis(1);
    }
  else
    {
      pols[0] =
        Polynomials::LagrangeEquidistant::generate_complete_basis(k-1);
      for (unsigned int d = 1; d < dim; ++d)
        pols[d] = Polynomials::LagrangeEquidistant::generate_complete_basis(k);
    }

  return pols;
}

template <int dim>
void
LocalPolynomialsRT<dim>::evaluate(
  const Point<dim> &           unit_point,
  std::vector<Tensor<1, dim>> &values,
  std::vector<Tensor<2, dim>> &grads,
  std::vector<Tensor<3, dim>> &grad_grads,
  std::vector<Tensor<4, dim>> &third_derivatives,
  std::vector<Tensor<5, dim>> &fourth_derivatives) const
{
  Assert(values.size() == this->n() || values.size() == 0,
         ExcDimensionMismatch(values.size(), this->n()));
  Assert(grads.size() == this->n() || grads.size() == 0,
         ExcDimensionMismatch(grads.size(), this->n()));
  Assert(grad_grads.size() == this->n() || grad_grads.size() == 0,
         ExcDimensionMismatch(grad_grads.size(), this->n()));
  Assert(third_derivatives.size() == this->n() || third_derivatives.size() == 0,
         ExcDimensionMismatch(third_derivatives.size(), this->n()));
  Assert(fourth_derivatives.size() == this->n() ||
           fourth_derivatives.size() == 0,
         ExcDimensionMismatch(fourth_derivatives.size(), this->n()));

  // have a few scratch
  // arrays. because we don't want to
  // re-allocate them every time this
  // function is called, we make them
  // static. however, in return we
  // have to ensure that the calls to
  // the use of these variables is
  // locked with a mutex. if the
  // mutex is removed, several tests
  // (notably
  // deal.II/create_mass_matrix_05)
  // will start to produce random
  // results in multithread mode
  static std::mutex           mutex;
  std::lock_guard<std::mutex> lock(mutex);

  static std::vector<double>         p_values;
  static std::vector<Tensor<1, dim>> p_grads;
  static std::vector<Tensor<2, dim>> p_grad_grads;
  static std::vector<Tensor<3, dim>> p_third_derivatives;
  static std::vector<Tensor<4, dim>> p_fourth_derivatives;

  const unsigned int n_sub = polynomial_space.n();
  p_values.resize((values.size() == 0) ? 0 : n_sub);
  p_grads.resize((grads.size() == 0) ? 0 : n_sub);
  p_grad_grads.resize((grad_grads.size() == 0) ? 0 : n_sub);
  p_third_derivatives.resize((third_derivatives.size() == 0) ? 0 : n_sub);
  p_fourth_derivatives.resize((fourth_derivatives.size() == 0) ? 0 : n_sub);

  for (unsigned int d = 0; d < dim; ++d)
    {
      // First we copy the point. The
      // polynomial space for
      // component d consists of
      // polynomials of degree k+1 in
      // x_d and degree k in the
      // other variables. in order to
      // simplify this, we use the
      // same AnisotropicPolynomial
      // space and simply rotate the
      // coordinates through all
      // directions.
      Point<dim> p;
      for (unsigned int c = 0; c < dim; ++c)
        p(c) = unit_point((c + d) % dim);

      polynomial_space.evaluate(p,
                                p_values,
                                p_grads,
                                p_grad_grads,
                                p_third_derivatives,
                                p_fourth_derivatives);

      for (unsigned int i = 0; i < p_values.size(); ++i)
        values[i + d * n_sub][d] = p_values[i];

      for (unsigned int i = 0; i < p_grads.size(); ++i)
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          grads[i + d * n_sub][d][(d1 + d) % dim] = p_grads[i][d1];

      for (unsigned int i = 0; i < p_grad_grads.size(); ++i)
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            grad_grads[i + d * n_sub][d][(d1 + d) % dim][(d2 + d) % dim] =
              p_grad_grads[i][d1][d2];

      for (unsigned int i = 0; i < p_third_derivatives.size(); ++i)
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            for (unsigned int d3 = 0; d3 < dim; ++d3)
              third_derivatives[i + d * n_sub][d][(d1 + d) % dim]
                               [(d2 + d) % dim][(d3 + d) % dim] =
                                 p_third_derivatives[i][d1][d2][d3];

      for (unsigned int i = 0; i < p_fourth_derivatives.size(); ++i)
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            for (unsigned int d3 = 0; d3 < dim; ++d3)
              for (unsigned int d4 = 0; d4 < dim; ++d4)
                fourth_derivatives[i + d * n_sub][d][(d1 + d) % dim]
                                  [(d2 + d) % dim][(d3 + d) % dim]
                                  [(d4 + d) % dim] =
                                    p_fourth_derivatives[i][d1][d2][d3][d4];
    }
}

template <int dim>
unsigned int
LocalPolynomialsRT<dim>::n_polynomials(const unsigned int k)
{
  if (dim == 1)
    return k - 1;
  if (dim == 2)
    return 2 * (k + 1) * (k);
  if (dim == 3)
    return 3 * (k + 1) * (k + 1) * (k);

  Assert(false, ExcNotImplemented());
  return 0;
}


template <int dim>
std::unique_ptr<TensorPolynomialsBase<dim>>
LocalPolynomialsRT<dim>::clone() const
{
  return std::make_unique<LocalPolynomialsRT<dim>>(*this);
}
//
//
template class LocalPolynomialsRT<1>;
template class LocalPolynomialsRT<2>;
template class LocalPolynomialsRT<3>;
