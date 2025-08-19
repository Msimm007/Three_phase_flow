#ifndef LOCALPOLYNOMIALSRT_HH
#define LOCALPOLYNOMIALSRT_HH

#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_polynomials_base.h>
#include <deal.II/base/tensor_product_polynomials.h>

#include <vector>


//namespace LocalPolynomialsRT
//{
using namespace dealii;

template <int dim>
class LocalPolynomialsRT : public TensorPolynomialsBase<dim>
{
public:
	LocalPolynomialsRT(const unsigned int k);

  /**
   * Compute the value and the first and second derivatives of each Raviart-
   * Thomas polynomial at @p unit_point.
   *
   * The size of the vectors must either be zero or equal <tt>n()</tt>.  In
   * the first case, the function will not compute these values.
   *
   * If you need values or derivatives of all tensor product polynomials then
   * use this function, rather than using any of the <tt>compute_value</tt>,
   * <tt>compute_grad</tt> or <tt>compute_grad_grad</tt> functions, see below,
   * in a loop over all tensor product polynomials.
   */
  void
  evaluate(const Point<dim> &           unit_point,
           std::vector<Tensor<1, dim>> &values,
           std::vector<Tensor<2, dim>> &grads,
           std::vector<Tensor<3, dim>> &grad_grads,
           std::vector<Tensor<4, dim>> &third_derivatives,
           std::vector<Tensor<5, dim>> &fourth_derivatives) const override;

  /**
   * Return the name of the space, which is <tt>RaviartThomas</tt>.
   */
  std::string
  name() const override;

  /**
   * Return the number of polynomials in the space <tt>RT(degree)</tt> without
   * requiring to build an object of PolynomialsRaviartThomas. This is
   * required by the FiniteElement classes.
   */
  static unsigned int
  n_polynomials(const unsigned int degree);

  /**
   * @copydoc TensorPolynomialsBase::clone()
   */
  virtual std::unique_ptr<TensorPolynomialsBase<dim>>
  clone() const override;

private:
  /**
   * An object representing the polynomial space for a single component. We
   * can re-use it by rotating the coordinates of the evaluation point.
   */
  const AnisotropicPolynomials<dim> polynomial_space;

  /**
   * A static member function that creates the polynomial space we use to
   * initialize the #polynomial_space member variable.
   */
  static std::vector<std::vector<Polynomials::Polynomial<double>>> create_polynomials(const unsigned int k);
};


template <int dim>
inline std::string
LocalPolynomialsRT<dim>::name() const
{
  return "LocalRaviartThomas";
}
//} // namespace LocalPolynomialsRT

#endif //LocalPolynomialsRT_HH
