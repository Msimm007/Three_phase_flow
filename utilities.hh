#ifndef UTILITIES_HH
#define UTILITIES_HH

#include <deal.II/base/parameter_handler.h>

namespace CouplingPressureSaturation
{
using namespace dealii;

class ParameterReader : public Subscriptor
{
public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string &);

private:
    void declare_parameters();
    ParameterHandler &prm;
};

} // namespace CouplingPressureSaturation

#endif // UTILITIES_HH
