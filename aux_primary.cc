#include "aux_primary.hh"

double amp_factor_cap_pressure = 1.e7;

double stab_sa_data = 1000.0;
double stab_sv_data = 5.0;

// if true, makes kappa_abs 1000 times lower in the region [25, 50] x [25, 50] (x [25, 50] if 3d)
bool hetero = true;
