#include "eye.hpp"

#include "helpers.hpp"

#include <iostream>

TEST_CASE("eye", "[device_matrix][eye]")
{
    device_matrix<double> x_d(3, 3);

    duda::eye(x_d.data(), x_d.rows(), x_d.cols());

    host_matrix<double> x_h = copy(x_d);

    std::cout << x_h << "\n";
}
