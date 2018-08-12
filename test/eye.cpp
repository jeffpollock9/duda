#include "helpers.hpp"

template <typename T>
void test_eye(const int dim)
{
    device_matrix<double> x_d(dim, dim);

    duda::eye(x_d.data(), dim);

    host_matrix<double> x_h = copy(x_d);

    REQUIRE(x_h.isApprox(host_matrix<double>::Identity(dim, dim)));
}

TEST_CASE("eye", "[device_matrix][eye]")
{
    const int n = 666;

    for (int dim = 7; dim < n; dim += 19)
    {
        test_eye<float>(dim);
        test_eye<double>(dim);
    }
}
