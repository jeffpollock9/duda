#include "helpers.hpp"

template <typename T>
void test_eye(const int dim)
{
    host_matrix<T> ans = host_matrix<T>::Identity(dim, dim);

    {
        device_matrix<T> x(dim, dim);
        duda::eye(x.data(), dim);
        REQUIRE(copy(x).isApprox(ans));
    }
    {
        const device_matrix<T> x = duda::eye<T>(dim);
        REQUIRE(copy(x).isApprox(ans));
    }
}

TEST_CASE("eye", "[device_matrix][eye]")
{
    const int n = 666;

    for (int dim = 7; dim < n; dim += 19)
    {
        test_eye<int>(dim);
        test_eye<float>(dim);
        test_eye<double>(dim);
    }
}
