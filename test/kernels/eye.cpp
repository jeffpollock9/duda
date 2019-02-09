#include <duda/kernels/eye.hpp>

#include <testing.hpp>

template <typename T>
void test_eye(const int dim)
{
    const auto host   = testing::host_matrix<T>::Identity(dim, dim);
    const auto device = duda::eye<T>(dim);

    REQUIRE(testing::all_close(device, host));
}

TEST_CASE("eye", "[device_matrix][eye]")
{
    const int n = 101;

    for (int dim = 7; dim < n; dim += 19)
    {
        test_eye<int>(dim);
        test_eye<float>(dim);
        test_eye<double>(dim);
    }
}
