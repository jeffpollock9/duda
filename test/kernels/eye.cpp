#include <duda/kernels/eye.hpp>

#include <testing.hpp>

template <typename T>
void test_eye(const int dim)
{
    const auto ans = testing::host_matrix<T>::Identity(dim, dim);

    {
        duda::device_matrix<T> x(dim, dim);
        duda::eye(x.data(), dim);
        REQUIRE(testing::all_close(x, ans));
    }
    {
        const auto x = duda::eye<T>(dim);
        REQUIRE(testing::all_close(x, ans));
    }
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
