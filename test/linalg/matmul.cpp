#include <duda/linalg/matmul.hpp>
#include <duda/random.hpp>

#include <testing.hpp>

template <typename T>
void test_default_matmul(const int n, const int m, const int k)
{
    const auto a_d = duda::random_normal<T>(n, m);
    const auto b_d = duda::random_normal<T>(m, k);
    const auto c_d = duda::matmul(a_d, b_d);

    const auto a_h = testing::copy(a_d);
    const auto b_h = testing::copy(b_d);
    const auto c_h = a_h * b_h;

    REQUIRE(testing::all_close(c_d, c_h));
}

TEST_CASE("default matmul", "[device_matrix][linalg]")
{
    test_default_matmul<float>(512, 3, 14);
    test_default_matmul<double>(14, 24, 51);
}
