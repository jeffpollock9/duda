#include <duda/random.hpp>

#include <testing.hpp>

template <typename T>
void test_repeatable(const int n)
{
    duda::curand_generator().seed(42);
    const auto x = duda::random_uniform<T>(n);

    duda::curand_generator().seed(42);
    const auto y = duda::random_uniform<T>(n);

    REQUIRE(testing::all_close(x, y));
}

TEST_CASE("repeatable", "[random]")
{
    test_repeatable<float>(32);
    test_repeatable<double>(128);
}
