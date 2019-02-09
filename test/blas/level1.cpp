#include <duda/blas/level1.hpp>
#include <duda/random.hpp>

#include <testing.hpp>

template <typename T>
void test_amax(const int n)
{
    const auto x_d = duda::random_normal<T>(n);
    const auto x_h = testing::copy(x_d);

    int result;

    duda::iamax(x_d, result);

    typename testing::host_vector<T>::Index index;

    x_h.cwiseAbs().maxCoeff(&index);

    REQUIRE(result == index);
}

TEST_CASE("amax", "[device_vector][blas]")
{
    test_amax<float>(512);
    test_amax<double>(16);
}

template <typename T>
void test_amin(const int n)
{
    const auto x_d = duda::random_normal<T>(n);
    const auto x_h = testing::copy(x_d);

    int result;

    duda::iamin(x_d, result);

    typename testing::host_vector<T>::Index index;

    x_h.cwiseAbs().minCoeff(&index);

    REQUIRE(result == index);
}

TEST_CASE("amin", "[device_vector][blas]")
{
    test_amin<float>(1080);
    test_amin<double>(666);
}

template <typename T, typename... Dim>
void test_asum(const Dim... dim)
{
    const auto x_d = duda::random_normal<T>(dim...);
    const auto x_h = testing::copy(x_d);

    T result;

    duda::asum(x_d, result);

    REQUIRE(result == Approx(x_h.cwiseAbs().sum()));
}

TEST_CASE("asum", "[device_vector][device_matrix][blas]")
{
    test_asum<float>(1080);
    test_asum<double>(666);

    test_asum<float>(32, 10);
    test_asum<double>(10, 10);
}

template <typename T, typename... Dim>
void test_axpy(const T alpha, const Dim... dim)
{
    const auto x_d = duda::random_normal<T>(dim...);
    auto y_d       = duda::random_normal<T>(dim...);

    const auto x_h = testing::copy(x_d);
    auto y_h       = testing::copy(y_d);

    duda::axpy(alpha, x_d, y_d);
    y_h = alpha * x_h + y_h;

    REQUIRE(testing::all_close(y_d, y_h));
}

TEST_CASE("axpy", "[device_vector][device_matrix][blas]")
{
    test_axpy<float>(0.666, 196);
    test_axpy<double>(0.42, 1080);

    test_axpy<float>(3.14, 5, 6);
    test_axpy<double>(-3.14, 51, 16);
}

template <typename T>
void test_dot(const int n)
{
    const auto x_d = duda::random_normal<T>(n);
    const auto y_d = duda::random_normal<T>(n);

    const auto x_h = testing::copy(x_d);
    const auto y_h = testing::copy(y_d);

    T result;

    duda::dot(x_d, y_d, result);

    REQUIRE(result == Approx(x_h.cwiseProduct(y_h).sum()));
}

TEST_CASE("dot", "[device_vector][blas]")
{
    test_dot<float>(256);
    test_dot<double>(16);
}

template <typename T>
void test_nrm2(const int n)
{
    const auto x_d = duda::random_normal<T>(n);
    const auto x_h = testing::copy(x_d);

    T result;

    duda::nrm2(x_d, result);

    REQUIRE(result == Approx(x_h.norm()));
}

TEST_CASE("nrm2", "[device_vector][blas]")
{
    test_nrm2<float>(32);
    test_nrm2<double>(128);
}

template <typename T>
void test_rot(const int n, const T c, const T s)
{
    auto x_d = duda::random_uniform<T>(n);
    auto y_d = duda::random_uniform<T>(n);

    const auto x_h = testing::copy(x_d);
    const auto y_h = testing::copy(y_d);

    duda::rot(x_d, y_d, c, s);

    const auto x_ans_h = c * x_h + s * y_h;
    const auto y_ans_h = -s * x_h + c * y_h;

    REQUIRE(testing::all_close(x_d, x_ans_h));
    REQUIRE(testing::all_close(y_d, y_ans_h));
}

TEST_CASE("rot", "[device_vector][blas]")
{
    test_rot<float>(32, 3.14, 7.5);
    test_rot<double>(128, 0.666, -1.9);
}

template <typename T, typename... Dim>
void test_scal(const T alpha, const Dim... dim)
{
    auto x_d = duda::random_uniform<T>(dim...);
    auto x_h       = testing::copy(x_d);

    duda::scal(alpha, x_d);
    x_h = alpha * x_h;

    REQUIRE(testing::all_close(x_d, x_h));
}

TEST_CASE("scal", "[device_vector][device_matrix][blas]")
{
    test_scal<float>(3.14, 32, 10);
    test_scal<double>(0.666, 128, 2);

    test_scal<float>(3.14, 16, 4);
    test_scal<double>(0.19, 8, 2);
}
