#include "Eigen/Dense"
#include "catch/catch.hpp"

#include "device_matrix.hpp"

TEST_CASE("host -> device -> host", "[device_matrix]")
{
    const int rows = 4;
    const int cols = 4;

    SECTION("using float")
    {
        Eigen::MatrixXf h1 = Eigen::MatrixXf::Random(rows, cols);
        Eigen::MatrixXf h2(rows, cols);

        duda::device_matrix<float> d(h1.data(), rows, cols);

        duda::copy(d, h2.data());

        REQUIRE(h1.isApprox(h2));
    }

    SECTION("using double")
    {
        Eigen::MatrixXd h1 = Eigen::MatrixXd::Random(rows, cols);
        Eigen::MatrixXd h2(rows, cols);

        duda::device_matrix<double> d(h1.data(), rows, cols);

        duda::copy(d, h2.data());

        REQUIRE(h1.isApprox(h2));
    }
}

TEST_CASE("axpy", "[device_matrix]")
{
    const int rows = 3;
    const int cols = 4;

    SECTION("using float")
    {
        const float alpha = 3.14;

        auto x1 = duda::device_matrix<float>::random(rows, cols);
        auto y1 = duda::device_matrix<float>::random(rows, cols);

        Eigen::MatrixXf x2(rows, cols);
        Eigen::MatrixXf y2(rows, cols);

        duda::copy(x1, x2.data());
        duda::copy(y1, y2.data());

        duda::axpy(alpha, x1, y1);
        y2 += alpha * x2;

        Eigen::MatrixXf y3(rows, cols);

        duda::copy(y1, y3.data());

        REQUIRE(y2.isApprox(y3));
    }
}
