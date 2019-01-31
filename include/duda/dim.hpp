#ifndef DUDA_DIM_HPP_
#define DUDA_DIM_HPP_

#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>
#include <duda/op.hpp>

#include <string>
#include <utility>

namespace duda
{

struct dim
{
    dim(const int rows_, const int cols_) : rows(rows_), cols(cols_)
    {}

    template <typename T>
    dim(const device_matrix<T>& x, const op f = op::none)
        : rows(x.rows()), cols(x.cols())
    {
        if (f != op::none)
        {
            std::swap(rows, cols);
        }
    }

    template <typename T>
    dim(const device_vector<T>& x, const op f = op::none)
        : rows(x.size()), cols(1)
    {
        if (f != op::none)
        {
            std::swap(rows, cols);
        }
    }

    int rows;
    int cols;
};

inline bool operator==(const dim& x, const dim& y)
{
    return x.rows == y.rows && x.cols == y.cols;
}

inline bool operator!=(const dim& x, const dim& y)
{
    return x.rows != y.rows || x.rows != y.rows;
}

inline std::string operator+(const std::string& x, const dim& y)
{
    using std::to_string;

    return x + " [" + to_string(y.rows) + ", " + to_string(y.cols) + "]";
}

} // namespace duda

#endif /* DUDA_DIM_HPP_ */
