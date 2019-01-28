#ifndef DUDA_DETAIL_HPP_
#define DUDA_DETAIL_HPP_

#include "complex.hpp"

#include <utility>

namespace duda
{

namespace detail
{

template <typename>
struct overload
{};

template <>
struct overload<float>
{
    template <typename Single,
              typename Double,
              typename Complex,
              typename DoubleComplex,
              typename... Args>
    static auto call(const Single f,
                     const Double,
                     const Complex,
                     const DoubleComplex,
                     Args&&... args)
    {
        return f(std::forward<Args>(args)...);
    }
};

template <>
struct overload<double>
{
    template <typename Single,
              typename Double,
              typename Complex,
              typename DoubleComplex,
              typename... Args>
    static auto call(const Single,
                     const Double f,
                     const Complex,
                     const DoubleComplex,
                     Args&&... args)
    {
        return f(std::forward<Args>(args)...);
    }
};

template <>
struct overload<complex>
{
    template <typename Single,
              typename Double,
              typename Complex,
              typename DoubleComplex,
              typename... Args>
    static auto call(const Single,
                     const Double,
                     const Complex f,
                     const DoubleComplex,
                     Args&&... args)
    {
        return f(std::forward<Args>(args)...);
    }
};

template <>
struct overload<double_complex>
{
    template <typename Single,
              typename Double,
              typename Complex,
              typename DoubleComplex,
              typename... Args>
    static auto call(const Single,
                     const Double,
                     const Complex,
                     const DoubleComplex f,
                     Args&&... args)
    {
        return f(std::forward<Args>(args)...);
    }
};

struct not_callable
{};

} // namespace detail

} // namespace duda

#endif /* DUDA_DETAIL_HPP_ */
