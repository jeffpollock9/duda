#ifndef DUDA_DETAIL_OVERLOAD_HPP_
#define DUDA_DETAIL_OVERLOAD_HPP_

#include <duda/utility/types.hpp>

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
              typename DoubleComplex>
    static auto
    fn(const Single f, const Double, const Complex, const DoubleComplex)
    {
        return f;
    }
};

template <>
struct overload<double>
{
    template <typename Single,
              typename Double,
              typename Complex,
              typename DoubleComplex>
    static auto
    fn(const Single, const Double f, const Complex, const DoubleComplex)
    {
        return f;
    }
};

template <>
struct overload<complex>
{
    template <typename Single,
              typename Double,
              typename Complex,
              typename DoubleComplex>
    static auto
    fn(const Single, const Double, const Complex f, const DoubleComplex)
    {
        return f;
    }
};

template <>
struct overload<double_complex>
{
    template <typename Single,
              typename Double,
              typename Complex,
              typename DoubleComplex>
    static auto
    fn(const Single, const Double, const Complex, const DoubleComplex f)
    {
        return f;
    }
};

struct not_callable
{};

} // namespace detail

} // namespace duda

#endif /* DUDA_DETAIL_OVERLOAD_HPP_ */
