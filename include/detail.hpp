#ifndef DUDA_DETAIL_HPP_
#define DUDA_DETAIL_HPP_

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
    template <typename Single, typename Double, typename... Args>
    static auto call(const Single f, const Double, Args&&... args)
    {
        return f(std::forward<Args>(args)...);
    }
};

template <>
struct overload<double>
{
    template <typename Single, typename Double, typename... Args>
    static auto call(const Single, const Double f, Args&&... args)
    {
        return f(std::forward<Args>(args)...);
    }
};

} // namespace detail

} // namespace duda

#endif /* DUDA_DETAIL_HPP_ */
