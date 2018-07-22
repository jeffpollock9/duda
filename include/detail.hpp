#ifndef DUDA_DETAIL_HPP_
#define DUDA_DETAIL_HPP_

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
        return f(args...);
    }
};

template <>
struct overload<double>
{
    template <typename Single, typename Double, typename... Args>
    static auto call(const Single, const Double g, Args&&... args)
    {
        return g(args...);
    }
};

} // namespace detail

} // namespace duda

#endif /* DUDA_DETAIL_HPP_ */