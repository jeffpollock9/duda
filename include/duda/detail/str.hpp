#ifndef DUDA_DETAIL_STR_HPP_
#define DUDA_DETAIL_STR_HPP_

#include <sstream>
#include <string>

namespace duda
{

namespace detail
{

template <typename T>
inline std::string str(const T& t)
{
    std::ostringstream os;
    os << t;
    return os.str();
}

} // namespace detail

} // namespace duda

#endif /* DUDA_DETAIL_STR_HPP_ */
