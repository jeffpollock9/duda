#ifndef DUDA_UTILITY_PRINT_PRECISION_HPP_
#define DUDA_UTILITY_PRINT_PRECISION_HPP_

namespace duda
{

struct print_precision_wrapper
{
    explicit print_precision_wrapper(const int value) : value_(value)
    {}

    int& value()
    {
        return value_;
    }

private:
    int value_;
};

inline print_precision_wrapper& print_precision()
{
    static print_precision_wrapper precision(4);
    return precision;
}

} // namespace duda

#endif /* DUDA_UTILITY_PRINT_PRECISION_HPP_ */
