#ifndef DUDA_UTILITY_MACROS_HPP_
#define DUDA_UTILITY_MACROS_HPP_

#ifdef __GNUC__
#define DUDA_UNLIKELY(x) __builtin_expect(x, 0)
#else
#define DUDA_UNLIKELY(x) x
#endif

#endif /* DUDA_UTILITY_MACROS_HPP_ */
