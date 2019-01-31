#ifndef DUDA_MACROS_HPP_
#define DUDA_MACROS_HPP_

#ifdef __GNUC__
#define DUDA_UNLIKELY(x) __builtin_expect(x, 0)
#else
#define DUDA_UNLIKELY(x) x
#endif

#endif /* DUDA_MACROS_HPP_ */
