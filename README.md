# duda

Probably a C++ linear algebra library using CUDA and some other stuff.

## dependencies

I couldn't decide how best to manage the `duda` dependencies and resorted to a bash
script which downloads and installs them for you. You then pass
`-DCMAKE_PREFIX_PATH=path/to/deps/install` to `cmake` when you build duda so that
`find_package` is able to find them.

## what's the point of this?

Well, not much really. duda is mostly a project I have been working on to learn about:

1. [cmake](https://cmake.org/)
1. [CUDA](https://developer.nvidia.com/cuda-zone)
1. [RAPIDS Memory Manager](https://github.com/rapidsai/rmm/tree/v0.7.0)
1. [pybind11](https://github.com/pybind/pybind11)
1. [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra)

Aside from a learning experience - duda aims to make using CUDA from C++ and python easy
via the following:

1. A much nicer CUDA interface using "modern" C++
1. Lot's of extra linear algebra functionallity not found in CUDA
1. Automatically managing GPU memory, in addition it is trivial to use a GPU memory pool
   with duda via: [RAPIDS Memory Manager](https://github.com/rapidsai/rmm/tree/v0.7.0)

## install

This example shows all the optional `duda` configure options:

```shell
mkdir path/to/build
(cd path/to/build && sh path/to/duda/build-deps.sh)
cmake -B path/to/build -S path/to/duda -DCMAKE_PREFIX_PATH=path/to/deps/install -DDUDA_PYTHON=ON -DDUDA_TEST=ON -DDUDA_BENCHMARK=ON -DDUDA_EXAMPLE=ON
```

## pyduda

Usage from the build directory after install with `-DDUDA_PYTHON=ON`:

```python
import python.pyduda as duda

x = duda.random_uniform(3, 3)
y = duda.random_uniform(3, 3)
z = duda.matmul(x, y)

print(x)
[ +2.0291e+00 +2.1526e+00 +1.6781e+00 ]
[ +1.6463e+00 +1.7002e+00 +1.2271e+00 ]
[ +8.1151e-01 +9.0760e-01 +7.4390e-01 ]

print(y)
[ +1.5411e-01 +6.1098e-01 +2.3427e-01 ]
[ +4.4517e-01 +3.0728e-01 +8.7933e-01 ]
[ +2.0800e-01 +4.1558e-01 +6.4623e-01 ]

print(z)
[ +1.6200e+00 +2.5986e+00 +3.4527e+00 ]
[ +1.2658e+00 +2.0383e+00 +2.6737e+00 ]
[ +6.8382e-01 +1.0838e+00 +1.4689e+00 ]
```
