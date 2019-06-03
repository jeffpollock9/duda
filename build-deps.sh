#!/usr/bin/env bash

readonly build=$(pwd)
readonly deps=${build}/deps

mkdir -p ${deps}/src
mkdir -p ${deps}/builds
mkdir -p ${deps}/install

install_dependency() {
    url=$1
    tag=$2
    name=$3
    flags=$4

    git clone --recurse-submodules --branch ${tag} --single-branch ${url} --depth 1 ${deps}/src/${name}
    cmake -S ${deps}/src/${name} -B ${deps}/builds/${name} -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${deps}/install -DCMAKE_CXX_FLAGS="-march=native" -GNinja ${flags}
    cmake --build ${deps}/builds/${name} --target install
}

install_dependency https://github.com/google/benchmark.git v1.5.0 benchmark -DBENCHMARK_ENABLE_TESTING=OFF
install_dependency https://github.com/pybind/pybind11.git v2.2.4 pybind11 -DPYBIND11_TEST=OFF
install_dependency https://github.com/eigenteam/eigen-git-mirror.git 3.3.7 eigen
install_dependency https://github.com/rapidsai/rmm.git v0.7.0 rmm -DBUILD_TESTS=OFF

echo -e "\n"
echo "** add -DCMAKE_PREFIX_PATH=${deps}/install to the duda cmake command **"
echo -e "\n"
