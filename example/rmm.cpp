#include <rmm/rmm.h>

#include <iostream>

int main()
{
    double* x;
    cudaStream_t stream;
    int bytes = 100;

    rmmOptions_t options;
    options.allocation_mode = rmmAllocationMode_t::PoolAllocation;
    options.initial_pool_size = 0;
    options.enable_logging = true;

    auto init_code = rmmInitialize(&options);
    std::cout << "init options: " << rmmGetErrorString(init_code) << "\n";

    auto stream_create = cudaStreamCreate(&stream);
    std::cout << "stream create string: " << cudaGetErrorString(stream_create)
              << "\n";
    std::cout << "stream create name: " << cudaGetErrorName(stream_create)
              << "\n";

    auto alloc_result = RMM_ALLOC(&x, bytes, stream);
    std::cout << "alloc: " << rmmGetErrorString(alloc_result) << "\n";

    bool init = rmmIsInitialized(&options);
    std::cout << "is init: " << init << "\n";

    auto free_result = RMM_FREE(x, stream);
    std::cout << "free: " << rmmGetErrorString(free_result) << "\n";

    auto stream_destroy = cudaStreamDestroy(stream);
    std::cout << "stream destroy string: " << cudaGetErrorString(stream_destroy)
              << "\n";
    std::cout << "stream destroy name: " << cudaGetErrorName(stream_destroy)
              << "\n";

    auto fin_code = rmmFinalize();
    std::cout << "fin options: " << rmmGetErrorString(fin_code) << "\n";

    return 0;
}
