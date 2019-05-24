#include <rmm/rmm.h>

#include <iostream>

int main()
{
    double* x;
    cudaStream_t stream;
    int bytes = 100;

    auto stream_create = cudaStreamCreate(&stream);
    std::cout << "stream create: " << stream_create << "\n";

    auto alloc_result = RMM_ALLOC(&x, bytes, stream);
    std::cout << "alloc: " << alloc_result << "\n";

    auto free_result = RMM_FREE(x, stream);
    std::cout << "free: " << free_result << "\n";

    auto stream_destroy = cudaStreamDestroy(stream);
    std::cout << "stream destroy: " << stream_destroy << "\n";

    return 0;
}
