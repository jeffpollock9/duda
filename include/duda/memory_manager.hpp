#ifndef DUDA_MEMORY_MANAGER_HPP_
#define DUDA_MEMORY_MANAGER_HPP_

#include <duda/utility/check_error.hpp>
#include <duda/utility/enums.hpp>

#include <rmm/rmm.h>

#include <cstddef>

namespace duda
{

struct memory_manager
{
    memory_manager(const rmmOptions_t& options) : options_{options}
    {
        check_error(rmmInitialize(&options_));
    }

    memory_manager(
        const allocation_mode allocation    = allocation_mode::pool_allocation,
        const std::size_t initial_pool_size = 0,
        const bool enable_logging           = false)
        : memory_manager({static_cast<rmmAllocationMode_t>(allocation),
                          initial_pool_size,
                          enable_logging})
    {}

    ~memory_manager()
    {
        check_error(rmmFinalize());
    }

private:
    rmmOptions_t options_;
};

} // namespace duda

#endif /* DUDA_MEMORY_MANAGER_HPP_ */
