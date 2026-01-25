/*
 * Cache maintenance overrides for Ethos-U driver.
 *
 * The Ethos-U driver calls ethosu_flush_dcache() and ethosu_invalidate_dcache()
 * internally to maintain cache coherency between CPU and NPU. These are defined
 * as weak symbols in the driver, defaulting to empty stubs.
 *
 * When D-Cache is enabled (BSP_CFG_DCACHE_ENABLED == 1), these functions MUST
 * be overridden to perform actual cache operations, otherwise the NPU will
 * read stale data and/or the CPU will not see NPU output.
 *
 * NOTE: The driver has a race condition where ethosu_invalidate_dcache() is called
 * BEFORE waiting for NPU completion. To fix this, we override ethosu_inference_end()
 * to do a second invalidate AFTER NPU completion is confirmed.
 *
 * Source: Derived from vision_ai_ethosu_mipicsi_glcd_ek_ra8p1_llvm_mera_mobilenet_v1
 */

#include "common_data.h"
#include "ethosu_driver.h"  /* For struct ethosu_driver */

/* Override weak ethosu_flush_dcache - called by driver BEFORE NPU inference */
void ethosu_flush_dcache(uint32_t *p, size_t bytes)
{
#if (BSP_CFG_DCACHE_ENABLED == 1)
    if (p != NULL)
    {
        SCB_CleanDCache_by_Addr(p, (int32_t) bytes);
    }
    else
    {
        SCB_CleanDCache();
    }
    __DSB();  /* Ensure cache operation completes before proceeding */
#endif
}

/* Override weak ethosu_invalidate_dcache - called by driver AFTER NPU inference */
void ethosu_invalidate_dcache(uint32_t *p, size_t bytes)
{
#if (BSP_CFG_DCACHE_ENABLED == 1)
    if (p != NULL)
    {
        SCB_InvalidateDCache_by_Addr(p, (int32_t) bytes);
    }
    else
    {
        SCB_InvalidateDCache();
    }
    __DSB();  /* Ensure cache operation completes before proceeding */
#endif
}

/*
 * Override weak ethosu_inference_end - called AFTER NPU completes and semaphore is taken.
 * This is the correct place to invalidate cache, as we know the NPU is done writing.
 * The driver's ethosu_invalidate_dcache is called too early (before semaphore wait),
 * so we do a second invalidate here to ensure cache coherency.
 */
void ethosu_inference_end(struct ethosu_driver *drv, void *user_arg)
{
    (void)user_arg;

#if (BSP_CFG_DCACHE_ENABLED == 1)
    /* Invalidate cache for all base addresses that were marked for invalidation.
     * This duplicates the driver's invalidation but at the correct time. */
    for (int i = 0; i < drv->job.num_base_addr; i++)
    {
        if (drv->basep_invalidate_mask & (1 << i))
        {
            SCB_InvalidateDCache_by_Addr((uint32_t *)(uintptr_t)drv->job.base_addr[i],
                                         (int32_t)drv->job.base_addr_size[i]);
        }
    }
    __DSB();
#endif
}
