/**
 * D/AVE 2D D-Cache Fix for RA8P1
 *
 * The FSP default implementations of d1_cacheflush() and d1_cacheblockflush()
 * are empty stubs that assume no cache. RA8P1 has a Cortex-M85 with D-Cache,
 * so we need proper cache flush implementations.
 *
 * These functions override the weak FSP implementations (FSP 6.3.0+).
 *
 * See note: https://en-support.renesas.com/knowledgeBase/22353219
 */

#include "r_drw_base.h"

/**
 * Flush entire D-Cache to RAM.
 * Called by D/AVE 2D library before hardware operations.
 */
d1_int_t d1_cacheflush(d1_device * handle, d1_int_t memtype)
{
    (void)handle;
    (void)memtype;

    SCB_CleanDCache();

    return 1;
}

/**
 * Flush specific memory region from D-Cache to RAM.
 * Called by D/AVE 2D library to flush display list data before DRW execution.
 */
d1_int_t d1_cacheblockflush(d1_device * handle,
                            d1_int_t memtype,
                            const void * ptr,
                            d1_uint_t size)
{
    (void)handle;
    (void)memtype;

    SCB_CleanDCache_by_Addr((uint32_t *)ptr, (int32_t)size);

    return 1;
}
