/* generated vector source file - do not edit */
#include "bsp_api.h"
/* Do not build these data structures if no interrupts are currently allocated because IAR will have build errors. */
#if VECTOR_DATA_IRQ_COUNT > 0
        BSP_DONT_REMOVE const fsp_vector_t g_vector_table[BSP_ICU_VECTOR_NUM_ENTRIES] BSP_PLACE_IN_SECTION(BSP_SECTION_APPLICATION_VECTORS) =
        {
                        [0] = rm_ethosu_isr, /* NPU IRQ (NPU IRQ) */
            [1] = sci_b_uart_eri_isr, /* SCI8 ERI (Receive error) */
            [2] = sci_b_uart_rxi_isr, /* SCI8 RXI (Receive data full) */
            [3] = sci_b_uart_tei_isr, /* SCI8 TEI (Transmit end) */
            [4] = sci_b_uart_txi_isr, /* SCI8 TXI (Transmit data empty) */
        };
        #if BSP_FEATURE_ICU_HAS_IELSR
        const bsp_interrupt_event_t g_interrupt_event_link_select[BSP_ICU_VECTOR_NUM_ENTRIES] =
        {
            [0] = BSP_PRV_VECT_ENUM(EVENT_NPU_IRQ,GROUP0), /* NPU IRQ (NPU IRQ) */
            [1] = BSP_PRV_VECT_ENUM(EVENT_SCI8_ERI,GROUP1), /* SCI8 ERI (Receive error) */
            [2] = BSP_PRV_VECT_ENUM(EVENT_SCI8_RXI,GROUP2), /* SCI8 RXI (Receive data full) */
            [3] = BSP_PRV_VECT_ENUM(EVENT_SCI8_TEI,GROUP3), /* SCI8 TEI (Transmit end) */
            [4] = BSP_PRV_VECT_ENUM(EVENT_SCI8_TXI,GROUP4), /* SCI8 TXI (Transmit data empty) */
        };
        #endif
        #endif
