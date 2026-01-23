/* generated vector source file - do not edit */
#include "bsp_api.h"
/* Do not build these data structures if no interrupts are currently allocated because IAR will have build errors. */
#if VECTOR_DATA_IRQ_COUNT > 0
        BSP_DONT_REMOVE const fsp_vector_t g_vector_table[BSP_ICU_VECTOR_NUM_ENTRIES] BSP_PLACE_IN_SECTION(BSP_SECTION_APPLICATION_VECTORS) =
        {
                        [0] = rm_ethosu_isr, /* NPU IRQ (NPU IRQ) */
            [1] = iic_master_rxi_isr, /* IIC1 RXI (Receive data full) */
            [2] = iic_master_txi_isr, /* IIC1 TXI (Transmit data empty) */
            [3] = iic_master_tei_isr, /* IIC1 TEI (Transmit end) */
            [4] = iic_master_eri_isr, /* IIC1 ERI (Transfer error) */
            [5] = sci_b_uart_eri_isr, /* SCI8 ERI (Receive error) */
            [6] = sci_b_uart_rxi_isr, /* SCI8 RXI (Receive data full) */
            [7] = sci_b_uart_tei_isr, /* SCI8 TEI (Transmit end) */
            [8] = sci_b_uart_txi_isr, /* SCI8 TXI (Transmit data empty) */
        };
        #if BSP_FEATURE_ICU_HAS_IELSR
        const bsp_interrupt_event_t g_interrupt_event_link_select[BSP_ICU_VECTOR_NUM_ENTRIES] =
        {
            [0] = BSP_PRV_VECT_ENUM(EVENT_NPU_IRQ,GROUP0), /* NPU IRQ (NPU IRQ) */
            [1] = BSP_PRV_VECT_ENUM(EVENT_IIC1_RXI,GROUP1), /* IIC1 RXI (Receive data full) */
            [2] = BSP_PRV_VECT_ENUM(EVENT_IIC1_TXI,GROUP2), /* IIC1 TXI (Transmit data empty) */
            [3] = BSP_PRV_VECT_ENUM(EVENT_IIC1_TEI,GROUP3), /* IIC1 TEI (Transmit end) */
            [4] = BSP_PRV_VECT_ENUM(EVENT_IIC1_ERI,GROUP4), /* IIC1 ERI (Transfer error) */
            [5] = BSP_PRV_VECT_ENUM(EVENT_SCI8_ERI,GROUP5), /* SCI8 ERI (Receive error) */
            [6] = BSP_PRV_VECT_ENUM(EVENT_SCI8_RXI,GROUP6), /* SCI8 RXI (Receive data full) */
            [7] = BSP_PRV_VECT_ENUM(EVENT_SCI8_TEI,GROUP7), /* SCI8 TEI (Transmit end) */
            [8] = BSP_PRV_VECT_ENUM(EVENT_SCI8_TXI,GROUP0), /* SCI8 TXI (Transmit data empty) */
        };
        #endif
        #endif
