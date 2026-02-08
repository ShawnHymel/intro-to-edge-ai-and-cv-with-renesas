/* generated vector source file - do not edit */
#include "bsp_api.h"
/* Do not build these data structures if no interrupts are currently allocated because IAR will have build errors. */
#if VECTOR_DATA_IRQ_COUNT > 0
        BSP_DONT_REMOVE const fsp_vector_t g_vector_table[BSP_ICU_VECTOR_NUM_ENTRIES] BSP_PLACE_IN_SECTION(BSP_SECTION_APPLICATION_VECTORS) =
        {
                        [0] = iic_master_rxi_isr, /* IIC1 RXI (Receive data full) */
            [1] = iic_master_txi_isr, /* IIC1 TXI (Transmit data empty) */
            [2] = iic_master_tei_isr, /* IIC1 TEI (Transmit end) */
            [3] = iic_master_eri_isr, /* IIC1 ERI (Transfer error) */
            [4] = drw_int_isr, /* DRW INT (DRW interrupt) */
            [5] = glcdc_line_detect_isr, /* GLCDC LINE DETECT (Specified line) */
            [6] = sci_b_spi_rxi_isr, /* SCI0 RXI (Receive data full) */
            [7] = sci_b_spi_txi_isr, /* SCI0 TXI (Transmit data empty) */
            [8] = sci_b_spi_tei_isr, /* SCI0 TEI (Transmit end) */
            [9] = sci_b_spi_eri_isr, /* SCI0 ERI (Receive error) */
            [10] = rm_ethosu_isr, /* NPU IRQ (NPU IRQ) */
            [11] = sci_b_uart_rxi_isr, /* SCI8 RXI (Receive data full) */
            [12] = sci_b_uart_txi_isr, /* SCI8 TXI (Transmit data empty) */
            [13] = sci_b_uart_tei_isr, /* SCI8 TEI (Transmit end) */
            [14] = sci_b_uart_eri_isr, /* SCI8 ERI (Receive error) */
            [15] = mipi_csi_dl_isr, /* MIPICSI DL (Data Lane interrupt) */
            [16] = mipi_csi_gst_isr, /* MIPICSI GST (Generic Short Packet interrupt) */
            [17] = mipi_csi_pm_isr, /* MIPICSI PM (Power Management interrupt) */
            [18] = mipi_csi_rx_isr, /* MIPICSI RX (Receive interrupt) */
            [19] = mipi_csi_vc_isr, /* MIPICSI VC (Virtual Channel interrupt) */
            [20] = vin_err_isr, /* VIN ERR (Interrupt Request for SYNC Error) */
            [21] = vin_irq_isr, /* VIN IRQ (Interrupt Request) */
        };
        #if BSP_FEATURE_ICU_HAS_IELSR
        const bsp_interrupt_event_t g_interrupt_event_link_select[BSP_ICU_VECTOR_NUM_ENTRIES] =
        {
            [0] = BSP_PRV_VECT_ENUM(EVENT_IIC1_RXI,GROUP0), /* IIC1 RXI (Receive data full) */
            [1] = BSP_PRV_VECT_ENUM(EVENT_IIC1_TXI,GROUP1), /* IIC1 TXI (Transmit data empty) */
            [2] = BSP_PRV_VECT_ENUM(EVENT_IIC1_TEI,GROUP2), /* IIC1 TEI (Transmit end) */
            [3] = BSP_PRV_VECT_ENUM(EVENT_IIC1_ERI,GROUP3), /* IIC1 ERI (Transfer error) */
            [4] = BSP_PRV_VECT_ENUM(EVENT_DRW_INT,GROUP4), /* DRW INT (DRW interrupt) */
            [5] = BSP_PRV_VECT_ENUM(EVENT_GLCDC_LINE_DETECT,GROUP5), /* GLCDC LINE DETECT (Specified line) */
            [6] = BSP_PRV_VECT_ENUM(EVENT_SCI0_RXI,GROUP6), /* SCI0 RXI (Receive data full) */
            [7] = BSP_PRV_VECT_ENUM(EVENT_SCI0_TXI,GROUP7), /* SCI0 TXI (Transmit data empty) */
            [8] = BSP_PRV_VECT_ENUM(EVENT_SCI0_TEI,GROUP0), /* SCI0 TEI (Transmit end) */
            [9] = BSP_PRV_VECT_ENUM(EVENT_SCI0_ERI,GROUP1), /* SCI0 ERI (Receive error) */
            [10] = BSP_PRV_VECT_ENUM(EVENT_NPU_IRQ,GROUP2), /* NPU IRQ (NPU IRQ) */
            [11] = BSP_PRV_VECT_ENUM(EVENT_SCI8_RXI,GROUP3), /* SCI8 RXI (Receive data full) */
            [12] = BSP_PRV_VECT_ENUM(EVENT_SCI8_TXI,GROUP4), /* SCI8 TXI (Transmit data empty) */
            [13] = BSP_PRV_VECT_ENUM(EVENT_SCI8_TEI,GROUP5), /* SCI8 TEI (Transmit end) */
            [14] = BSP_PRV_VECT_ENUM(EVENT_SCI8_ERI,GROUP6), /* SCI8 ERI (Receive error) */
            [15] = BSP_PRV_VECT_ENUM(EVENT_MIPICSI_DL,GROUP7), /* MIPICSI DL (Data Lane interrupt) */
            [16] = BSP_PRV_VECT_ENUM(EVENT_MIPICSI_GST,GROUP0), /* MIPICSI GST (Generic Short Packet interrupt) */
            [17] = BSP_PRV_VECT_ENUM(EVENT_MIPICSI_PM,GROUP1), /* MIPICSI PM (Power Management interrupt) */
            [18] = BSP_PRV_VECT_ENUM(EVENT_MIPICSI_RX,GROUP2), /* MIPICSI RX (Receive interrupt) */
            [19] = BSP_PRV_VECT_ENUM(EVENT_MIPICSI_VC,GROUP3), /* MIPICSI VC (Virtual Channel interrupt) */
            [20] = BSP_PRV_VECT_ENUM(EVENT_VIN_ERR,GROUP4), /* VIN ERR (Interrupt Request for SYNC Error) */
            [21] = BSP_PRV_VECT_ENUM(EVENT_VIN_IRQ,GROUP5), /* VIN IRQ (Interrupt Request) */
        };
        #endif
        #endif
