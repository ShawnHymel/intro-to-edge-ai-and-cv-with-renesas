/***********************************************************************************************************************
 * File Name    : common_utils.h
 * Description  : Contains macros, data structures, and functions commonly used in the EP.
 **********************************************************************************************************************/
/***********************************************************************************************************************
* Copyright (c) 2020 - 2025 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
***********************************************************************************************************************/

#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_

/***********************************************************************************************************************
 * Includes
 **********************************************************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "hal_data.h"

/***********************************************************************************************************************
 * Macro definitions
 **********************************************************************************************************************/
/* Macros for the terminal interface */
#if (USE_VIRTUAL_COM == 1)
  #include "SERIAL_TERM/serial.h"
  #define TERM_BUFFER_SIZE              (SERIAL_RX_MAX_SIZE)
  #define TERM_INIT()                   (serial_init())
  #define TERM_PRINTF(fmt, ...)         (serial_printf((fmt), ##__VA_ARGS__))
  #define TERM_READ(buf, len)           (serial_read((buf), (len)))
  #define TERM_HAS_DATA()               (serial_has_data())
  #define TERM_HAS_KEY()                (serial_has_key())
#else
  #include "SEGGER_RTT/SEGGER_RTT.h"
  #define SEGGER_INDEX                  (0)
  #define TERM_BUFFER_SIZE              (BUFFER_SIZE_DOWN)
  #define TERM_INIT()                   /* No initialization needed for SEGGER RTT */
  #define TERM_PRINTF(fmt, ...)         (SEGGER_RTT_printf(SEGGER_INDEX, (fmt), ##__VA_ARGS__))
  #define TERM_READ(buf, len)           (SEGGER_RTT_Read(SEGGER_INDEX, (buf), (len)))
  #define TERM_HAS_DATA()               (SEGGER_RTT_HasData(SEGGER_INDEX))
  #define TERM_HAS_KEY()                (SEGGER_RTT_HasKey())
#endif /* USE_VIRTUAL_COM */

/* Macros for terminal functionality in the RTOS project */
#if (BSP_CFG_RTOS != 0U)
  #if (BSP_CFG_RTOS == 1U)
    #define TERM_BYTE_POOL_SIZE         (4096U)
  #endif /* BSP_CFG_RTOS == 1U */
  #define TERM_OUTPUT_QUEUE_SIZE        (100U)
  #define TERM_INPUT_QUEUE_SIZE         (100U)
#endif /* BSP_CFG_RTOS != 0U */

/* Macros commonly used */
#define LVL_ERR                         (1U)       /* Error condition */
#define RESET_VALUE                     (0x00)
#define NULL_CHAR                       ('\0')
#define MODULE_CLOSE                    (0U)

#define APP_PRINT(fn_, ...)             (TERM_PRINTF((fn_), ##__VA_ARGS__))

#if LVL_ERR
  #define APP_ERR_PRINT(fn_, ...)       (APP_PRINT("\r\n[ERR] In Function: %s(), %s", __FUNCTION__, (fn_),\
                                                   ##__VA_ARGS__))
#else
  #define APP_ERR_PRINT(fn_, ...)
#endif /* LVL_ERR */

#define APP_ERR_RET(con, err, fn_, ...) ({\
                                        if (con)\
                                        {\
                                        APP_ERR_PRINT((fn_), ##__VA_ARGS__);\
                                        return (err);\
                                        }\
                                        })

#define ERROR_TRAP                      ({ \
                                        __asm("BKPT #0\n");\
                                        })

#if (USE_VIRTUAL_COM == 1)
#define APP_ERR_TRAP(err)               ({\
                                        if(err)\
                                        {\
                                        APP_PRINT("\r\nReturned Error Code: 0x%x  \r\n", (err));\
                                        serial_deinit();\
                                        /* Trap upon the error */ \
                                        ERROR_TRAP; \
                                        }\
                                        })
#else
#define APP_ERR_TRAP(err)               ({\
                                        if(err)\
                                        {\
                                        APP_PRINT("\r\nReturned Error Code: 0x%x  \r\n", (err));\
                                        /* Trap upon the error */ \
                                        ERROR_TRAP; \
                                        }\
                                        })
#endif /* USE_VIRTUAL_COM */

#define APP_READ(buf, len)              (TERM_READ(buf, len))

#define APP_CHECK_DATA                  (TERM_HAS_DATA())

#define APP_CHECK_KEY                   (TERM_HAS_KEY())

/* Macro for handle error */
#define APP_ERR_HANDLE(err, fn_)        ({\
		                                if(err){\
		                                handle_error((err), (uint8_t *)(fn_));\
		                                }\
                                        })

/* sync events */
#define HARDWARE_DISPLAY_INIT_DONE      (1 << 0)
#define HARDWARE_CAMERA_INIT_DONE       (1 << 1)
#define HARDWARE_ETHOSU_INIT_DONE       (1 << 2)
#define SOFTWARE_AI_INFERENCE_INIT_DONE (1 << 3)
#define GLCDC_VSYNC                     (1 << 10)
#define MIPI_MESSAGE_SENT               (1 << 11)
#define CAMERA_CAPTURE_COMPLETED        (1 << 12)
#define AI_INFERENCE_INPUT_IMAGE_READY  (1 << 13)
#define AI_INFERENCE_RESULT_UPDATED     (1 << 14)
#define DISPLAY_PAUSE                   (1 << 15)
#define CAMERA_AUTO_FOCUS_EXECUTE       (1 << 16)

#define APP_ERROR_TRAP(err)        if(err) { __asm("BKPT #0\n");} /* system execution breaks  */


/***********************************************************************************************************************
 * Typedef definitions
 **********************************************************************************************************************/
/* Structure for exchanging information between application threads and the terminal thread */
#if (BSP_CFG_RTOS != 0U)
typedef struct st_term_msg
{
    uint32_t id;
    uint32_t size;
    uint32_t time;
    char msg[];
}term_msg_t;
#endif /* BSP_CFG_RTOS != 0U */


/* The coordinate of the bounding box corner, bounding box width and height based on 192x192 gray pixel area */
typedef struct ai_detection_point_t {
  signed short      m_x;
  signed short      m_y;
  signed short      m_w;
  signed short      m_h;
} st_ai_detection_point_t;

/* The possibilities of the detected category based on 224x224x3 rgb pixel area */
typedef struct ai_classification_point_t {
  unsigned short    category;
  float             prob;
} st_ai_classification_point_t;

typedef enum
{
    CAM_VGA_WIDTH          = 640,
    CAM_VGA_HEIGHT         = 480,
    CAM_QVGA_WIDTH         = 320,
    CAM_QVGA_HEIGHT        = 240,

} camera_size_list_t;

#define CAM_BYTE_PER_PIXEL              (2)
#define RGB888_BYTE_PER_PIXEL           (3)

/* image intermediate input size to convert to MIPI display size */
#define IMAGE_INPUT_WIDTH       CAM_QVGA_HEIGHT
#define IMAGE_INPUT_HEIGHT      CAM_QVGA_HEIGHT


/** Common error codes */
typedef enum e_vision_ai_app_err
{
    VISION_AI_APP_SUCCESS                = 0,

    VISION_AI_APP_ERR_AI_INIT            = 1,  ///< AI init failed
    VISION_AI_APP_ERR_AI_INFERENCE       = 2,  ///< AI inference failed
    VISION_AI_APP_ERR_IMG_PROCESS        = 3,  ///< Image crop failed
    VISION_AI_APP_ERR_IMG_ROTATION       = 4,  ///< Image rotation failed
    VISION_AI_APP_ERR_NULL_POINTER       = 5,  ///< null pointer
    VISION_AI_APP_ERR_GLCDC_OPEN         = 6,  ///< glcdc open failed
    VISION_AI_APP_ERR_MIPI_CMD           = 7,  ///< mipi command failed
    VISION_AI_APP_ERR_GLCDC_START        = 8,  ///< glcdc start failed
    VISION_AI_APP_ERR_GLCDC_LAYER_CHANGE = 9,  ///< graphics layer change failed
    VISION_AI_APP_ERR_GRAPHICS_INIT      = 10, ///< One of the graphics system initialization failed
    VISION_AI_APP_ERR_GPT_OPEN           = 11, ///< GPT open failed
    VISION_AI_APP_ERR_CEU_OPEN           = 12, ///< CEU open failed
    VISION_AI_APP_ERR_WRITE_OV3640_REG   = 13, ///< Write OV3640 register failed
    VISION_AI_APP_ERR_WRITE_SENSOR_ARRAY = 14, ///< Write OV3640 register array failed
    VISION_AI_APP_ERR_CAMERA_INIT        = 15, ///< Camera init failed
    VISION_AI_APP_ERR_IIC_MASTER_OPEN    = 16, ///< IIC master open failed
    VISION_AI_APP_ERR_IIC_MASTER_WRITE   = 17, ///< IIC master write failed
    VISION_AI_APP_ERR_IIC_MASTER_READ    = 18, ///< IIC master read failed
    VISION_AI_APP_ERR_CONSOLE_OPEN       = 19, ///< jlink uart open error
    VISION_AI_APP_ERR_CONSOLE_WRITE      = 20, ///< JLink console write failed
    VISION_AI_APP_ERR_CONSOLE_READ       = 21, ///< JLink console read failed
    VISION_AI_APP_ERR_EXTERNAL_IRQ_INIT  = 22, ///< External IRQ init failed
} vision_ai_app_err_t;

/** process_time report */
typedef struct st_processing_time_info_t
{
    uint32_t camera_image_capture_time_ms;          ///< Camera frame capture time
    uint32_t camera_post_processing_time_ms;        ///< Post processing time for captured camera image
    uint32_t lcd_display_update_refresh_ms;         ///< LCD display refresh time
    uint32_t ai_inference_pre_processing_time_ms;   ///< Pre processing time for AI inference
    uint32_t ai_inference_time_ms;                  ///< AI inference processing time
} processinf_time_info_t;

/***********************************************************************************************************************
 * Public function prototypes
 **********************************************************************************************************************/
/* Terminal API prototype for the RTOS project */
#if (BSP_CFG_RTOS != 0U)
void term_framework_init_check(void);
uint32_t term_framework_init(void);
uint32_t term_get_input_queue(char * p_msg, uint32_t * p_size, uint32_t wait);
uint32_t term_get_output_queue(void ** pp_msg_st, uint32_t wait);
uint32_t term_send_input_queue(uint32_t id, void * const p_data, uint32_t size);
uint32_t term_send_output_queue(uint32_t id, void * const p_data, uint32_t size);
#endif /* BSP_CFG_RTOS != 0U */

#endif /* COMMON_UTILS_H_ */
