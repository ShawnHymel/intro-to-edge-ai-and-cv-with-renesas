/*
* Copyright (c) 2020 - 2025 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**********************************************************************************************************************
 * File Name    : image_classification_screen_mipi.c
 * Version      : .
 * Description  : The image classification screen display on mipi lcd.
 *********************************************************************************************************************/
/***************************************************************************************************************************
 * Includes   <System Includes> , "Project Includes"
 ***************************************************************************************************************************/

#include "hal_data.h"
#include <stdio.h>

#include "camera_layer/camera_layer.h"
#include "bg_font_18_full.h"
#include "display_layer_config.h"
#include "display_screen.h"

/*********************************************************************************************************************
 *  display_camera_image function
 *  			 Crops center 480x480 from VGA (640x480) and scales to AI_INPUT size (352x352).
 *  @param   	None
 *  @retval     None.
***********************************************************************************************************************/
void display_camera_image(void)
{
    // Source: VGA 640x480, crop center 480x480, scale to 352x352
    #define SRC_WIDTH   CAMERA_CAPTURE_IMAGE_WIDTH
    #define SRC_HEIGHT  CAMERA_CAPTURE_IMAGE_HEIGHT
    #define CROP_SIZE   480  // Square crop (smaller dimension)
    #define CROP_X      ((SRC_WIDTH - CROP_SIZE) / 2)   // Center horizontally
    #define CROP_Y      0    // No vertical crop needed

    SCB_CleanDCache_by_Addr(&camera_capture_image_rgb565[0], (int32_t)camera_capture_image_rgb565_size);

    /* Ensure cache clean completes before DRW reads */
    __DSB();

    /* Specify camera input. */
    /* Note: The MIPI-DSI display panel of EK-RA8D1 prefers 90-degrees counter-clock-wised rotated image. Therefore input raw data of camera capture image. */
    d2_setblitsrc(d2_handle,
                  (void *)&camera_capture_image_rgb565[0],
                  SRC_WIDTH, SRC_WIDTH, SRC_HEIGHT,
                  d2_mode_rgb565);

	/* Display on MIPI LCD */
	d2_blitcopy(d2_handle,
	            (d2_s32) CROP_SIZE, (d2_s32) CROP_SIZE,       // Source crop size (480x480)
	            (d2_blitpos) CROP_X, (d2_blitpos) CROP_Y,     // Source position (crop offset)
	            (d2_width) (AI_INPUT_WIDTH << 4),             // Dest width in 4.4 fixed point
	            (d2_width) (AI_INPUT_HEIGHT << 4),            // Dest height in 4.4 fixed point
	            (d2_point) (0 << 4), (d2_point) (0 << 4),     // Dest position
	            d2_tm_filter);
}

void process_str(const char* input, char* output, int max_len) {
    int i;
    for (i = 0; input[i] != '\0' && i < max_len - 1; i++) {
        if (input[i] == ',') {
            break;
        }
        output[i] = input[i];
    }
    for(; i < max_len - 1; i++){
        output[i] = ' ';
    }
    output[max_len - 1] = '\0';
}
