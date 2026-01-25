/* Generated libraris */
#include "hal_data.h"

/* Custom libraries */
#include "camera_layer/camera_layer.h"
#include "common_utils.h"
#include "display_layer/display_screen.h"
#include "inference/model.h"
#include "utils/utils.h"

/******************************************************************************
 * Settings
 ******************************************************************************/

/* Target image dimensions */
#define IMG_WIDTH 48
#define IMG_HEIGHT 48

/* Adjust color/brightness of saved BMP (per-channel) */
/* 100 = no change, 80 = 20% darker, 120 = 20% brighter */
#define BMP_RED_PERCENT    170
#define BMP_GREEN_PERCENT  150
#define BMP_BLUE_PERCENT   200

/* Grayscale conversion coefficients */
#define GRAY_R_COEFF 0.299f
#define GRAY_G_COEFF 0.587f
#define GRAY_B_COEFF 0.114f

/* Labels */
#define NUM_CLASSES 4
const char* class_names[] = {
    "_background",
    "_other",
    "thumbs_down",
    "thumbs_up",
};

/******************************************************************************
 * Static variables
 ******************************************************************************/


/******************************************************************************
 * Function prototypes
 ******************************************************************************/

static void __attribute__((noinline)) image_resize_and_grayscale(const uint16_t *p_input,
									   float *p_output,
									   uint16_t in_width,
									   uint16_t in_height,
									   uint16_t out_width,
									   uint16_t out_height);

/******************************************************************************
 * Functions
 ******************************************************************************/

/**
 * Crop, resize (nearest neighbor), and convert RGB565 to grayscale.
 * NOTE: This is probably not the most efficient way to do this, but it should
 * give you an idea of how we're preprocessing the image
 */
static void __attribute__((noinline)) image_resize_and_grayscale(const uint16_t *p_input,
									   float *p_output,
									   uint16_t in_width,
									   uint16_t in_height,
									   uint16_t out_width,
									   uint16_t out_height)
{
	/* Crop image to center */
	const uint32_t crop_offset = (in_width - in_height) / 2;

	/* Go through each row */
	for (uint32_t y = 0; y < out_height; y++)
	{
		/* Nearest neighbor row mapping: scale output row to input row */
		uint32_t src_y = (in_height * y) / out_height;

		/* Calculate pointer to start of this source row, offset by crop amount */
		const uint16_t *p_row = p_input + (src_y * in_width) + crop_offset;

		/* Go through each pixel in the row */
		for (uint32_t x = 0; x < out_width; x++)
		{
			/* Nearest neighbor column mapping within cropped square region.
			 * Uses in_height (not in_width) since we're sampling from a square crop. */
			uint32_t src_x = (in_height * x) / out_width;
			uint16_t pixel = p_row[src_x];

			/* Extract RGB565 components and scale to 0-255 */
			float r = ((pixel >> 11) & 0x1F) * (255.0f / 31.0f);
			float g = ((pixel >> 5) & 0x3F) * (255.0f / 63.0f);
			float b = (pixel & 0x1F) * (255.0f / 31.0f);

			/* Apply same brightness adjustments used when saving training BMPs */
			r = r * (BMP_RED_PERCENT / 100.0f);
			g = g * (BMP_GREEN_PERCENT / 100.0f);
			b = b * (BMP_BLUE_PERCENT / 100.0f);

			/* Clamp to valid range (brightness boost can exceed 255) */
			if (r > 255.0f) r = 255.0f;
			if (g > 255.0f) g = 255.0f;
			if (b > 255.0f) b = 255.0f;

			/* Grayscale conversion using coefficients */
			float gray = (GRAY_R_COEFF * r) + (GRAY_G_COEFF * g) + (GRAY_B_COEFF * b);
			if (gray > 255.0f) gray = 255.0f;

			/* Output as float for model input */
			*p_output++ = gray;
		}
	}
}

/******************************************************************************
 * Main entry point
 ******************************************************************************/
void hal_entry(void)
{
    fsp_err_t err;
    float* input_ptr;
    float* output_ptr;
    bool pressed = false;
    bool inference_pending = false;
    uint32_t preprocess_time_us;
    uint32_t inference_time_us;

    /* Initialize debugging terminal */
    TERM_INIT();
    APP_PRINT("Image capture\r\n");

    /* Initialize and start microsecond timer */
    err = init_timer(&g_timer0_ctrl, &g_timer0_cfg);
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Timer init failed: %d\r\n", err);
        while(1);
    }

    /* Workaround: Copy config with correct secure/privilege settings */
	rm_ethosu_cfg_t ethosu_cfg = g_rm_ethosu0_cfg;
	ethosu_cfg.secure_enable = 1;
	ethosu_cfg.privilege_enable = 1;

	/* Initialize Ethos core with modified config */
	err = RM_ETHOSU_Open(&g_rm_ethosu0_ctrl, &ethosu_cfg);
	if (FSP_SUCCESS != err) {
		APP_PRINT("  ERROR: RM_ETHOSU_Open failed with code: %d\r\n", err);
		while(1);
	}
	APP_PRINT("Ethos-U NPU successfully initialized\r\n");

	/* Get pointers to model input/output buffers */
	input_ptr = GetModelInputPtr_input();
	output_ptr = GetModelOutputPtr_output_70014();

    /* Enable MIPI I/F on the EK-RA8P1 */
    err = R_IOPORT_PinWrite(&g_ioport_ctrl, MIPI_IF_EN, BSP_IO_LEVEL_LOW);
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Could not toggle MIPI_IF_EN pin\r\n");
        while (1);
    }

    /* Initialize display */
    APP_PRINT("Initializing display...\r\n");
    display_image_buffer_initialize();

    /* Initialize the 2D draw engine */
    err = drw_init();
    if(FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Could not initialize 2D draw enginer\r\n");
        while (1);
    }

    /* Initialize the display peripheral module and connected LCD display */
    err = display_init();
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Could not initialize display\r\n");
        while (1);
    }

    /* Clear camera image buffer */
    APP_PRINT("Initializing camera...\r\n");
    camera_image_buffer_initialize();

    /* Initialize the camera capture peripheral module and connected camera */
    err = camera_init(false);
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Could not initialize camera\r\n");
    }

    /* Start camera capture */
    APP_PRINT("Starting camera capture...\r\n");
    camera_capture_start();
    APP_PRINT("Capture and classify an image by pressing SW1\r\n");

    /* Main loop */
    while (1)
    {
    	/* See if SW1 button has been pressed (with debounce logic) */
		err = check_button_sw1(&pressed);
		if (FSP_SUCCESS != err) {
			APP_PRINT("Error: Could not check SW1 button: %d\r\n", err);
		}

		/* If pressed, set flag to save on next frame */
		if (pressed)
		{
			inference_pending = true;
		}

        /* Poll for camera ready flag */
        if (g_camera_frame_ready)
        {
            g_camera_frame_ready = false;

            /* Post processing for camera image capture. After this completes,
             * image is stored in camera_capture_image_rgb565[].
             */
            camera_capture_post_process();

            /* If button has been pressed, perform inference*/
            if (inference_pending)
            {
            	inference_pending = false;

            	/* Invalidate cache to ensure CPU sees latest data from DMA */
				SCB_InvalidateDCache_by_Addr((uint32_t *)camera_capture_image_rgb565,
											 (int32_t)camera_capture_image_rgb565_size);

				/* Preprocess image before inference (resize, convert to grayscale) */
				preprocess_time_us = micros();
				image_resize_and_grayscale((const uint16_t *)camera_capture_image_rgb565,
										   input_ptr,
										   CAMERA_CAPTURE_IMAGE_WIDTH,
										   CAMERA_CAPTURE_IMAGE_HEIGHT,
										   IMG_WIDTH,
										   IMG_HEIGHT);
				preprocess_time_us = micros() - preprocess_time_us;

				/* Run inference (cache coherency handled by ethosu_dcache.c hooks) */
				inference_time_us = micros();
				RunModel(true);
				inference_time_us = micros() - inference_time_us;

				/* Find predicted class */
				int predicted_class = 0;
				float max_logit = output_ptr[0];
				for (int i = 1; i < NUM_CLASSES; i++) {
					if (output_ptr[i] > max_logit) {
						max_logit = output_ptr[i];
						predicted_class = i;
					}
				}

				/* Print results */
				APP_PRINT("\r\nResults:\r\n");
				APP_PRINT("  Preprocess time: %u us\r\n", preprocess_time_us);
				APP_PRINT("  Inference time: %u us\r\n", inference_time_us);
				APP_PRINT("  Predicted: %d (%s)\r\n", predicted_class, class_names[predicted_class]);
				APP_PRINT("  Logits:\r\n");
				for (int i = 0; i < NUM_CLASSES; i++) {
					APP_PRINT("    %-15s %12.6f\r\n", class_names[i], output_ptr[i]);
				}
            }

            /* Wait for vsync flag */
            while (!g_display_vsync_ready);
            g_display_vsync_ready = false;

            /* Start a new graphics frame */
            graphics_start_frame();

            /* Display the camera image */
            display_camera_image();

            /* Wait for previous frame rendering to finish, then finalize this frame and flip the buffers */
            graphics_end_frame();
        }
    }
}
