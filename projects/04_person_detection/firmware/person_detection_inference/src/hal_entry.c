/* Generated libraries */
#include "hal_data.h"

/* Custom libraries */
#include "camera_layer/camera_layer.h"
#include "common_utils.h"
#include "display_layer/display_screen.h"
#include "display_layer/display_layer_config.h"
#include "inference/model.h"
#include "postprocess/postprocess.h"
#include "utils/utils.h"

/******************************************************************************
 * Settings
 ******************************************************************************/

/* Target image dimensions */
#define IMG_WIDTH 320
#define IMG_HEIGHT 320
#define IMG_CHANNELS 3

/* YOLO detection settings */
#define NUM_CLASSES 1
#define NUM_ANCHORS 3
#define NUM_ATTRS 6     /* tx, ty, tw, th, obj, cls */
#define CONF_THRESHOLD 0.5f
#define IOU_THRESHOLD 0.5f
#define MAX_DETECTIONS 10

/* Grid sizes */
#define GRID_1_SIZE 10   /* Coarse grid for larger objects */
#define GRID_2_SIZE 10  /* Fine grid for smaller objects */

/* Quantization parameters */
#define INPUT_SCALE 0.0039215689f
#define INPUT_ZP -128
#define OUTPUT_GRID1_SCALE 0.0829966888f
#define OUTPUT_GRID1_ZP -14
#define OUTPUT_GRID2_SCALE 0.0829966888f
#define OUTPUT_GRID2_ZP -14

/* Anchor masks */
/* Grid 1 (10x10) uses anchors: [3, 4, 5] */
/* Grid 2 (10x10) uses anchors: [0, 1, 2] */

/* Anchors for grid 1 (10x10 - larger objects) */
static const float anchors_grid1[NUM_ANCHORS][2] = {
    {115.0f, 73.0f},  /* anchor 3 */
    {119.0f, 199.0f},  /* anchor 4 */
    {242.0f, 238.0f},  /* anchor 5 */
};

/* Anchors for grid 2 (10x10 - smaller objects) */
static const float anchors_grid2[NUM_ANCHORS][2] = {
    {12.0f, 18.0f},  /* anchor 0 */
    {37.0f, 49.0f},  /* anchor 1 */
    {52.0f, 132.0f},  /* anchor 2 */
};

/* Labels */
const char* class_names[] = {
    "person",
};

/* Run inference every Nth frame */
#define INFERENCE_INTERVAL 10

/* Enable serial terminal output */
#define SERIAL_PRINT_RESULTS 1

/* Display box settings */
#define BOX_COLOR       0x00FF00  /* Green (24-bit RGB) */
#define BOX_LINE_WIDTH  3         /* Pixels */

/* Scale factor: model coords (IMG_WIDTH) -> display coords (AI_INPUT_WIDTH) */
#define DISPLAY_SCALE ((float)AI_INPUT_WIDTH / (float)IMG_WIDTH)

/******************************************************************************
 * Persistent detection state (drawn every frame, updated on inference frames)
 ******************************************************************************/
static detection_t g_detections[MAX_DETECTIONS * 2];
static int g_detection_count = 0;

/******************************************************************************
 * Function prototypes
 ******************************************************************************/

static void __attribute__((noinline)) image_crop_and_resize_rgb(const uint16_t *p_input,
								int8_t *p_output,
								uint16_t in_width,
								uint16_t in_height,
								uint16_t out_width,
								uint16_t out_height);

/******************************************************************************
 * Functions
 ******************************************************************************/

/**
 * Crop to center square, resize (nearest neighbor), and convert RGB565 to
 * RGB int8.  Output is HWC order (R,G,B,R,G,B,...) with values shifted to
 * int8 range: int8_val = uint8_pixel - 128.
 */
static void __attribute__((noinline)) image_crop_and_resize_rgb(const uint16_t *p_input,
								int8_t *p_output,
								uint16_t in_width,
								uint16_t in_height,
								uint16_t out_width,
								uint16_t out_height)
{
	/* Crop image to center square */
	const uint32_t crop_offset = (in_width - in_height) / 2;

	for (uint32_t y = 0; y < out_height; y++)
	{
		/* Nearest neighbor row mapping */
		uint32_t src_y = (in_height * y) / out_height;
		const uint16_t *p_row = p_input + (src_y * in_width) + crop_offset;

		for (uint32_t x = 0; x < out_width; x++)
		{
			/* Nearest neighbor column mapping within cropped square */
			uint32_t src_x = (in_height * x) / out_width;
			uint16_t pixel = p_row[src_x];

			/* Extract RGB565 components and scale to 0-255 */
			uint8_t r = (uint8_t)(((pixel >> 11) & 0x1F) * 255 / 31);
			uint8_t g = (uint8_t)(((pixel >> 5) & 0x3F) * 255 / 63);
			uint8_t b = (uint8_t)((pixel & 0x1F) * 255 / 31);

			/* Convert uint8 [0,255] to int8 [-128,127] for model input */
			*p_output++ = (int8_t)((int)r - 128);
			*p_output++ = (int8_t)((int)g - 128);
			*p_output++ = (int8_t)((int)b - 128);
		}
	}
}

/******************************************************************************
 * Main entry point
 ******************************************************************************/
void hal_entry(void)
{
    fsp_err_t err;
    int8_t* input_ptr;
    int8_t* output1_ptr;
    int8_t* output2_ptr;
    uint32_t preprocess_time_us;
    uint32_t inference_time_us;
    uint32_t postprocess_time_us;
    uint32_t frame_count = 0;

    /* Initialize debugging terminal */
    TERM_INIT();
    APP_PRINT("Person Detection (Camera)\r\n");

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
	input_ptr = GetModelInputPtr_image_input();
	output1_ptr = GetModelOutputPtr_Identity_70275();      /* grid 1 */
	output2_ptr = GetModelOutputPtr_Identity_1_70284();    /* grid 2 */

	APP_PRINT("Buffer addresses:\r\n");
	APP_PRINT("  input:   %p\r\n", (void*)input_ptr);
	APP_PRINT("  output1: %p (%dx%d grid)\r\n", (void*)output1_ptr, GRID_1_SIZE, GRID_1_SIZE);
	APP_PRINT("  output2: %p (%dx%d grid)\r\n", (void*)output2_ptr, GRID_2_SIZE, GRID_2_SIZE);

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
    APP_PRINT("Running continuous person detection...\r\n");

    /* Main loop — display every frame, run inference periodically */
    while (1)
    {
        /* Poll for camera ready flag */
        if (g_camera_frame_ready)
        {
            g_camera_frame_ready = false;

            /* Post processing for camera image capture. After this completes,
             * image is stored in camera_capture_image_rgb565[].
             */
            camera_capture_post_process();

            /* Run inference on every Nth frame */
            if (++frame_count >= INFERENCE_INTERVAL)
            {
                frame_count = 0;

                /* Invalidate cache to ensure CPU sees latest data from DMA */
                SCB_InvalidateDCache_by_Addr((uint32_t *)camera_capture_image_rgb565,
                                             (int32_t)camera_capture_image_rgb565_size);

                /* Preprocess: crop to square, resize to 320x320, convert to RGB int8 */
                preprocess_time_us = micros();
                image_crop_and_resize_rgb((const uint16_t *)camera_capture_image_rgb565,
                              input_ptr,
                              CAMERA_CAPTURE_IMAGE_WIDTH,
                              CAMERA_CAPTURE_IMAGE_HEIGHT,
                              IMG_WIDTH,
                              IMG_HEIGHT);
                preprocess_time_us = micros() - preprocess_time_us;

                /* Run inference */
                inference_time_us = micros();
                RunModel(true);
                inference_time_us = micros() - inference_time_us;

                /* Decode detections from both grids */
                detection_t detections[MAX_DETECTIONS * 2];

                postprocess_time_us = micros();
                int count = 0;
                count += decode_yolo_output_int8(
                    output1_ptr, GRID_1_SIZE,
                    IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES,
                    OUTPUT_GRID1_SCALE, OUTPUT_GRID1_ZP,
                    anchors_grid1, NUM_ANCHORS,
                    CONF_THRESHOLD,
                    &detections[count], MAX_DETECTIONS);

                int count2 = decode_yolo_output_int8(
                    output2_ptr, GRID_2_SIZE,
                    IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES,
                    OUTPUT_GRID2_SCALE, OUTPUT_GRID2_ZP,
                    anchors_grid2, NUM_ANCHORS,
                    CONF_THRESHOLD,
                    &detections[count], MAX_DETECTIONS);
                count += count2;

                /* Apply NMS */
                count = apply_nms(detections, count, IOU_THRESHOLD);
                postprocess_time_us = micros() - postprocess_time_us;

                /* Store detections for display */
                g_detection_count = count;
                for (int i = 0; i < count; i++) {
                    g_detections[i] = detections[i];
                }

                /* Print results */
#if SERIAL_PRINT_RESULTS
                APP_PRINT("Preprocess: %u us | Inference: %u us | Postprocess: %u us\r\n",
                          preprocess_time_us, inference_time_us, postprocess_time_us);
                if (count > 0) {
                    APP_PRINT("Detected %d person(s)\r\n", count);
                    for (int i = 0; i < count; i++) {
                        APP_PRINT("  [%d] (%.0f,%.0f)-(%.0f,%.0f) %.2f\r\n",
                              i,
                              detections[i].x1, detections[i].y1,
                              detections[i].x2, detections[i].y2,
                              detections[i].confidence);
                    }
                }
#endif
            }

            /* Always update display */
            while (!g_display_vsync_ready);
            g_display_vsync_ready = false;

            graphics_start_frame();
            display_camera_image();

            /* Draw bounding boxes over the camera image */
            if (g_detection_count > 0)
            {
                d2_selectrendermode(d2_handle, d2_rm_outline);
                d2_outlinewidth(d2_handle, (d2_width)(BOX_LINE_WIDTH << 4));
                d2_setcolor(d2_handle, 0, BOX_COLOR);

                for (int i = 0; i < g_detection_count; i++)
                {
                    d2_s32 x = (d2_s32)(g_detections[i].x1 * DISPLAY_SCALE);
                    d2_s32 y = (d2_s32)(g_detections[i].y1 * DISPLAY_SCALE);
                    d2_s32 w = (d2_s32)((g_detections[i].x2 - g_detections[i].x1) * DISPLAY_SCALE);
                    d2_s32 h = (d2_s32)((g_detections[i].y2 - g_detections[i].y1) * DISPLAY_SCALE);

                    d2_renderbox(d2_handle,
                                 (d2_point)(x << 4), (d2_point)(y << 4),
                                 (d2_width)(w << 4), (d2_width)(h << 4));
                }
            }

            graphics_end_frame();
        }
    }
}
