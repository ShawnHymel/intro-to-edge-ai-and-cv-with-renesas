/* Generated libraries */
#include "hal_data.h"

/* Custom libraries */
#include "camera_layer/camera_layer.h"
#include "common_utils.h"
#include "display_layer/display_screen.h"
#include "inference/model.h"
#include "utils/utils.h"

#include <math.h>

/******************************************************************************
 * Settings
 ******************************************************************************/

/* Target image dimensions */
#define IMG_WIDTH 192
#define IMG_HEIGHT 192
#define IMG_CHANNELS 3

/* YOLO detection settings */
#define NUM_CLASSES 1
#define NUM_ANCHORS 3
#define NUM_ATTRS 6     /* tx, ty, tw, th, obj, cls */
#define CONF_THRESHOLD 0.25f
#define IOU_THRESHOLD 0.45f
#define MAX_DETECTIONS 10

/* Grid sizes */
#define GRID_1_SIZE 6   /* Coarse grid for larger objects */
#define GRID_2_SIZE 12  /* Fine grid for smaller objects */

/* Quantization parameters */
#define INPUT_SCALE 0.0039215689f
#define INPUT_ZP -128
#define OUTPUT_6x6_SCALE 0.0758222416f
#define OUTPUT_6x6_ZP -11
#define OUTPUT_12x12_SCALE 0.0748264119f
#define OUTPUT_12x12_ZP -12

/* Anchor masks */
/* Grid 1 (6x6) uses anchors: [3, 4, 5] */
/* Grid 2 (12x12) uses anchors: [0, 1, 2] */

/* Anchors for grid 1 (6x6 - larger objects) */
static const float anchors_grid1[NUM_ANCHORS][2] = {
    {115.0f, 73.0f},  /* anchor 3 */
    {119.0f, 199.0f},  /* anchor 4 */
    {242.0f, 238.0f},  /* anchor 5 */
};

/* Anchors for grid 2 (12x12 - smaller objects) */
static const float anchors_grid2[NUM_ANCHORS][2] = {
    {12.0f, 18.0f},  /* anchor 0 */
    {37.0f, 49.0f},  /* anchor 1 */
    {52.0f, 132.0f},  /* anchor 2 */
};

/* Labels */
const char* class_names[] = {
    "person",
};

/******************************************************************************
 * Detection structure and helper functions
 ******************************************************************************/

typedef struct {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    int valid;
} detection_t;

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Dequantize int8 value to float
 */
static inline float dequantize(int8_t val, float scale, int zero_point)
{
    return ((float)val - (float)zero_point) * scale;
}

/**
 * Decode YOLO output tensor (int8, NHWC format) into detections
 *
 * NHWC layout: output[y][x][channel]
 * Flattened index: y * grid_size * 18 + x * 18 + channel
 *
 * Per anchor: 6 channels (tx, ty, tw, th, obj, cls)
 * Anchor 0: channels 0-5, Anchor 1: channels 6-11, Anchor 2: channels 12-17
 */
static int decode_yolo_output_int8(
    const int8_t* output,
    int grid_size,
    float scale,
    int zero_point,
    const float anchors[][2],
    int num_anchors,
    float conf_threshold,
    detection_t* detections,
    int max_detections)
{
    int num_detections = 0;
    int stride = IMG_WIDTH / grid_size;
    int num_attrs = 5 + NUM_CLASSES;  /* 6: tx, ty, tw, th, obj, cls */
    int num_channels = num_anchors * num_attrs;  /* 18 */

    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            for (int a = 0; a < num_anchors; a++) {
                /* NHWC indexing */
                int spatial_base = (y * grid_size + x) * num_channels;
                int ch_offset = a * num_attrs;

                /* Dequantize values */
                float tx  = dequantize(output[spatial_base + ch_offset + 0], scale, zero_point);
                float ty  = dequantize(output[spatial_base + ch_offset + 1], scale, zero_point);
                float tw  = dequantize(output[spatial_base + ch_offset + 2], scale, zero_point);
                float th  = dequantize(output[spatial_base + ch_offset + 3], scale, zero_point);
                float obj = dequantize(output[spatial_base + ch_offset + 4], scale, zero_point);
                float cls = dequantize(output[spatial_base + ch_offset + 5], scale, zero_point);

                float obj_score = sigmoid(obj);
                float cls_score = sigmoid(cls);
                float confidence = obj_score * cls_score;

                if (confidence < conf_threshold) {
                    continue;
                }

                float bx = (sigmoid(tx) + (float)x) * stride;
                float by = (sigmoid(ty) + (float)y) * stride;
                float bw = expf(tw) * anchors[a][0];
                float bh = expf(th) * anchors[a][1];

                float x1 = bx - bw / 2.0f;
                float y1 = by - bh / 2.0f;
                float x2 = bx + bw / 2.0f;
                float y2 = by + bh / 2.0f;

                if (x1 < 0) x1 = 0;
                if (y1 < 0) y1 = 0;
                if (x2 > IMG_WIDTH) x2 = IMG_WIDTH;
                if (y2 > IMG_HEIGHT) y2 = IMG_HEIGHT;

                if (num_detections < max_detections) {
                    detections[num_detections].x1 = x1;
                    detections[num_detections].y1 = y1;
                    detections[num_detections].x2 = x2;
                    detections[num_detections].y2 = y2;
                    detections[num_detections].confidence = confidence;
                    detections[num_detections].class_id = 0;
                    detections[num_detections].valid = 1;
                    num_detections++;
                }
            }
        }
    }

    return num_detections;
}

static float compute_iou(const detection_t* a, const detection_t* b)
{
    float x1 = (a->x1 > b->x1) ? a->x1 : b->x1;
    float y1 = (a->y1 > b->y1) ? a->y1 : b->y1;
    float x2 = (a->x2 < b->x2) ? a->x2 : b->x2;
    float y2 = (a->y2 < b->y2) ? a->y2 : b->y2;

    float inter_w = (x2 - x1 > 0) ? (x2 - x1) : 0;
    float inter_h = (y2 - y1 > 0) ? (y2 - y1) : 0;
    float inter_area = inter_w * inter_h;

    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0) return 0;
    return inter_area / union_area;
}

static int apply_nms(detection_t* detections, int count, float iou_threshold)
{
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (detections[j].confidence > detections[i].confidence) {
                detection_t temp = detections[i];
                detections[i] = detections[j];
                detections[j] = temp;
            }
        }
    }

    for (int i = 0; i < count; i++) {
        if (!detections[i].valid) continue;
        for (int j = i + 1; j < count; j++) {
            if (!detections[j].valid) continue;
            float iou = compute_iou(&detections[i], &detections[j]);
            if (iou > iou_threshold) {
                detections[j].valid = 0;
            }
        }
    }

    int valid_count = 0;
    for (int i = 0; i < count; i++) {
        if (detections[i].valid) {
            if (i != valid_count) {
                detections[valid_count] = detections[i];
            }
            valid_count++;
        }
    }

    return valid_count;
}

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
    bool pressed = false;
    bool inference_pending = false;
    uint32_t preprocess_time_us;
    uint32_t inference_time_us;
    uint32_t postprocess_time_us;

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
	output1_ptr = GetModelOutputPtr_StatefulPartitionedCall_1_0_70275();  /* 6x6 grid */
	output2_ptr = GetModelOutputPtr_StatefulPartitionedCall_1_1_70284();  /* 12x12 grid */

	APP_PRINT("Buffer addresses:\r\n");
	APP_PRINT("  input:   %p\r\n", (void*)input_ptr);
	APP_PRINT("  output1: %p (6x6)\r\n", (void*)output1_ptr);
	APP_PRINT("  output2: %p (12x12)\r\n", (void*)output2_ptr);

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
    APP_PRINT("Press SW1 to capture and run person detection\r\n");

    /* Main loop */
    while (1)
    {
    	/* See if SW1 button has been pressed (with debounce logic) */
		err = check_button_sw1(&pressed);
		if (FSP_SUCCESS != err) {
			APP_PRINT("Error: Could not check SW1 button: %d\r\n", err);
		}

		/* If pressed, set flag to run inference on next frame */
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

            /* If button has been pressed, perform inference */
            if (inference_pending)
            {
            	inference_pending = false;

            	/* Invalidate cache to ensure CPU sees latest data from DMA */
				SCB_InvalidateDCache_by_Addr((uint32_t *)camera_capture_image_rgb565,
											 (int32_t)camera_capture_image_rgb565_size);

				/* Preprocess: crop to square, resize to 192x192, convert to RGB int8 */
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
					OUTPUT_6x6_SCALE, OUTPUT_6x6_ZP,
					anchors_grid1, NUM_ANCHORS,
					CONF_THRESHOLD,
					&detections[count], MAX_DETECTIONS);


				int count2 = decode_yolo_output_int8(
					output2_ptr, GRID_2_SIZE,
					OUTPUT_12x12_SCALE, OUTPUT_12x12_ZP,
					anchors_grid2, NUM_ANCHORS,
					CONF_THRESHOLD,
					&detections[count], MAX_DETECTIONS);
				count += count2;

				/* Apply NMS */
				count = apply_nms(detections, count, IOU_THRESHOLD);
				postprocess_time_us = micros() - postprocess_time_us;

				/* Print timing */
				APP_PRINT("\r\n=== Person Detection Results ===\r\n");
				APP_PRINT("  Preprocess time: %u us\r\n", preprocess_time_us);
				APP_PRINT("  Inference time:  %u us\r\n", inference_time_us);
				APP_PRINT("  Postprocess time:  %u us\r\n", postprocess_time_us);

				/* Print final detections */
				APP_PRINT("  After NMS: %d\r\n", count);
				for (int i = 0; i < count; i++) {
					APP_PRINT("  [%d] (%.1f, %.1f)-(%.1f, %.1f) conf=%.4f\r\n",
						  i,
						  detections[i].x1, detections[i].y1,
						  detections[i].x2, detections[i].y2,
						  detections[i].confidence);
				}
				if (count == 0) {
					APP_PRINT("  No persons detected\r\n");
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
