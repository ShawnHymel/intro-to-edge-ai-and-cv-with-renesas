/* Custom libraries */
#include "common_utils.h"
#include "hal_data.h"
#include "inference/model.h"
#include "inference/sub_0000_invoke.h"
#include "inference/sub_0000_tensors.h"

#include "test_sample.h"
#include <math.h>

/* Target image dimensions */
#define IMG_WIDTH 320
#define IMG_HEIGHT 320
#define IMG_CHANNELS 3

/* YOLO detection settings */
#define NUM_CLASSES 1
#define NUM_ANCHORS 3
#define NUM_ATTRS 6     /* tx, ty, tw, th, obj, cls */
#define CONF_THRESHOLD 0.25f
#define IOU_THRESHOLD 0.45f
#define MAX_DETECTIONS 10

/* Grid sizes */
#define GRID_1_SIZE 10   /* Coarse grid for larger objects */
#define GRID_2_SIZE 20  /* Fine grid for smaller objects */

/* Quantization parameters */
#define INPUT_SCALE 0.0039215689f
#define INPUT_ZP -128
#define OUTPUT_GRID1_SCALE 0.0829966888f
#define OUTPUT_GRID1_ZP -14
#define OUTPUT_GRID2_SCALE 0.0748412311f
#define OUTPUT_GRID2_ZP -8

/* Anchor masks */
/* Grid 1 (10x10) uses anchors: [3, 4, 5] */
/* Grid 2 (20x20) uses anchors: [0, 1, 2] */

/* Anchors for grid 1 (10x10 - larger objects) */
static const float anchors_grid1[NUM_ANCHORS][2] = {
    {115.0f, 73.0f},  /* anchor 3 */
    {119.0f, 199.0f},  /* anchor 4 */
    {242.0f, 238.0f},  /* anchor 5 */
};

/* Anchors for grid 2 (20x20 - smaller objects) */
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
 * Main entry point
 ******************************************************************************/

void hal_entry(void)
{
    fsp_err_t err;

    TERM_INIT();
    APP_PRINT("\r\n\r\n\r\n~~~ Person Detection Static Inference Test ~~~\r\n");
    APP_PRINT("Model: yolo-fastest_320_person (INT8 I/O)\r\n");

    rm_ethosu_cfg_t ethosu_cfg = g_rm_ethosu0_cfg;
    ethosu_cfg.secure_enable = 1;
    ethosu_cfg.privilege_enable = 1;

    err = RM_ETHOSU_Open(&g_rm_ethosu0_ctrl, &ethosu_cfg);
    if (FSP_SUCCESS != err) {
        APP_PRINT("  ERROR: RM_ETHOSU_Open failed with code: %d\r\n", err);
        while(1);
    }
    APP_PRINT("Ethos-U NPU successfully initialized!\r\n");

    /* INT8 pointers — update function names to match your generated headers */
    int8_t* input_ptr = GetModelInputPtr_image_input();
    int8_t* output1_ptr = GetModelOutputPtr_Identity_70275();      /* grid 1 */
    int8_t* output2_ptr = GetModelOutputPtr_Identity_1_70284();    /* grid 2 */

    APP_PRINT("\nBuffer addresses:\r\n");
    APP_PRINT("  input_ptr:   %p\r\n", (void*)input_ptr);
    APP_PRINT("  output1_ptr: %p\r\n", (void*)output1_ptr);
    APP_PRINT("  output2_ptr: %p\r\n", (void*)output2_ptr);

    /*
     * Input conversion: uint8 pixel [0,255] -> int8 [-128, 127]
     * Since scale ≈ 1/255 and zero_point = -128:
     *   int8_val = round(pixel/255.0 / (1/255.0)) + (-128) = pixel - 128
     *
     * Model expects NHWC: [1, 320, 320, 3]
     * test_input[] is already HWC uint8, so layout matches — just offset values
     */
    APP_PRINT("\nCopying test input (HWC uint8 -> HWC int8)...\r\n");

    int input_size = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;

    for (int i = 0; i < input_size; i++) {
        input_ptr[i] = (int8_t)((int)test_input[i] - 128);
    }

    APP_PRINT("Input size: %d bytes (HWC format)\r\n", input_size);

    /* Verify input - show first 12 values */
    APP_PRINT("First 12 input values (R,G,B,R,G,B,...):\r\n");
    for (int i = 0; i < 12; i++) {
        float dequant = ((float)input_ptr[i] - (float)INPUT_ZP) * INPUT_SCALE;
        APP_PRINT("  [%d] int8=%d -> float=%.4f (orig uint8=%d)\r\n",
                  i, input_ptr[i], dequant, test_input[i]);
    }

    /* Run NPU inference */
    APP_PRINT("\n=== Running NPU Inference ===\r\n");
    RunModel(true);
    APP_PRINT("NPU inference complete.\r\n");

    /* Debug: Print raw int8 and dequantized values */
    int num_channels = NUM_ANCHORS * NUM_ATTRS;  /* 18 */
    int grid1_total = GRID_1_SIZE * GRID_1_SIZE * num_channels;
    int grid2_total = GRID_2_SIZE * GRID_2_SIZE * num_channels;

    APP_PRINT("\n=== Raw INT8 Output ===\r\n");

    APP_PRINT("Output1 (%dx%d) first 18 (one spatial position, all channels):\r\n",
              GRID_1_SIZE, GRID_1_SIZE);
    for (int i = 0; i < 18; i++) {
        float val = dequantize(output1_ptr[i], OUTPUT_GRID1_SCALE, OUTPUT_GRID1_ZP);
        APP_PRINT("  [%d] int8=%d -> float=%.4f\r\n", i, output1_ptr[i], val);
    }

    APP_PRINT("\nOutput2 (%dx%d) first 18 (one spatial position, all channels):\r\n",
              GRID_2_SIZE, GRID_2_SIZE);
    for (int i = 0; i < 18; i++) {
        float val = dequantize(output2_ptr[i], OUTPUT_GRID2_SCALE, OUTPUT_GRID2_ZP);
        APP_PRINT("  [%d] int8=%d -> float=%.4f\r\n", i, output2_ptr[i], val);
    }

    /* Sum all output values to compare with TFLite Python */
    APP_PRINT("\n=== Output Buffer Check ===\r\n");
    float sum1 = 0, sum2 = 0;
    for (int i = 0; i < grid1_total; i++) {
        sum1 += dequantize(output1_ptr[i], OUTPUT_GRID1_SCALE, OUTPUT_GRID1_ZP);
    }
    for (int i = 0; i < grid2_total; i++) {
        sum2 += dequantize(output2_ptr[i], OUTPUT_GRID2_SCALE, OUTPUT_GRID2_ZP);
    }
    APP_PRINT("Output1 (%dx%d) dequant sum: %.2f\r\n", GRID_1_SIZE, GRID_1_SIZE, sum1);
    APP_PRINT("Output2 (%dx%d) dequant sum: %.2f\r\n", GRID_2_SIZE, GRID_2_SIZE, sum2);

    /* Decode detections */
    detection_t detections[MAX_DETECTIONS * 2];
    int count = 0;

    count += decode_yolo_output_int8(
        output1_ptr, GRID_1_SIZE,
        OUTPUT_GRID1_SCALE, OUTPUT_GRID1_ZP,
        anchors_grid1, NUM_ANCHORS,
        CONF_THRESHOLD,
        &detections[count], MAX_DETECTIONS
    );
    APP_PRINT("\nDetections from grid 1 (%dx%d): %d\r\n", GRID_1_SIZE, GRID_1_SIZE, count);

    int count2 = decode_yolo_output_int8(
        output2_ptr, GRID_2_SIZE,
        OUTPUT_GRID2_SCALE, OUTPUT_GRID2_ZP,
        anchors_grid2, NUM_ANCHORS,
        CONF_THRESHOLD,
        &detections[count], MAX_DETECTIONS
    );
    APP_PRINT("Detections from grid 2 (%dx%d): %d\r\n", GRID_2_SIZE, GRID_2_SIZE, count2);
    count += count2;

    APP_PRINT("Total detections before NMS: %d\r\n", count);

    count = apply_nms(detections, count, IOU_THRESHOLD);

    APP_PRINT("\r\n=== Person Detection Results ===\r\n");
    APP_PRINT("Detections after NMS: %d\r\n", count);
    for (int i = 0; i < count; i++) {
        APP_PRINT("  Box %d: (%.1f, %.1f) to (%.1f, %.1f), conf=%.4f\r\n",
                  i,
                  detections[i].x1, detections[i].y1,
                  detections[i].x2, detections[i].y2,
                  detections[i].confidence);
    }

    APP_PRINT("\r\nTest Complete\r\n");

    while(1) {
        R_BSP_SoftwareDelay(1000, BSP_DELAY_UNITS_MILLISECONDS);
    }
}
