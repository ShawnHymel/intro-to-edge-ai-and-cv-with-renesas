#include "postprocess.h"
#include <math.h>

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float dequantize(int8_t val, float scale, int zero_point)
{
    return ((float)val - (float)zero_point) * scale;
}

/**
 * Decode a YOLO output tensor (int8, NHWC format) into detections.
 *
 * Per anchor: (5 + num_classes) channels — tx, ty, tw, th, obj, cls0, [cls1, ...]
 * Returns the number of detections written.
 */
int decode_yolo_output_int8(
    const int8_t* output,
    int grid_size,
    int img_width,
    int img_height,
    int num_classes,
    float scale,
    int zero_point,
    const float anchors[][2],
    int num_anchors,
    float conf_threshold,
    detection_t* detections,
    int max_detections)
{
    int num_detections = 0;
    int stride = img_width / grid_size;
    int num_attrs = 5 + num_classes;
    int num_channels = num_anchors * num_attrs;

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
                if (x2 > img_width) x2 = img_width;
                if (y2 > img_height) y2 = img_height;

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

/**
 * Apply non-maximum suppression in-place.
 * Returns the number of surviving detections (compacted to the front of the array).
 */
int apply_nms(detection_t* detections, int count, float iou_threshold)
{
    /* Sort by confidence (descending) */
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (detections[j].confidence > detections[i].confidence) {
                detection_t temp = detections[i];
                detections[i] = detections[j];
                detections[j] = temp;
            }
        }
    }

    /* Suppress overlapping boxes */
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

    /* Compact valid detections to front */
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
