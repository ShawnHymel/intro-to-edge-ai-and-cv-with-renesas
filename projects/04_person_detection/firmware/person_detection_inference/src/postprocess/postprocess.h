#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <stdint.h>

/* Struct to hold detection information */
typedef struct {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    int valid;
} detection_t;

/* Function prototypes */
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
    int max_detections);
int apply_nms(detection_t* detections, int count, float iou_threshold);

#endif /* POSTPROCESS_H */
