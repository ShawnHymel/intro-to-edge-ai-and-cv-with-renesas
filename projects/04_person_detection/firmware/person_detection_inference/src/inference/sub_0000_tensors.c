#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 2, 10976, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 3, 420000, "MODEL", 0xffffffff },
  { "_split_1_scratch", 4, 1228800, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 5, 1228800, "FAST_SCRATCH", 0x0 },
  { "image_input", 6, 307200, "INPUT_TENSOR", 0x64000 },
  { "Identity_1_70284", 0, 7200, "OUTPUT_TENSOR", 0xed80 },
  { "Identity_70275", 1, 1800, "OUTPUT_TENSOR", 0x2580 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_image_input = 0x64000;
const uint32_t sub_0000_address_Identity_1_70284 = 0xed80;
const uint32_t sub_0000_address_Identity_70275 = 0x2580;

