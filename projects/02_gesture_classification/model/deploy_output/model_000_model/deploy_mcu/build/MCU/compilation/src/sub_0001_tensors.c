#include "sub_0001_tensors.h"

const TensorInfo sub_0001_tensors[] = {
  { "_split_1_command_stream", 0, 532, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 40080, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 672, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 672, "FAST_SCRATCH", 0x0 },
  { "input_10018", 4, 600, "INPUT_TENSOR", 0x0 },
  { "output_70010_10026", 5, 5, "OUTPUT_TENSOR", 0x20 },
};

const size_t sub_0001_tensors_count = sizeof(sub_0001_tensors) / sizeof(sub_0001_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0001_address_input_10018 = 0x0;
const uint32_t sub_0001_address_output_70010_10026 = 0x20;

