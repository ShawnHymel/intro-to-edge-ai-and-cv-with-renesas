#include "sub_0001_tensors.h"

const TensorInfo sub_0001_tensors[] = {
  { "_split_1_command_stream", 0, 716, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 1984, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 46080, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 46080, "FAST_SCRATCH", 0x0 },
  { "input_70022_10052", 4, 2304, "INPUT_TENSOR", 0x9000 },
  { "max_pool2d_1_70012_10030", 5, 2304, "OUTPUT_TENSOR", 0x2400 },
};

const size_t sub_0001_tensors_count = sizeof(sub_0001_tensors) / sizeof(sub_0001_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0001_address_input_70022_10052 = 0x9000;
const uint32_t sub_0001_address_max_pool2d_1_70012_10030 = 0x2400;

