#include "sub_0003_tensors.h"

const TensorInfo sub_0003_tensors[] = {
  { "_split_1_command_stream", 0, 328, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 10112, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 2320, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 2320, "FAST_SCRATCH", 0x0 },
  { "max_pool2d_1_70012_70023_10056", 4, 2304, "INPUT_TENSOR", 0x0 },
  { "output_70014_10042", 5, 4, "OUTPUT_TENSOR", 0x900 },
};

const size_t sub_0003_tensors_count = sizeof(sub_0003_tensors) / sizeof(sub_0003_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0003_address_max_pool2d_1_70012_70023_10056 = 0x0;
const uint32_t sub_0003_address_output_70014_10042 = 0x900;

