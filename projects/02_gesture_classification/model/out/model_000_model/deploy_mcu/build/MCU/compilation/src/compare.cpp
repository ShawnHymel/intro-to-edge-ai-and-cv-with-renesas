/*
 * This file is developed by EdgeCortix Inc. to be used with certain Renesas Electronics Hardware only.
 *
 * Copyright © 2025 EdgeCortix Inc. Licensed to Renesas Electronics Corporation with the
 * right to sublicense under the Apache License, Version 2.0.
 *
 * This file also includes source code originally developed by the Renesas Electronics Corporation.
 * The Renesas disclaimer below applies to any Renesas-originated portions for usage of the code.
 *
 * The Renesas Electronics Corporation
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No
 * other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all
 * applicable laws, including copyright laws.
 * THIS SOFTWARE IS PROVIDED 'AS IS' AND RENESAS MAKES NO WARRANTIES REGARDING
 * THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM
 * EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES
 * SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS
 * SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 * Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of
 * this software. By using this software, you agree to the additional terms and conditions found by accessing the
 * following link:
 * http://www.renesas.com/disclaimer
 *
 * Changed from original python code to C source code.
 * Copyright (C) 2017 Renesas Electronics Corporation. All rights reserved.
 *
 * This file also includes source codes originally developed by the TensorFlow Authors which were distributed under the following conditions.
 *
 * The TensorFlow Authors
 * Copyright 2023 The Apache Software Foundation
 *
 * This product includes software developed at
 * The Apache Software Foundation (http://www.apache.org/).
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>
#include <cmath>

#define INT_EPSILON 2
#define FLOAT_EPSILON 0.00001

extern "C" {
#include "compute_sub_0002.h"
}

template<typename T>
std::vector<T> load_file(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  std::streampos file_size;
  file.seekg(0, std::ios::end);
  file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<T> vec;
  vec.reserve(file_size);
  vec.resize(file_size / sizeof(T));
  file.read(reinterpret_cast<char*>(vec.data()), file_size);
  return (vec);
}

template<typename T>
void print_samples(const std::vector<T>& ref, const std::vector<T>& got, int nsamples) {
  std::cout << "  All matches: " << std::endl;
  for (int i = 0; i < ref.size(); i++) {
    if constexpr (std::is_integral_v<T>) {
      std::cout << "  Sample " << i << ": Ref: " << static_cast<int32_t>(ref.at(i)) << " - Got: " << static_cast<int32_t>(got.at(i)) << std::endl;
    } else {
      std::cout << "  Sample " << i << ": Ref: " << ref.at(i) << " - Got: " << got.at(i) << std::endl;
    }
    if (i > nsamples) {
      break;
    }
  }
}

template<typename T>
bool compare(int output_id, std::vector<T> const & lhs, std::vector<T> const & rhs, float epsilon, std::enable_if_t<std::is_integral_v<T>, void*> = 0) {
  epsilon = INT_EPSILON;
  std::cout << "Comparing output #" << output_id << "..." << std::endl;
  if (lhs.size() != rhs.size()) {
    std::cout << "  Error: vector sizes mismatch" << std::endl;
    return false;
  }
  bool all_good = true;
  for (int i = 0; i < rhs.size(); i++) {
    int got = rhs.at(i);
    int ref = lhs.at(i);
    if (abs(got - ref) >= epsilon) {
      std::cout << "  Mismatch (int): Ref: " << ref << " - Got: " << got << " - Diff: " << abs(ref - got) << std::endl;
      all_good = false;
    }
  }
  if (all_good) {
    print_samples<T>(lhs, rhs, 8);
  }
  return all_good;
} 

bool compare(int output_id, std::vector<float> const & lhs, std::vector<float> const & rhs, float epsilon) {
  epsilon = FLOAT_EPSILON;
  std::cout << "Comparing output #" << output_id << "..." << std::endl;
  if (lhs.size() != rhs.size()) {
    std::cout << "  Error: vector sizes mismatch" << std::endl;
    return false;
  }
  bool all_good = true;
  for (int i = 0; i < rhs.size(); i++) {
    float got = rhs.at(i);
    float ref = lhs.at(i);
    if (std::isnan(got) && std::isnan(ref)) {
      std::cout << "  NOTE:  Both NAN: Got: " << got << " Ref: " << ref << std::endl;
    } else if (std::isnan(got) && !std::isnan(ref)) {
      all_good = false;
      std::cout << "  Mismatch:  Got NAN: " << got << " Ref: " << ref << std::endl;
    } else if (!std::isnan(got) && std::isnan(ref)) {
      all_good = false;
      std::cout << "  Mismatch:  Got: " << got << " Ref NAN: " << ref << std::endl;
    } else if (std::isinf(got) && std::isinf(ref)) {
      if (got == INFINITY && ref == INFINITY) {
        std::cout << "  NOTE:  Both +INF Got: " << got << " Ref: " << ref << std::endl;
      } else if (got == -INFINITY && ref == -INFINITY) {
        std::cout << "  NOTE:  Both -INF Got: " << got << " Ref: " << ref << std::endl;
      } else if (got == INFINITY && ref == -INFINITY) {
        all_good = false;
        std::cout << "  Mismatch: Got +INF: " << got << " Ref -INF: " << ref << std::endl;
      } else if (got == -INFINITY && ref == INFINITY) {
        all_good = false;
        std::cout << "  Mismatch: Got -INF: " << got << " Ref +INF: " << ref << std::endl;
      }
    } else if (fabs(got - ref) >= epsilon) {
      std::cout << "  Mismatch: Ref: " << ref << " - Got: " << got << " - Diff: " << fabs(ref - got) << std::endl;
      all_good = false;
    }
  }
  if (all_good) {
    print_samples<float>(lhs, rhs, 8);
  }
  return all_good;
} 


int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Expected reference data directory" << std::endl;
    return -1;
  }

  // main storage
  std::vector<uint8_t> main_storage(kBufferSize_sub_0002);

  // inputs
  
  auto ref_input_0 = load_file<int8_t>(std::string(argv[1]) + "/ref_in_0.bin");
  

  // outputs
  
  auto ref_output_0 = load_file<float>(std::string(argv[1]) + "/ref_out_0.bin");
  

  
  std::vector<float> result_0(5);
  

  compute_sub_0002(
    // buffer for intermediate results
    main_storage.data(),

    // inputs
    
    ref_input_0.data(), // int8_t, 1,5
    

    // outputs
    
    result_0.data() // float, 1,5
    
  );

  
  compare(0, ref_output_0, result_0, 0); // float, 5 elements
  

  std::cout << "Success" << std::endl;
  return 0;
}
