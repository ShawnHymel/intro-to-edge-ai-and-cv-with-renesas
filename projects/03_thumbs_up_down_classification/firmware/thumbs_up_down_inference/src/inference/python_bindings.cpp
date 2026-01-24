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

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

extern "C" {
#include "compute_sub_0004.h"
}

namespace py = pybind11;

bool check_dims(int ndim, const py::ssize_t* shape, const py::ssize_t* expected_shape) {
  for (int i = 0; i < ndim; ++i) {
    if (shape[i] !=  expected_shape[i]) {
      std::stringstream ss;
      ss << "Wrong dimension size at position " << i << ": expected: ";
      ss << expected_shape[i] << " but got size: " << shape[i];
      return false;
    }
  }
  return true;
}

py::object wrapper(
  
  const py::array_t<int8_t, py::array::c_style>& input_0 
  
) {
  
  py::array_t<float, py::array::c_style> output_0 ({ 1,4 });
  
  
  const py::ssize_t expected_in_shape_0[] = { 1,4 };
  
  
  const py::ssize_t expected_out_shape_0[] = { 1,4 };
  

  
  check_dims(input_0.ndim(), input_0.shape(), expected_in_shape_0);
  

  
  const auto* in_ptr_0 = static_cast<const int8_t*>(input_0.request().ptr);
  
  
  auto* out_ptr_0 = static_cast<float*>(output_0.request().ptr);
  

  std::vector<uint8_t> main_storage(kBufferSize_sub_0004);

  compute_sub_0004(
    // buffer for intermediate results
    main_storage.data(),

    // inputs
    
    in_ptr_0,
    
    // outputs
    
    out_ptr_0 
    
  );
  return py::make_tuple(
  
    py::array_t<float, py::array::c_style>(output_0) 
  
  );
}

PYBIND11_MODULE(py_compute, m) {
  m.doc() = "Compute Python binding";
  m.def("compute", &wrapper,
    
    py::arg("input_0").noconvert(true).none(false),
    

    "Generated compute_sub_0004 function"
  );

  m.def("get_signature",
    [](){}, 
    "Get the input sizes and output sizes and dtype"
  );
}
