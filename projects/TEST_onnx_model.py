import onnx
from onnx import numpy_helper

model = onnx.load("/projects/04_person_detection/model/model.onnx")

# Build initializer lookup
inits = {init.name: init for init in model.graph.initializer}

# Check all Conv nodes for unusual weight shapes
for i, node in enumerate(model.graph.node):
    if node.op_type == "Conv":
        weight_name = node.input[1] if len(node.input) > 1 else None
        if weight_name and weight_name in inits:
            arr = numpy_helper.to_array(inits[weight_name])
            group = 1
            for attr in node.attribute:
                if attr.name == "group":
                    group = attr.i
            if arr.ndim != 4 or arr.shape[1] == 1:
                print(f"Node {i}: inputs={list(node.input)[:2]}, "
                      f"weight_shape={arr.shape}, group={group}")