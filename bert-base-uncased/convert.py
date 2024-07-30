import onnx
from onnx import helper, TensorProto

# Load your ONNX model
model = onnx.load("./model_repository/bert/1/model.onnx")

for input in model.graph.input:
    print(input.name, input.type.tensor_type.shape)

# Convert input data types to int32
for input in model.graph.input:
    if input.type.tensor_type.elem_type == TensorProto.INT64:
        input.type.tensor_type.elem_type = TensorProto.INT32

# Save the modified model
# onnx.save(model, "model_int32.onnx")

