import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="/home/verse/Python/GD_Net/mcunet_model/person-det.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tensor_details = interpreter.get_tensor_details()

print("== Input ==")
for inp in input_details:
    print(f"Name: {inp['name']}, Shape: {inp['shape']}, Type: {inp['dtype']}")

print("\n== Output ==")
for out in output_details:
    print(f"Name: {out['name']}, Shape: {out['shape']}, Type: {out['dtype']}")

print("\n== All Tensors ==")
for t in tensor_details[:20]:  # 只显示前20个
    print(f"{t['index']}: {t['name']} - {t['shape']} - {t['dtype']}")
