import tensorflow as tf

print("\n=== VERIFIKASI SEKTOR HARDWARE MAPAN ===")
print(f"TensorFlow Berhasil Dimuat (Versi: {tf.__version__})")

print("\n=== GPU CHECK ===")
gpu = tf.config.list_physical_devices('GPU')
print(f"Jumlah GPU Nvidia yang terdeteksi: {len(gpu)}")

if gpu:
    print(f"GPU yang terdeteksi: {gpu}")
    print("[SUKSES] GPU Nvidia terdeteksi. Tensorflow terkoneksi ke GPU lokal latopmu!")
else:
    print("[GAGAL] GPU tidak terdeteksi oleh Tensorflow. Sistem berjalan via CPU!")