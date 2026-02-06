import sys
from rknn.api import RKNN

# CONFIG PATHS
ONNX_MODEL = 'runs/detect/aiv6s2/weights/best.onnx'  # Your Nano model
RKNN_MODEL = 'aiv6s2.rknn'

def export_rknn():
    rknn = RKNN(verbose=True)

    # 1. Config for RK3588
    # target_platform="rk3588" is required
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')

    # 2. Load ONNX
    print(f'--> Loading {ONNX_MODEL}')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('❌ Load model failed!')
        exit(ret)

    # 3. Build (Compile for NPU)
    print('--> Building model (This takes 1-2 mins)...')
    # do_quantization=False keeps it in FP16 (Higher accuracy, usually fast enough)
    ret = rknn.build(do_quantization=False) 
    if ret != 0:
        print('❌ Build model failed!')
        exit(ret)

    # 4. Export .rknn file
    print(f'--> Exporting to {RKNN_MODEL}')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('❌ Export rknn model failed!')
        exit(ret)

    print('✅ DONE! Model ready for upload.')

if __name__ == '__main__':
    export_rknn()