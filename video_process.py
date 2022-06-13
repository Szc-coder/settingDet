import paddle.inference as paddleinf
import cv2
import numpy as np
import paddle.nn.functional
from source.keypoint_infer import KeyPointDetector, visualize
from source.preprocess import preprocess
from source.keypoint_preprocess import TopDownEvalAffine
from source.preprocess import Permute
from source.infer import create_inputs
from source.visualize import visualize_pose
from source.utils import shoulder_tilt, Head_Sitting


def load_model():
    config = paddleinf.Config('D:\\dev\\settingDet\\detModel\\model.pdmodel',
                              'D:\\dev\\settingDet\\detModel\\model.pdiparams')
    predictor = paddleinf.create_predictor(config)
    return predictor


def init_ops(detector):
    preprocess_ops = []
    for op_info in detector.pred_config.preprocess_infos:
        new_op_info = op_info.copy()
        op_type = new_op_info.pop('type')
        preprocess_ops.append(eval(op_type)(**new_op_info))
    return preprocess_ops


def predict(detector, frame, ops):
    input_im_lst = []
    input_im_info_lst = []
    im, im_info = preprocess(frame, ops)
    input_im_lst.append(im)
    input_im_info_lst.append(im_info)
    inputs = create_inputs(input_im_lst, input_im_info_lst)
    input_names = detector.predictor.get_input_names()

    for i in range(len(input_names)):
        input_tensor = detector.predictor.get_input_handle(input_names[i])
        input_tensor.copy_from_cpu(inputs[input_names[i]])

    result = detector.predict(repeats=1)
    result = detector.postprocess(inputs, result)
    return result


"""
COCO keypoint indexes:
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
"""


def get_state(keypoint):
    # keypoint [1,1,17,3]
    HT = Head_Sitting(keypoint[0][5][0:2], keypoint[0][6][0:2], keypoint[0][0][0:2])
    ST = shoulder_tilt(keypoint[0][5][0:2], keypoint[0][6][0:2])
    return HT, ST


def put_text(shower, text):
    cv2.putText(shower, text, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 5)


def main():
    if paddle.is_compiled_with_cuda():
        device = 'GPU'
    else:
        device = 'CPU'
    # Load model
    detector = KeyPointDetector(
        'detModel',
        device=device,
        run_mode='paddle',
        batch_size=1,
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        trt_calib_mode=False,
        cpu_threads=1,
        enable_mkldnn=False,
        threshold=0.5,
        use_dark=True
    )

    ops = init_ops(detector)

    # cv2 读取摄像头
    cap = cv2.VideoCapture(0)
    # 640 480

    while True:
        ret, frame = cap.read()
        # predict
        result = predict(detector, frame, ops)
        shower = visualize_pose(frame, result)
        # 计算夹角
        HS, ST = get_state(keypoint=result['keypoint'])
        if HS and ST:
            put_text(shower, 'Head_Sitting,Shoulder_Tilt')
        elif HS:
            put_text(shower, 'Head_Sitting')
        elif ST:
            put_text(shower, 'Shoulder_Tilt')
        # Show
        cv2.imshow('PoseDet', shower)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
