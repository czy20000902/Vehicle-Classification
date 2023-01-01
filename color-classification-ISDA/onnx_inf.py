import argparse
import glob
import os

import cv2
import numpy as np
import onnxruntime
import tqdm

np.set_printoptions(suppress=True)

def get_parser():
    parser = argparse.ArgumentParser(description="onnx model inference")

    parser.add_argument("--model_path",
                        default="./carcolor_RESNET18.onnx",
                        help="onnx model path")
    parser.add_argument("--fc_path",
                        default="./carcolor_FC.onnx",
                        help="onnx fc path")
    parser.add_argument("--input",
                        default="/home/chenzhaoyang/color_classification_ISDA/*.jpg",
                        nargs="+",
                        help="A list of space separated input images; "
                             "or a single glob pattern such as 'directory/*.jpg'", )
    parser.add_argument("--output",
                        default='./onnx_output',
                        help='path to save converted caffe model')
    parser.add_argument("--height",
                        type=int,
                        default=224,
                        help="height of image")
    parser.add_argument("--width",
                        type=int,
                        default=224,
                        help="width of image")
    return parser


def preprocess(image_path, image_height, image_width):
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img


def preprocess(image_path, image_height, image_width):
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]
    # mean = np.array([[[0.485, 0.456, 0.406]]])
    # std = np.array([[[0.229, 0.224, 0.225]]])
    mean = np.array([[[0.5, 0.5, 0.5]]])
    std = np.array([[[0.5, 0.5, 0.5]]])
    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img / 255.0
    img = (img - mean) / std

    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)

    return img


def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


if __name__ == "__main__":
    args = get_parser().parse_args()

    ort_sess = onnxruntime.InferenceSession(args.model_path)
    fc_sess = onnxruntime.InferenceSession(args.fc_path)

    input_name = ort_sess.get_inputs()[0].name
    fc_input_name = fc_sess.get_inputs()[0].name

    if not os.path.exists(args.output): os.makedirs(args.output)

    if args.input:
        print(args.input)
        # if os.path.isdir(args.input):
        #     args.input = glob.glob(os.path.expanduser(args.input[0]))
        #     assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            image = preprocess(path, args.height, args.width)
            feat = ort_sess.run(None, {input_name: image})[0]
            # print(feat)
            feat = fc_sess.run(None, {fc_input_name: feat[0]})[0]
            print(path[2:])
            print(feat)
            feat = normalize(feat, axis=0)
            print(feat)
            # feat = normalize(feat, axis=1)
            np.save(os.path.join(args.output, path.replace('.jpg', '.npy').split('/')[-1]), feat)
