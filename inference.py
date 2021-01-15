# MIT License

# Copyright (c) 2020 megvii-model

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import json

import cv2
import megengine
import megengine.data.transform as T
import megengine.functional as F
import numpy as np

# pylint: disable=import-error
import model as repvgg_model

logging = megengine.logger.get_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="RepVGG-A0",
        choices=repvgg_model.func_dict.keys(),
        help="model architecture (default: RepVGG-A0)")
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("-i", "--image", default=None, type=str)
    args = parser.parse_args()

    model = repvgg_model.get_RepVGG_func_by_name(args.arch)(deploy=True)

    if args.model is not None:
        logging.info("load from checkpoint %s", args.model)
        checkpoint = megengine.load(args.model)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint)

    if args.image is None:
        path = "assets/cat.jpg"
    else:
        path = args.image
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            # T.Normalize(
            #     mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
            # ),  # BGR
            T.ToMode("CHW"),
        ]
    )

    def infer_func(processed_img):
        model.eval()
        logits = model(processed_img)
        probs = F.softmax(logits)
        return probs

    processed_img = transform.apply(image)[np.newaxis, :]
    probs = infer_func(processed_img)

    top_probs, classes = F.topk(probs, k=5, descending=True)

    with open("assets/imagenet_class_info.json") as fp:
        imagenet_class_index = json.load(fp)

    for rank, (prob, classid) in enumerate(
        zip(top_probs.numpy().reshape(-1), classes.numpy().reshape(-1))
    ):
        print(
            "{}: class = {:20s} with probability = {:4.1f} %".format(
                rank, imagenet_class_index[str(classid)][1], 100 * prob
            )
        )


if __name__ == "__main__":
    main()
