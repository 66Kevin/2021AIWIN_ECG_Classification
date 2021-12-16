import glob

import numpy as np
import cv2
import paddle
from paddle import  to_tensor

from resnet import ResNet50_vd

def main():

    model = ResNet50_vd(pretrained="output/best_model/model", class_num=2)
    model.eval()

    img_lists = glob.glob("dataset/val/*.png")
    img_lists = sorted(img_lists)
    lines = ["name,tag\n"]
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists):
            img = cv2.imread(im_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img = img / 255.0
            img -= (0.5, 0.5, 0.5)
            img /= (0.5, 0.5, 0.5)
            img = np.transpose(img, [2, 0, 1])
            img = img.astype("float32")
            img = to_tensor(img)
            img = paddle.unsqueeze(img, axis=0)
            out = model(img)
            out = paddle.nn.functional.softmax(out)
            pre = paddle.argmax(out).numpy()
            lines.append("{},{}\n".format(im_path.split('/')[-1].split('.')[0], pre[0]))
    with open("answer.csv", "w") as f:
        f.writelines(lines)
    print("Save to answer.csv")


if __name__ == '__main__':
    main()
