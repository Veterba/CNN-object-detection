import cv2
import json
import torch
from PIL import Image
from torchvision import transforms

from model import Model
from decode import decode
from config import DATASET_ANNFILE


def load_class_names(ann_file):
    # read category id -> name straight from the coco json (no pycocotools needed)
    annotations = json.load(open(ann_file))
    return {category["id"]: category["name"] for category in annotations["categories"]}


class capture:

    def __init__(self, weights="model.weights.pt", conf_threshold=0.15):
        self.cap = cv2.VideoCapture(0)
        self.conf_threshold = conf_threshold
        self.class_names = load_class_names(DATASET_ANNFILE)

        # load once, freeze for inference (dropout off, BN uses running stats)
        self.model = Model()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

        # must match training preprocessing exactly
        self.preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def detect(self, frame):
        # cv2 gives BGR numpy; training saw RGB PIL -> convert before preprocessing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model_input = self.preprocess(Image.fromarray(rgb_frame)).unsqueeze(0)
        with torch.no_grad():
            predictions = self.model(model_input)
        return decode(predictions, conf_threshold=self.conf_threshold)

    def draw(self, frame, detections):
        frame_height, frame_width = frame.shape[:2]
        for detection in detections:
            left, top, right, bottom = detection["box"]
            # decode returns [0,1] coords -> scale to this frame's pixels
            left = int(left * frame_width)
            right = int(right * frame_width)
            top = int(top * frame_height)
            bottom = int(bottom * frame_height)

            name = self.class_names.get(detection["class_id"], str(detection["class_id"]))
            caption = f"{name} {detection['score']:.2f}"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, caption, (left, max(top - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def record(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            unmirrored = cv2.flip(frame, 1)
            detections = self.detect(unmirrored)
            self.draw(unmirrored, detections)

            cv2.imshow('frame', unmirrored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture().record()
