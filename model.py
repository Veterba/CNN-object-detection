import torch
import torch.nn as nn
import cv2


class Model(nn.Module):

    def __init__(self, num_classes=20, num_archos=3):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.head = nn.Conv2d(128, num_classes * (5 + num_archos), kernel_size=1)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.norm1(self.conv1(x))))
        x = self.pool(self.relu(self.norm2(self.conv2(x))))
        x = self.pool(self.relu(self.norm3(self.conv3(x))))

        x = self.dropout(x)
        x = self.head(x)
        return x


class capture:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def record(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            unmirrored = cv2.flip(frame, 1)

            # Display
            cv2.imshow('frame', unmirrored)

            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# if __name__ == '__main__':
#    cam = capture()
#    cam.record()

if __name__ == '__main__':
    m = Model()
    print(m(torch.randn(1, 3, 128, 128)).shape)
