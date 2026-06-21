import cv2



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



