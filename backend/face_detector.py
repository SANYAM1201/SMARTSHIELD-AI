import cv2


class FaceDetector:

    def __init__(self):

        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )


    def detect_faces(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40,40)
        )

        face_boxes = []

        for (x,y,w,h) in faces:

            face_boxes.append(
                (
                    int(x),
                    int(y),
                    int(w),
                    int(h)
                )
            )

        return face_boxes