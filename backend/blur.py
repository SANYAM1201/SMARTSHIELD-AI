import cv2


class Blur:

    def __init__(self, blur_type="gaussian", kernel_size=99):

        self.blur_type = blur_type

        # Gaussian kernel size must always be odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.kernel_size = kernel_size


    def blur_regions(self, image, boxes):

        output = image.copy()

        for (x, y, w, h) in boxes:

            x = max(0, x)
            y = max(0, y)

            w = min(w, output.shape[1] - x)
            h = min(h, output.shape[0] - y)

            roi = output[y:y+h, x:x+w]

            if roi.size == 0:  #.size means total number of elements w*h*colorchannels
                continue

            # Gaussian Blur
            if self.blur_type == "gaussian":

                blurred = cv2.GaussianBlur(
                    roi,
                    (self.kernel_size, self.kernel_size),
                    30
                )

                output[y:y+h, x:x+w] = blurred

            # Black Box
            elif self.blur_type == "black":

                cv2.rectangle(
                    output,
                    (x, y),
                    (x+w, y+h),
                    (0, 0, 0),
                    -1
                )

        return output