import sys
import numpy as np
import cv2

def image_insertion(img_to_be_inserted: str, img_into_which_is_inserted: str, img_result: str='./result.png'):
    input_img = cv2.imread(img_to_be_inserted) 
    Yo, Xo, _ = input_img.shape
    insert_img = cv2.imread(img_into_which_is_inserted) 
    Y, X, _ = insert_img.shape

    # color space change
    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    # shades color limits
    lower_green = np.array([50, 50, 50])
    upper_green = np.array([70, 255, 255])
    # figure selection
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # contour selection
    edged = cv2.Canny(mask, 1000, 1500)
    # contour closure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    # find contours
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            pts1 = np.float32([[0, 0],[X, 0],[0, Y],[X, Y]])
            pts2 = np.array([i[0] for i in approx])
            pts2 = np.float32(pts2)
            pts2 = pts2 +1 
            pts2[0], pts2[1] = pts2[1].copy(), pts2[0].copy()

            if pts2[0][1] == np.min(np.min(pts2, axis=1)):
                temp = pts2[1:].copy()
                if not temp[0][1] == np.min(np.min(temp, axis=1)):
                    pts2[0], pts2[1], pts2[2], pts2[3] = pts2[1].copy(), pts2[3].copy(), pts2[0].copy(), pts2[2].copy()
            elif pts2[1][1] == np.min(np.min(pts2, axis=1)):
                temp = np.concatenate(([pts2[0]], pts2[2:]))
                if not temp[0][1] == np.min(np.min(temp, axis=1)):
                    pts2[0], pts2[1], pts2[2], pts2[3] = pts2[1].copy(), pts2[3].copy(), pts2[0].copy(), pts2[2].copy()

            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(insert_img, M, (Xo, Yo))

            # cut all black pixels
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2BGRA)
            dst[np.all(dst == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]
            
            # cut green rectangle
            stencil = np.zeros(input_img.shape).astype(input_img.dtype)
            contours = [approx]
            cv2.fillPoly(input_img, contours, [0,0,0])
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2BGRA)
            input_img[np.all(input_img == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

            added_image = cv2.addWeighted(input_img, 1, dst, 1, 0)

    cv2.imwrite(img_result, added_image)      

def main():
    try:
        image_insertion(sys.argv[1], sys.argv[2])
    except IndexError as e:
        print('Error: Must be entered path to image files.')
if __name__ == '__main__':
    main()
