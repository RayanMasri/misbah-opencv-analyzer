import cv2
import numpy as np
import pytesseract
from tesserocr import PyTessBaseAPI
import os
main_api = PyTessBaseAPI(lang='ara')
def optimized_ocr(image, api=None, color='BGR'):
    api = api or main_api
    bytes_per_pixel = image.shape[2] if len(image.shape) == 3 else 1
    height, width   = image.shape[:2]
    bytes_per_line  = bytes_per_pixel * width

    if bytes_per_pixel != 1 and color != 'RGB':
        # non-RGB color image -> convert to RGB
        image = cv2.cvtColor(image, getattr(cv2, f'COLOR_{color}2RGB'))
    elif bytes_per_pixel == 1 and image.dtype == bool:
        # binary image -> convert to bitstream
        image = np.packbits(image, axis=1)
        bytes_per_line  = image.shape[1]
        width = bytes_per_line * 8
        bytes_per_pixel = 0
    # else image already RGB or grayscale

    api.SetImageBytes(image.tobytes(), width, height,
                        bytes_per_pixel, bytes_per_line)
    
    return api.GetUTF8Text()


# print()
tessdata_dir_config = r'--tessdata-dir "C:\Users\MrRya\OneDrive\Desktop\main\python\misbah analyzer (qudrat package)/tessdata"'
def expand(image, scale_percent=60):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)    
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    
_image = cv2.imread('outfile.png')
image = expand(_image, 20)
# blur = cv2.pyrMeanShiftFiltering(image, 11, 21)
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# cv2.drawContours(image, cnts, -1, (0,0,255), 3)

cyan = np.array([164, 161, 77])

def ColorDistance(bgr1,bgr2):
    rm = 0.5*(bgr1[2]+bgr2[2])
    d = abs(sum((2+rm,4,3-rm)*(abs(bgr1-bgr2))**2))**0.5
    return d




def min_clamp(number, minimum):
    if number < minimum:
        return minimum
    return number
def max_clamp(number, maximum):
    if number > maximum:
        return maximum
    return number


boxes = []
all_boxes = []

for c in cnts:
    # Filter by area
    if abs(750 - cv2.contourArea(c)) > 50: continue
    rect = cv2.minAreaRect(c)

    # Filter by rotation
    if abs(rect[2]) % 90 != 0: continue

    # Filter by centroid
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    box = cv2.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
    box = np.intp(box)
    
    all_boxes.append([cx, cy, box])

    cv2.drawContours(image,[box],0,(0,0,255),2)
    if abs(1104 - cx) > 10: continue
    cv2.drawContours(image,[box],0,(0,255,0),2)



    # Get contour image
    x,y,w,h = cv2.boundingRect(c)   
    crop = image[y:y+h,x:x+w]    

    white_pixels = np.where(
        (abs(crop[:, :, 0] - 255) <= 25) &
        (abs(crop[:, :, 1] - 255) <= 25) &
        (abs(crop[:, :, 2] - 255) <= 25) 
    )

    # Set pixels to cyan
    crop[white_pixels] = [164, 161, 77]

    boxes.append([cx, cy, box])

boxes = list(sorted(boxes, key=lambda e: e[1]))

for i in range(len(boxes)):
    # Get adjacent boxes
    adjacent = [min_clamp(i - 1, 0), max_clamp(i + 1, len(boxes) - 1)]
    adjacent = list(map(lambda e: [0, 0] if e == i else boxes[e], adjacent))

    # Get closest adjacent box
    (x, y, box) = boxes[i]
    closest = sorted(adjacent, key=lambda e: abs(e[1] - y))[0]

    closest_index = next((index for index, item in enumerate(adjacent) if item[0] == closest[0] and item[1] == closest[1]), -1)

    # If direction is below, box is a question, if direction is above, box is an answer
    if closest_index == 1:
        for j in range(450, 650):
            right = max(box, key=lambda e: e[0])[0] * 5
            left = min(box, key=lambda e: e[0])[0] * 5
            top = min(box, key=lambda e: e[1])[1] * 5
            bottom = max(box, key=lambda e: e[1])[1] * 5


            crop = _image[top:bottom, (left - j):left]
            # crop = expand(crop, 200)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.medianBlur(crop, 3)
            crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            crop = cv2.bitwise_not(crop)

            (height, width) = crop.shape
            b_height = height * 4
            b_width = width * 2
            white = np.zeros((b_height, b_width), dtype=np.uint8)
            white.fill(255)
            # print([width, height])
            # print([b_width, b_height])
            overlay_x = int((b_width - width) / 4)
            overlay_y = int((b_height - height) / 2)
            # print([overlay_x, overlay_y])
            white[overlay_y:overlay_y + height, overlay_x:overlay_x + width] = crop


            # text = pytesseract.image_to_string(crop)
            # text = pytesseract.image_to_string(white, lang="ara", config=tessdata_dir_config)
            text = optimized_ocr(crop)
            print(j)
            print(text)
            cv2.imshow('omage', crop)
            # cv2.imshow('oamage', white)
            cv2.waitKey(0)
            # print()
            # print(box)

    
    # print(f'Me: {boxes[i]}')
    # print(f'Adjacent: {adjacent}')
    # print(f'Closest: {closest}')
    # print(f'Closest Index: {closest_index}')

    # cv2.circle(image, (x, y), 2, (0, 0, 0), 2)

    # avg_color = np.array(cv2.mean(crop)).astype(np.uint8)[:3]
    # dist = ColorDistance(avg_color, cyan)

    resized = expand(image, 70)
    # resize image
    # cv2.imshow(f'thresh ', thresh)
    cv2.imshow(f'image ', resized)
    cv2.waitKey()

