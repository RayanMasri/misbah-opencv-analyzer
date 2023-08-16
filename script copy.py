import fitz
import cv2
import time
# # len(doc) - length of pages
# doc = fitz.open('file.pdf')
# page = doc.load_page(1)
# page.get_pixmap().save('image.png')
# doc.close()

def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

# reading image
img = cv2.imread('image.png')
  
# converting image into grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# setting threshold of gray image
_, threshold = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)


# using a findContours() function
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
i = 0

# for asd in range(1, 1000):

# list for storing names of shapes
for contour in contours:

    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        # contour, (asd / 1000) * cv2.arcLength(contour, True), True)
        contour, (0.035) * cv2.arcLength(contour, True), True)
        # contour, (asd / 0.031) * cv2.arcLength(contour, True), True)
    
    # using drawContours() function


    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

    # putting shape name at center of each shape
    # if len(approx) == 3:
        # cv2.putText(img, 't', (x, y),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if len(approx) == 4:
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
        # print(contour)
        cv2.putText(img, 'q', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # elif len(approx) == 5:
    #     cv2.putText(img, 'p', (x, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # elif len(approx) == 6:
    #     cv2.putText(img, 'h', (x, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # else:
    #     cv2.putText(img, 'c', (x, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# displaying the image after drawing contours
cv2.imshow(f'shapes', zoom(img, 2.5))
cv2.waitKey(0)
cv2.destroyAllWindows()