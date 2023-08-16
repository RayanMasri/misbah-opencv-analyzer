import fitz
import cv2
import time
# # len(doc) - length of pages
# doc = fitz.open('file.pdf')
# page = doc.load_page(1)
# page.get_pixmap().save('image.png')
# doc.close()
def compare_colors(color1, color2, precision=0):
    return abs(color1[0] - color2[0]) <= precision and abs(color1[1] - color2[1]) <= precision and abs(color1[2] - color2[2]) <= precision


def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

# reading image
img = cv2.imread('image.png')

(height, width, channels) = img.shape

# (b, g, r) = [y][x]
# print(img[196][546])
#[175 171 106]
btn_color = [175, 171, 106]

for y in range(height):
    for x in range(width):
        pixel = img[y][x]

        if compare_colors(pixel, btn_color, 20):
            img[y][x] = (0, 0, 255)
    # break


# print([width, height])
# for i in range(height):
#     img[i][546] = (0, 0, 255)

# for i in range(width):
#     img[196][i] = (0, 255, 0)

cv2.imshow('a', zoom(img))
cv2.waitKey(0)