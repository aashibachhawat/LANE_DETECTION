import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

# # Steps
# Read the image
# Grayscale
# Calculate ROI
# Masking
# Canny edge detection
# Hough Transform

img = cv.imread('road.jpg')
plt.figure()
plt.imshow(img)
plt.show()

height = img.shape[0]
width = img.shape[1]
region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]

gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cannyed_image = cv.Canny(gray_image, 100, 200)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 
    
    cv.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image



cropped_image = region_of_interest(
    cannyed_image,
    np.array([region_of_interest_vertices], np.int32)
)
plt.figure()
plt.imshow(cropped_image)
plt.show()

lines = cv.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)


# printLines
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
        
    # If there are no lines to draw, exit.
        if lines is None:
            return
    # Make a copy of the original image.
        image = np.copy(img)
    # Create a blank image that matches the original in size.
        line_image = np.zeros(
            (
            image.shape[0],
            image.shape[1],
            3
            ),
            dtype=np.uint8,
        )
    # Loop over all lines and draw them on the blank image.
        for line in lines:
            for x1, y1, x2, y2 in line:
             x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            #  print(type(x1), type(y1), type(x2), type(y2))
             cv.line(line_image, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
        image = cv.addWeighted(image, 0.8, line_image, 1.0, 0.0)
    # Return the modified image.
        return image


left_line_x = []
left_line_y = []
right_line_x = []
right_line_y = []
for line in lines:
    for x1, y1, x2, y2 in line:
        slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
        if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
            continue
        if slope <= 0: # <-- If the slope is negative, left group.
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else: # <-- Otherwise, right group.
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])
min_y = img.shape[0] * (3 / 5) # <-- Just below the horizon
max_y = img.shape[0] # <-- The bottom of the image
poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
))
left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))
poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
))
right_x_start = int(poly_right(max_y))
right_x_end = int(poly_right(min_y))
line_image = draw_lines(
    img,
    [[
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y],
    ]],
    thickness=5,
)

plt.figure()
plt.imshow(line_image)
plt.show()
