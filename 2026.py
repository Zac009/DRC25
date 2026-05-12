import cv2
import numpy as np

# =========================
# CONFIG
# =========================
print("Starting lane detection...")
IMAGE_PATH = "AAAA.png"
WIDTH, HEIGHT = 640, 720

# HSV color ranges
LOWER_BLUE = np.array([100, 100, 50])
UPPER_BLUE = np.array([140, 255, 255])

LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])

# Perspective transform points (EDIT THESE FOR YOUR CAMERA)
SRC_POINTS = np.float32([
    [100, 400],
    [540, 400],
    [0, 720],
    [640, 720]
])

DST_POINTS = np.float32([
    [0, 0],
    [WIDTH, 0],
    [0, HEIGHT],
    [WIDTH, HEIGHT]
])

# =========================
# LOAD IMAGE
# =========================
frame = cv2.imread(IMAGE_PATH)
frame = cv2.resize(frame, (WIDTH, HEIGHT))

# =========================
# COLOR SEGMENTATION
# =========================
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

combined_mask = cv2.bitwise_or(blue_mask, yellow_mask)

# =========================
# MORPHOLOGICAL CLEANUP
# =========================
kernel = np.ones((5, 5), np.uint8)

clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

# =========================
# PERSPECTIVE TRANSFORM
# =========================
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
warped = cv2.warpPerspective(clean_mask, M, (WIDTH, HEIGHT))

# =========================
# OCCUPANCY GRID (2D MAP)
# =========================
occupancy_grid = (warped > 0).astype(np.uint8)

# =========================
# CONTOUR EXTRACTION
# =========================
contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_map = np.zeros_like(warped)
cv2.drawContours(contour_map, contours, -1, 255, 2)

# =========================
# CENTERLINE ESTIMATION
# =========================
blue_points = np.column_stack(np.where(blue_mask > 0))
yellow_points = np.column_stack(np.where(yellow_mask > 0))

centerline_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

if len(blue_points) > 0 and len(yellow_points) > 0:
    min_len = min(len(blue_points), len(yellow_points))
    
    blue_points = blue_points[:min_len]
    yellow_points = yellow_points[:min_len]

    center_points = (blue_points + yellow_points) // 2

    for y, x in center_points:
        cv2.circle(centerline_img, (x, y), 1, (0, 255, 0), -1)

# =========================
# VISUALIZATION
# =========================
cv2.imshow("Original", frame)
cv2.imshow("Blue Mask", blue_mask)
cv2.imshow("Yellow Mask", yellow_mask)
cv2.imshow("Clean Mask", clean_mask)
cv2.imshow("Warped (Top-Down)", warped)
cv2.imshow("Occupancy Grid", occupancy_grid * 255)
cv2.imshow("Contours", contour_map)
cv2.imshow("Centerline", centerline_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# =========================
# OPTIONAL: EXPORT MAP
# =========================
np.save("occupancy_grid.npy", occupancy_grid)
cv2.imwrite("warped_map.png", warped)