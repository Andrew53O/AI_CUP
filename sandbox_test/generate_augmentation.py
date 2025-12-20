import cv2
import numpy as np

img = cv2.imread("../datasets/test/images1/patient0051_0011.png", 0)
h, w = img.shape

# Rotation (±10° example)
M_rot = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
rotated = cv2.warpAffine(img, M_rot, (w, h))

# Translation (5%)
tx, ty = int(0.05*w), int(0.05*h)
M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, M_trans, (w, h))

# Scaling (±10% example)
scale_factor = 1.1
M_scale = cv2.getRotationMatrix2D((w//2, h//2), 0, scale_factor)
scaled = cv2.warpAffine(img, M_scale, (w, h))

# Horizontal flip
flipped = cv2.flip(img, 1)

# Combine images into a 2x2 block
top_row = np.hstack((rotated, translated))
bottom_row = np.hstack((scaled, flipped))
combined = np.vstack((top_row, bottom_row))

# Save images
cv2.imwrite("original.png", img)
cv2.imwrite("rotated.png", rotated)
cv2.imwrite("translated.png", translated)
cv2.imwrite("scaled.png", scaled)
cv2.imwrite("flipped.png", flipped)
cv2.imwrite("combined.png", combined)
