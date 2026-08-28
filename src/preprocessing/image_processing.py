import cv2

image_path = (
    "dataset/valid/images/"
    "HA-27-_jpg.rf.6fb564fcabf5ddab6d4938f5afcac005.jpg"
)

image = cv2.imread(image_path)

if image is None:
    print("Failed to load image")
    exit()

# -------------------------
# Original image
# -------------------------

print("Original shape:", image.shape)


# -------------------------
# Resize
# -------------------------

resized = cv2.resize(image, (320, 320))

print("Resized shape:", resized.shape)


# -------------------------
# Crop
# -------------------------

cropped = image[100:400, 100:400]

print("Cropped shape:", cropped.shape)


# -------------------------
# Grayscale
# -------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Grayscale shape:", gray.shape)


# -------------------------
# Display images
# -------------------------

cv2.imshow("Original", image)
cv2.imshow("Resized", resized)
cv2.imshow("Cropped", cropped)
cv2.imshow("Grayscale", gray)


# -------------------------
# Save cropped image
# -------------------------

cv2.imwrite(
    "results/camera_test/cropped_test.jpg",
    cropped
)

print("Cropped image saved successfully")


# -------------------------
# Wait and close
# -------------------------

cv2.waitKey(0)
cv2.destroyAllWindows()