import cv2

image_path = (
    "dataset/valid/images/"
    "HA-27-_jpg.rf.6fb564fcabf5ddab6d4938f5afcac005.jpg"
)

image = cv2.imread(image_path)

if image is not None:
    print("Image loaded successfully")
    print("Image dimensions:", image.shape)
    print("Image type:", type(image))

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("BGR pixel:", image[100, 100])
    print("RGB pixel:", rgb_image[100, 100])

    cv2.imshow("CropGuard Image", image)
    cv2.waitKey(0)

else:
    print("Failed to load image")

cv2.destroyAllWindows()