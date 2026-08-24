import cv2

image = cv2.imread("dataset/valid/images/HA-27-_jpg.rf.6fb564fcabf5ddab6d4938f5afcac005.jpg")

if image is not None:
    print("Image loaded successfully")
    print("Image dimensions:", image.shape)

    cv2.imshow("CropGuard Image", image)
    cv2.waitKey(0)

print("BGR pixel:", image[100, 100])
print("RGB pixel:", rgb_image[100, 100])
print(image.shape)
cv2.destroyAllWindows()