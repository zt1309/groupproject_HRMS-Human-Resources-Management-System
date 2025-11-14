from src.antispoof import check_liveness
import cv2

img = cv2.imread("data/frames_output/test/real/7141944186551_f000005.jpg")
print("Real face liveness:", check_liveness(img))

img = cv2.imread("data/frames_output/test/fake/6c426052-9cd4-4106-8499-e704882f8dcd_f000005.jpg")
print("Fake face liveness:", check_liveness(img))