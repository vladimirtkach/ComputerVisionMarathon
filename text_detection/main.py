import cv2
import easyocr
import matplotlib.pyplot as plt


image_path = "/home/vladimir/ml/ComputerVisionMarathon/text_detection/data/test4.png"

img = cv2.imread(image_path)

reader = easyocr.Reader(['en'], gpu=False)

text_ = reader.readtext(img)

for t in text_:
    print(t)

    bbox, text, score = t

    cv2.rectangle(img, bbox[0], bbox[2],(0, 255, 0), 5 )

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()




