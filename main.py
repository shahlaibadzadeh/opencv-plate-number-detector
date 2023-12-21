import cv2 as cv

video = cv.VideoCapture(1)
licence_cascade = cv.CascadeClassifier("haarcascade_russian_plate_number.xml")

while True:
    _, img = video.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    licences = licence_cascade.detectMultiScale(gray, 1.1, 6)

    for (x, y, w, h) in licences:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(img, "Plate", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (24, 24, 23), 2)
        imroi = img[y:y + h, x:x + w]
        canny = cv.Canny(imroi, 125, 175)
        cv.imshow("ROI", canny)

    cv.imshow('Plate number detector', img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
