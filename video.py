import cv2
import numpy as np
#Трекер шайбы на изображении

#Класс фильтра Калмана
class KalmanFilter:
	kf = cv2.KalmanFilter(4, 2)
	kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
	kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

	def Estimate(self, coordX, coordY):
		#оценка местоположения шайбы
		measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
		self.kf.correct(measured)
		predicted = self.kf.predict()
		return predicted


#захват видео
cap = cv2.VideoCapture('C:/andrey/tracker/v3.mp4')

#первоначальная детекция и выделение движущихся частей
object_detector = cv2.createBackgroundSubtractorMOG2()

#объект фильтра Калмана
kfObj = KalmanFilter()
predictedCoords = np.zeros((2, 1), np.float32)

#центр шайбы и радиус
a,b,r =0,0,0


while True:
	ret, frame = cap.read()

	#Детекция движущихся объектов, предобработка изображения и нахождение шайбы преобразованиями Хафа
	mask = object_detector.apply(frame)
	_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	canny = cv2.Canny(mask, 200, 20)
	detected_circles = cv2.HoughCircles(canny,
										cv2.HOUGH_GRADIENT, 1.2, 120, param1=100,
										param2=20, minRadius=15, maxRadius=23)

	#Нарисуем найденные круги
	if detected_circles is not None:
		#приводим параметры a, b, r к int
		detected_circles = np.uint16(np.around(detected_circles))

		for pt in detected_circles[0, :]:
			a, b, r = pt[0], pt[1], pt[2]
			print("cur", a, b)
			print(r)
			#Предсказание фильтром Калмана
			pred = kfObj.Estimate(a, b)
			#Рисуем круг
			cv2.circle(frame, (a, b), r, (0, 255, 0), 2)

	else:
		pred = kfObj.Estimate(a, b)
		#a,b = int(pred[[0]]), int(pred[[1]])
		print("pred", pred)
		cv2.circle(frame, (a,b), 20, (0, 0, 255), 2)


	cv2.imshow("frame", frame)
	cv2.imshow("mask", mask)

	key = cv2.waitKey(30)
	if key ==27:
		break

cap.release()
cv2.destroyAllWindows()
