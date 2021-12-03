#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//объект фильтра Калмана и матрица измерений
KalmanFilter KF;
Mat_<float> measurement(2, 1); 

//void initKalman(float x, float y)
//{
//Тут могла быть инициализация параметров и разичных матриц фильтра Калмана - перенесено в main
//}


//функция предсказаний фильтра Калмана
Point kalmanPredict()
{
    Mat prediction = KF.predict();
    Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

    KF.statePre.copyTo(KF.statePost);
    KF.errorCovPre.copyTo(KF.errorCovPost);

    return predictPt;
}

//функция обновления показателей фильтра Калмана
void kalmanCorrect(float x, float y)
{
    measurement(0) = x;
    measurement(1) = y;
    Mat estimated = KF.correct(measurement);
    Point statePt(estimated.at<float>(0), estimated.at<float>(1));
   // return statePt;
}

int main()
{
    //ввод и считывание расположения видеофайла
    string dir;
    cout << "Enter video directory : ";
    cin >> dir;

    //создание объекта, отвечающего за извлечение заднего фона
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg = createBackgroundSubtractorMOG2(500, 16.0, true);
    
    //создание и инициализация объекта, захватывающего видео
    VideoCapture cap(dir);//("C:/andrey/v1.mp4");

    //проверка на ошибки при открытии файла
	if (!cap.isOpened()) {
	    cout << "Error opening video stream or file" << endl;
	    return -1;
	}

    ////координаты центра шайбы
    Point center;

    //инициализация  фильтра Кальмана и дальнейшая инициализация матрицы измерений и отображений
    //4 динамических параметра - координаты центра шайбы и скорость
    //2 параметра измерения - координаты центра шайбы
    KF.init(4, 2, 0);
   

    //инициализация параметров и разичных матриц фильтра Калмана
    KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
    measurement.setTo(Scalar(0));
    measurement.at<float>(0, 0) = center.x;
    measurement.at<float>(1, 0) = center.y;
   
    KF.statePre.at<float>(0) = center.x;
    KF.statePre.at<float>(1) = center.y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;

    KF.statePost.at<float>(0) = center.x;
    KF.statePost.at<float>(1) = center.y;
    KF.statePost.at<float>(2) = 0;
    KF.statePost.at<float>(3) = 0;

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(0));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    //создание объектов изображения, заднего и переднего плана 
    Mat img, foregroundMask, backgroundImage, foregroundImg;

    //вектор хранящий найденные круги(которых по факту или 1, или 0) и радиус
    vector<Vec3f> circles;
    int radius = 0;

    //основной цикл программы
    while (true) {
       kalmanPredict();
        //захват видео и проверка на ошибку
        bool ok = cap.grab();
        if (ok == false) {
            std::cout << "Video Capture Fail" << std::endl;
        }
        //если все ОК!
        else {
            cap.retrieve(img, CAP_OPENNI_BGR_IMAGE);

            if (foregroundMask.empty()) {
                foregroundMask.create(img.size(), img.type());
            }
            bg->apply(img, foregroundMask, true ? -1 : 0);


            //установление порога для цвета и наложение фильтра Кэнни
            threshold(foregroundMask, foregroundMask, 254, 255, THRESH_BINARY);
            Canny(foregroundMask, foregroundMask, 20, 200, 3);
            
            //преобразования Хафа для нахождения шайбы
            HoughCircles(foregroundMask, circles, HOUGH_GRADIENT, 1.2, 120, 200, 20, 18, 23); 

            //в случае если окружностей не найдено - делаем предсказание и рисуем круг
            if ((circles.size() == 0) ){
                Point p;
                p = kalmanPredict();
                circle(img, p, radius, Scalar(255, 255, 255), 2, 8, 0);
            }

            //если нашли окружность - отрисовываем её и улучшаем предсказание
            for (size_t i = 0; i < circles.size(); i++) {
                Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                radius = cvRound(circles[i][2]);
                circle(img, center, radius, Scalar(255, 255, 255), 2, 8, 0);
                kalmanCorrect(circles[i][0], circles[i][1]);
            }

            //очищаем circles
            circles.clear();

            foregroundImg = Scalar::all(0);
            imshow("foreground image", img);
            int key6 = waitKey(40);
            if (!backgroundImage.empty()) {

                int key5 = waitKey(40);
            }
        }
    }
    return EXIT_SUCCESS;
}
