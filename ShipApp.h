#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_ShipApp.h"
#include "Yolov8Detection.h"
#include<QSettings>
#include<QFileDialog>
#include<opencv2/opencv.hpp>
#include<opencv2/videoio.hpp>
#include"tracksUtils/TrackUtils.h"
#include "Tracker.h"
#include "utils/trackerHelper.h"
#include <vector>
#include <map>

class ShipApp : public QMainWindow
{
    Q_OBJECT

    struct {
        bool videoLoaded{ false };
        bool videoPlaying{ false };
        bool modelLoaded{ false };
        bool detectionRunning{ false };
    }_ShipAppState;
    struct {
        int fps;
        int frameNumber;
        int totalFrames;
    }_VideoInfo;
   
public:
    ShipApp(QWidget *parent = nullptr);
    ~ShipApp();

private:
    Ui::ShipAppClass ui;
    QSettings _registry;
    QString _path{ "" };
    int _CONFIDENCE_THRESHOLD = 0.6;
    //opencv
private:
    cv::VideoCapture cap;
    cv::Mat frame;
    cv::VideoWriter writer;
    //yolov8 detection
private:
    Yolov8Detection detection;
    std::vector<Result> results;
    //tracker
private:
    TrackUtils tracker;
    /*CustomTracker::Tracker _tracker;*/

	//functions
private:
    QPixmap getPixmap(cv::Mat& Img);
    void loadVideo();
    void playVideo();
    void detectAndTrack();
    void showImage(cv::Mat& img);


private slots:
        void on_btn_Start_clicked();
        void on_btn_Stop_clicked();
        void on_btn_LoadVideo_clicked();
        void resizeEvent(QResizeEvent* event);
        void closeEvent(QCloseEvent* event);
};
