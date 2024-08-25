#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_ShipApp.h"
#include "Yolov8Detection.h"
#include<QSettings>
#include<QFileDialog>
#include<opencv2/opencv.hpp>
#include<opencv2/videoio.hpp>
#include "CollisionTracker.h"
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
		int fps{ 30 };
		int frameNumber{ 0 };
		int totalFrames{ 0 };
	}_VideoInfo;
public:
	ShipApp(QWidget* parent = nullptr);
	~ShipApp();
private:
	Ui::ShipAppClass ui;
	QSettings _registry;
	QString _path{ "" };
	float _CONFIDENCE_THRESHOLD = 0.2;
private:
	//opencv
	cv::VideoCapture video;
	cv::Mat frame;
	cv::VideoWriter writer;
private:
	//yolov8 detection
	Yolov8Detection detection_model;
	std::vector<Result> results;
private:
	//tracker
	CollisionTracker tracker;
private:
	//functions
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
