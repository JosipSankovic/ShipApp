#pragma once
#include "Yolov8Detection.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
struct  Track
{
	cv::Rect box{ 0,0,0,0 };
	cv::Point2f speed{ 0.0,0.0 };
	float score{ 0.0 };
	int track_id{ -1 };
	int frameNumber{ 0 };
	int label = 0;

	bool operator==(const Track& tr) const {
		return box == tr.box&&speed==tr.speed&&score==tr.score&&track_id==tr.track_id&&frameNumber==tr.frameNumber&&label==tr.label;
	}
};
class CustomTracker
{
public:

	struct KalmanState {
		Eigen::Matrix<float, 4, 4> A;
		Eigen::Matrix<double, 4, 4> P;
		Eigen::Matrix<float, 4, 1> State;
		Eigen::Matrix<float, 2, 4> H;
		Eigen::Matrix<double, 4, 4> Q;
		Eigen::Matrix<double, 2, 2> R;

		KalmanState(cv::Point initial_point)
			: A((Eigen::Matrix<float, 4, 4>() << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1).finished()),
			P(Eigen::Matrix<double, 4, 4>::Identity()),
			State((Eigen::Matrix<float, 4, 1>() << initial_point.x, initial_point.y, 0.0, 0.0).finished()),
			H((Eigen::Matrix<float, 2, 4>() << 1, 0, 0, 0, 0, 1, 0, 0).finished()),
			Q(Eigen::Matrix<double, 4, 4>::Identity() * 0.001),
			R(Eigen::Matrix<double, 2, 2>::Identity() * 10)
		{}

		KalmanState()  // Default constructor
			: A(Eigen::Matrix<float, 4, 4>::Identity()),
			P(Eigen::Matrix<double, 4, 4>::Identity()),
			State(Eigen::Matrix<float, 4, 1>::Zero()),
			H(Eigen::Matrix<float, 2, 4>::Identity()),
			Q(Eigen::Matrix<double, 4, 4>::Identity() * 0.001),
			R(Eigen::Matrix<double, 2, 2>::Identity() * 50)
		{}
	};

	CustomTracker();

private:
	//objectId,state
	std::map<int,KalmanState> objects;
	std::map<int, std::vector<Track>>tracks;
	float low_score;
	float high_score;
	int fr_num = 0;
	int track_id_counter = 0;
	int track_lost_tresh = 50;
	int DISTANCE_THRESH;

public:
	std::vector<Track> update(std::vector<Result> detected_objects, cv::Mat& frame, float conf_thresh=-1,float low_conf_thresh=-1);
	cv::Point rect_center(cv::Rect rect) { return cv::Point(rect.x+rect.width/2,rect.y+rect.height/2); }
	void reset() {
		objects.clear();
		tracks.clear();
		fr_num = 0;
		track_id_counter = 0;
	};
private:
	void get_speed_vector(Track& track);
	double get_distance(cv::Point first_point, cv::Point second_point) { return sqrt(pow(first_point.x - second_point.x, 2) + pow(first_point.y - second_point.y, 2)); };
	bool use_object_id(int track_id);
	bool active_id(int track_id);
	double get_speed(cv::Point2f point) { return sqrt(point.x * point.x + point.y * point.y); };
	double get_radius(cv::Rect rect) { return sqrt(rect.width * rect.width + rect.height * rect.height) / 2; };
};

