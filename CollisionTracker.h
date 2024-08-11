#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include<opencv2/opencv.hpp>
#include "./tracker/BYTETracker.h"
#include "Yolov8Detection.h"
class CollisionTracker
{

public:
	CollisionTracker() {};
	~CollisionTracker() {};

private:

	std::vector<cv::Scalar> ColorsClassId{ {0,0,255},{0,255,0},{255,0,0} };
	struct KalmanState {
		Eigen::Matrix<float, 4, 4> A;
		Eigen::Matrix<double, 4, 4> P;
		Eigen::Matrix<float, 4, 1> State;
		Eigen::Matrix<float, 2, 4> H;
		Eigen::Matrix<double, 4, 4> Q;
		Eigen::Matrix<double, 2, 2> R;

		KalmanState(int initial_x, int initial_y)
			: A((Eigen::Matrix<float, 4, 4>() << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1).finished()),
			P(Eigen::Matrix<double, 4, 4>::Identity()),
			State((Eigen::Matrix<float, 4, 1>() << initial_x, initial_y, 0.0, 0.0).finished()),
			H((Eigen::Matrix<float, 2, 4>() << 1, 0, 0, 0, 0, 1, 0, 0).finished()),
			Q(Eigen::Matrix<double, 4, 4>::Identity() * 0.001),
			R(Eigen::Matrix<double, 2, 2>::Identity() * 5)
		{}

		KalmanState()  // Default constructor
			: A(Eigen::Matrix<float, 4, 4>::Identity()),
			P(Eigen::Matrix<double, 4, 4>::Identity()),
			State(Eigen::Matrix<float, 4, 1>::Zero()),
			H(Eigen::Matrix<float, 2, 4>::Identity()),
			Q(Eigen::Matrix<double, 4, 4>::Identity() * 0.001),
			R(Eigen::Matrix<double, 2, 2>::Identity() * 10)
		{}
	};
	struct TrackPoint {
		int frameNumber{ 0 };
		cv::Rect bbox;
		cv::Point center;
		float confidence{ 0.0 };
		int label_id{ -1 };
		int track_id{ -1 };
		cv::Point2f speed_vector{ 0.0,0.0 };

	}_TrackPoint;
	int frameNumber{ 0 };
	struct Tracks {
		//trackId,tracks
		std::map<int, std::vector<TrackPoint>> track_history;
		//frameNumber,tracksId
		std::map<int, std::vector<int>> track_frames;
		//trackId,kalmanState
		std::map<int, KalmanState> KS;
	}_Tracks;
private:
	BYTETracker tracker;
public:
	void update_track(std::vector<Result> objects, float conf_thresh);
	void show_speed_vector(cv::Mat& frame, int dt=20);
	void show_track(cv::Mat& frame,int number_of_past_points=1);
	void find_colision(cv::Mat& frame, int dt = 200);
private:
	void add_tracks(std::vector<STrack>& tracks);
	void get_speed_vector(TrackPoint& track_point);
	bool track_stoped(std::vector<TrackPoint>& track);
	cv::Rect tlwhToRect(std::vector<float> track) { return cv::Rect((int)(track[0]), (int)(track[1]), (int)(track[2]), (int)(track[3])); };
	cv::Point tlwhToCenter(std::vector<float> track) { return cv::Point((int)(track[0] + track[2] / 2), (int)(track[1] + track[3] / 2)); };
	inline bool CollisionTracker::isInsideCircle(cv::Point2f pt1, cv::Point2f pt2, float radius);
	double get_distance(cv::Point first_point, cv::Point second_point) { return sqrt(pow(first_point.x - second_point.x, 2)+pow(first_point.y - second_point.y, 2)); };
};

