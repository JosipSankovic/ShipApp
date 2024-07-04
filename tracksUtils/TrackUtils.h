#pragma once
#include "../tracker/BYTETracker.h"
#include "../Yolov8Detection.h"
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>
class TrackUtils
{
private:

	struct TrackKalman {
		Eigen::Matrix<float, 4, 4> A;
		Eigen::Matrix<double, 4, 4> P;
		Eigen::Matrix<float, 4, 1> State;
		Eigen::Matrix<float, 2, 4> H;
		Eigen::Matrix<double, 4, 4> Q;
		Eigen::Matrix<double, 2, 2> R;

		// Constructor to initialize the matrices
		TrackKalman(int initial_x, int initial_y)
			: A((Eigen::Matrix<float, 4, 4>() << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1).finished()),
			P(Eigen::Matrix<double, 4, 4>::Identity()),
			State((Eigen::Matrix<float, 4, 1>() << initial_x, initial_y, 0.0, 0.0).finished()),
			H((Eigen::Matrix<float, 2, 4>() << 1, 0, 0, 0, 0, 1, 0, 0).finished()),
			Q(Eigen::Matrix<double, 4, 4>::Identity() * 0.001),
			R(Eigen::Matrix<double, 2, 2>::Identity() * 10)
		{}
	};
	
	//trackId,TrackKalman
	std::map<int, TrackKalman> trackSpeedVector;
	struct Track {
		int frameNumber;
		cv::Rect bbox;
		cv::Point bboxCenter;
		float confidence;
		int label;
		int Id;
		cv::Point2f speed_vector{ 0.0f,0.0f };
	};
	struct CollisionInfo {
		int Id_1;
		cv::Point2f speed_1;
		int Id_2;
		cv::Point2f speed_2;
		int time_to_collision;
		int frameNumber;
		float error{ 0.0f };
	};
	struct {

		//trackId,vector of IdTrack
		std::map<int, std::vector<Track>> trackHistory;
		//frameNumber,allTracks in that frame
		std::map<int, std::vector<Track>> pastPoints;
		//vector<trackId,trackId>
		std::vector<CollisionInfo> collision_pairs;
	}TracksHistory;
	struct {
		cv::Scalar Boat{ 0,0,255 };
		cv::Scalar Ship{ 0,255,0 };
		cv::Scalar SmallShip{255,0,0};
	}ColorsForClasses;
	public:
	TrackUtils();
	~TrackUtils();

private:
	BYTETracker* tracker = nullptr;
	std::vector<STrack> detectedTracks;
	std::vector<cv::Scalar> ColorsClassId{ {0,0,255},{0,255,0},{255,0,0} };
public:
	void track(std::vector<Result>& results,float confThresh,int frameNumber);
	void drawTracks(cv::Mat& frame);
	void drawPastPoints(cv::Mat& frame,int frameNumber, int numberOfTrackedPoints = 0);
	void resetTracks() { tracker->reset(); TracksHistory.pastPoints.clear(); TracksHistory.trackHistory.clear(); detectedTracks.clear(); };
	void drawSpeedVector(cv::Mat& frame,int trackId,int time=30);
	void findPossibleCollisions(cv::Mat& frame, int frameNumber);
	cv::Mat createDistanceMap(int height, int width);

private:
	void addTracks(int frameNumber);
	float getSpeed(int trackId);
	inline bool isInsideCircle(cv::Point2f pt1, cv::Point2f pt2,float radius);
	cv::Point2f getSpeedVector(int trackId);
	bool notMovingTrack(int trackId,int speed_thresh,int distance_thresh);
	int getDistancePassed(int trackId);
	cv::Rect tlwhToRect(std::vector<float> track) { return cv::Rect((int)(track[0]), (int)(track[1]), (int)(track[2]), (int)(track[3]));};
	cv::Point tlwhToCenter(std::vector<float> track) { return cv::Point((int)(track[0]+track[2]/2), (int)(track[1]+track[3]/2)); };
};