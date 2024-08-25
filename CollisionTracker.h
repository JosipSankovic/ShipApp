#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include<opencv2/opencv.hpp>
#include "./tracker/BYTETracker.h"
#include "Yolov8Detection.h"
#include "CustomTracker.h"
#include <forward_list>
class CollisionTracker
{

public:
	CollisionTracker() {};
	~CollisionTracker() {};

private:

	std::vector<cv::Scalar> ColorsClassId{ {0,0,255},{0,255,0},{255,0,0} };
	int frameNumber{ 0 };
	struct CollisionPoint {
		int track_id_1;
		int track_id_2;
		cv::Point collision_point;
	};
	struct Tracks {
		//trackId,tracks
		std::map<int, std::vector<Track>> track_history;
		//frameNumber,tracksId
		std::map<int, std::vector<int>> track_frames;
	}_Tracks;
	//frameNumber,collisionIds,point
	std::forward_list<CollisionPoint> collision_points;
private:
	CustomTracker tracker;
public:
	void update_track(std::vector<Result> objects, float conf_thresh=0.5);
	void show_speed_vector(cv::Mat& frame, int dt=20);
	void show_track(cv::Mat& frame,int number_of_past_points=1);
	void find_collision(int dt = 200);
	void draw_collision_points(cv::Mat& frame);
	void resetTracker() {
		_Tracks.track_frames.clear();
		_Tracks.track_history.clear();
		tracker.reset();
	};
private:
	void add_tracks(std::vector<Track>& tracks);
	bool track_stoped(std::vector<Track>& track);
	inline bool isInsideCircle(cv::Point2f pt1, cv::Point2f pt2, float radius);
	double get_distance(cv::Point first_point, cv::Point second_point) { return sqrt(pow(first_point.x - second_point.x, 2)+pow(first_point.y - second_point.y, 2)); };
	cv::Point2f rect_to_point(cv::Rect& rect) { return cv::Point2f((rect.x+rect.width/2), (rect.y + rect.height / 2)); }
};

