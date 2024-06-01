#pragma once
#include "../Yolov8Detection.h"
#include "math.h"
#define PI 3.14
namespace CustomTracker {
	struct InputTracks;
	struct Track;
}
namespace Utils {
	void yoloToTrackerInput(std::vector<Result>& yolo, std::vector<CustomTracker::InputTracks>& output);
	void calculateNextPosition(std::vector<CustomTracker::Track>& tracks, double& nextPositionDistance, double angleToBePossible);
	int closestToPosition(std::vector<CustomTracker::InputTracks>& inputTracks, double& nextPositionDistance, double anglesToBeNext, cv::Rect position, std::set<int>& used);
	cv::Point boxToPoint(cv::Rect box);
	double distance(cv::Point p1, cv::Point p2);
}	