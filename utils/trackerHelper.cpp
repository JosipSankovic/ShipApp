#include "trackerHelper.h"

void Utils::yoloToTrackerInput(std::vector<Result>& yolo, std::vector<CustomTracker::InputTracks>& output)
{
	output.clear();
	for (const auto& result : yolo) {
		output.push_back({ result.classId,result.confidence ,result.boundingBox });
	}
}

cv::Point Utils::boxToPoint(cv::Rect box)
{
	cv::Point pt;
	pt.x = box.x + box.width / 2;
	pt.y = box.y + box.height / 2;
	return pt;
}

double Utils::distance(cv::Point p1, cv::Point p2)
{
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void Utils::calculateNextPosition(std::vector<CustomTracker::Track>& tracks, double& nextPositionDistance,double angleToBePossible)
{
	if (tracks.size() == 1) {
		nextPositionDistance = 1.0;
		angleToBePossible = PI;
		return;
	}
	else {
		double distancePassed = 0;
		auto trackLast = tracks[tracks.size()-1];
		distancePassed = Utils::distance(Utils::boxToPoint(tracks[0].bbox), Utils::boxToPoint(trackLast.bbox));
		double timePassed = trackLast.frameNumber - tracks[0].frameNumber;
		nextPositionDistance= distancePassed / timePassed;
	}
}

int Utils::closestToPosition(std::vector<CustomTracker::InputTracks>& inputTracks, double& nextPositionDistance, double anglesToBeNext,cv::Rect position, std::set<int>& used)
{
	if (inputTracks.empty()) {
		CustomTracker::InputTracks tr;
		return -1;
	}
	double minDifference = 3000;
	int i = 0;
	int selected = -1;
	for (const auto& track : inputTracks) {
		if (used.find(i) != used.end()) {
			i++; continue;
		}
		double distanceToPoint = Utils::distance(boxToPoint(track.bbox), boxToPoint(position));
		double distanceDifference = std::abs(distanceToPoint - nextPositionDistance);
		if (abs(distanceDifference) < minDifference) {
			selected = i;
			minDifference = distanceDifference;
		}
		i++;
	}
	return selected;
}

