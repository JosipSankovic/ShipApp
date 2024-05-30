#pragma once
#include "../tracker/BYTETracker.h"
#include "../Yolov8Detection.h"

class TrackUtils
{
private:
	struct Track {
		int frameNumber;
		cv::Rect bbox;
		cv::Point bboxCenter;
		float confidence;
		int label;
		int Id;
	};
	struct {

		//classId,vector of IdTrack
		std::map<int, std::vector<Track>> trackHistory;
		//frameNumber,allTracks in that frame
		std::map<int, std::vector<Track>> pastPoints;
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
	bool notMovingTrack(int trackId,int speed_thresh,int distance_thresh);
	float getSpeed(int trackId);
	int getDistancePassed(int trackId);
	void resetTracks() { tracker->reset(); TracksHistory.pastPoints.clear(); TracksHistory.trackHistory.clear(); };

private:
	void addTracks(int frameNumber);
	cv::Rect tlwhToRect(std::vector<float> track) { return cv::Rect((int)(track[0]), (int)(track[1]), (int)(track[2]), (int)(track[3]));};
	cv::Point tlwhToCenter(std::vector<float> track) { return cv::Point((int)(track[0]+track[2]/2), (int)(track[1]+track[3]/2)); };
};