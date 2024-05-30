#include "TrackUtils.h"



TrackUtils::TrackUtils()
{
	tracker = new BYTETracker(10, 1000);
}

TrackUtils::~TrackUtils()
{
	if (tracker != nullptr) {
		delete tracker;
		tracker = nullptr;
	}
}

void TrackUtils::track(std::vector<Result>& results,float confThresh,int frameNumber)
{
	std::vector<Object> objectsForTracks;
	
	for (auto& result : results)
	{
		if (result.confidence > confThresh)
			objectsForTracks.push_back({ (cv::Rect2f)result.boundingBox,result.classId,result.confidence });
	}
	detectedTracks=tracker->update(objectsForTracks);
	addTracks(frameNumber);
}

void TrackUtils::drawTracks(cv::Mat& frame)
{
	for (auto& track : detectedTracks)
	{
		cv::Rect rec = tlwhToRect(track.tlwh);
		cv::rectangle(frame, rec, ColorsClassId[track.label], 2);
		cv::putText(frame, std::to_string(track.track_id), cv::Point(rec.x, rec.y), cv::FONT_HERSHEY_SIMPLEX, 1, ColorsClassId[track.label], 2);
	}
}

void TrackUtils::drawPastPoints(cv::Mat& frame,int frameNumber,int numberOfTrackedPoints)
{
	//uzmi sve trackove iz trenutnog frame-a
	if(TracksHistory.pastPoints.find(frameNumber)!=TracksHistory.pastPoints.end())
		for (const auto& track : TracksHistory.pastPoints[frameNumber])
		{
			//za svaki track iz trenutnog frame-a uzmi 20 prethodnih pozicija
			//if(notMovingTrack(track.Id,5,50))
				//continue;
			std::vector<cv::Point> points;
			int j = 0;
			for (int i = TracksHistory.trackHistory[track.Id].size() - 1; i >= 0; i--) {
				// u slucaju da je i<0 znaci da je to kraj tracka
				if(i<0||j>numberOfTrackedPoints)
					break;
				if(numberOfTrackedPoints>0)
					j++;
				auto point=TracksHistory.trackHistory[track.Id][i].bboxCenter;
				points.push_back(point);
			}
			
			cv::polylines(frame, points, false, ColorsClassId[track.label], 2);
			cv::putText(frame, std::to_string(track.Id), points[0], cv::FONT_HERSHEY_SIMPLEX, 1, ColorsClassId[track.label], 2);
		}
}

bool TrackUtils::notMovingTrack(int trackId,int speed_thresh,int distance_thresh)
{
	if(getSpeed(trackId)<speed_thresh&&getDistancePassed(trackId)<distance_thresh)
		return true;
	else
		return false;
}
float TrackUtils::getSpeed(int trackId)
{
	if (TracksHistory.trackHistory.find(trackId) != TracksHistory.trackHistory.end())
	{
		int last=TracksHistory.trackHistory[trackId].size() - 1;
		int numberOfPoints = 3;
		if (last> numberOfPoints)
		{
			auto lastPoint = TracksHistory.trackHistory[trackId][last];
			auto preLastPoint = TracksHistory.trackHistory[trackId][0];
			auto distance = cv::norm(lastPoint.bboxCenter - preLastPoint.bboxCenter);
			auto time = lastPoint.frameNumber - preLastPoint.frameNumber;
			return (distance / time)*100;
		}
	}
	return 0.0f;
}
int TrackUtils::getDistancePassed(int trackId)
{
	if (TracksHistory.trackHistory.find(trackId) != TracksHistory.trackHistory.end())
	{
		int last = TracksHistory.trackHistory[trackId].size() - 1;
		if (last > 0)
		{
			auto lastPoint = TracksHistory.trackHistory[trackId][last];
			auto preLastPoint = TracksHistory.trackHistory[trackId][0];
			auto distance = cv::norm(lastPoint.bboxCenter - preLastPoint.bboxCenter);
			return distance;
		}
	}
	return 0;
}
void TrackUtils::addTracks(int frameNumber)
{

	for (const auto& track : detectedTracks)
	{
		Track tr = { frameNumber,tlwhToRect(track.tlwh),tlwhToCenter(track.tlwh),track.score,track.label,track.track_id };
		TracksHistory.pastPoints[frameNumber].push_back(tr);
		TracksHistory.trackHistory[track.track_id].push_back(tr);
	}

}