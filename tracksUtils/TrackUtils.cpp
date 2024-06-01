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

// Function to generate a Gaussian kernel
std::vector<float> generateGaussianKernel(int size, float sigma) {
	std::vector<float> kernel(size);
	float sum = 0.0;
	int halfSize = size / 2;
	for (int i = -halfSize; i < halfSize; ++i) {
		float x = static_cast<float>(i);
		kernel[i+halfSize] = std::exp(-0.5f * (x * x) / (sigma * sigma));
		sum += kernel[i + halfSize];
	}
	// Normalize the kernel
	for (int i = 0; i < size; ++i) {
		kernel[i] /= sum;
	}
	return kernel;
}
void TrackUtils::drawSpeedVector(cv::Mat& frame,int trackId)
{
	for (const auto& track : TracksHistory.pastPoints[TracksHistory.pastPoints.size()-1]) {
		cv::Point2f speed_vector = getSpeedVector(track.Id)*30;
		cv::Point second_point = track.bboxCenter + (cv::Point)speed_vector;
		cv::arrowedLine(frame, track.bboxCenter, second_point, { 255,255,255 }, 2);
	}
}
//function for future use if we want to calculate speed of boat
//creates map of how much distance every pixel represents
cv::Mat TrackUtils::createDistanceMap(int height, int width)
{
	cv::Mat image(height, width, CV_32F, cv::Scalar(0));
	int ref_x = width / 2;
	int ref_y = height;
	
	// calculate the distance from the reference point for each pixel
	// gives more value to points that are y 
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float distance = std::sqrt((x - ref_x) * (x - ref_x) + (y - ref_y) * (y - ref_y)*4);
			image.at<float>(y, x) = distance;
		}
	}

	// Normalize the image to the range 0-255
	double minVal, maxVal;
	cv::minMaxLoc(image, &minVal, &maxVal);
	if (maxVal > 0) { // avoid division by zero
		image = image / maxVal * 255;
	}

	// Convert the image to 8-bit unsigned integer
	image.convertTo(image, CV_8U);
	cv::resize(image, image, { 640,640 });
	cv::imshow("image", image);
	cv::waitKey(0);

	return image;
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
cv::Point2f TrackUtils::getSpeedVector(int trackId) {
	std::vector<Track>& track = TracksHistory.trackHistory[trackId];
	if (track.size() < 3) return cv::Point2f(0.0f, 0.0f);

	int length = 30;
	int selected_track_id = track.size() - length;
	if (selected_track_id < 0) {
		selected_track_id = 0;
		length = track.size();
	}

	// Generate a Gaussian kernel
	float sigma = 10.0f; // Standard deviation for Gaussian kernel, adjust as needed
	std::vector<float> gaussianKernel = generateGaussianKernel(length, sigma);

	cv::Point2f weightedSum(0.0f, 0.0f);
	float totalWeight = 0.0f;

	// Apply Gaussian weights to positions
	for (int i = 0; i < length; ++i) {
		int idx = selected_track_id + i;
		weightedSum += (cv::Point2f)track[idx].bboxCenter * gaussianKernel[i];
		totalWeight += gaussianKernel[i];
	}

	// Calculate the weighted mean position
	cv::Point2f weightedMean = weightedSum / totalWeight;

	// Calculate speed vector
	cv::Point2f pt1 = track[selected_track_id].bboxCenter;
	cv::Point2f pt2 = track.back().bboxCenter;
	cv::Point2f rawSpeed = (pt2 - pt1) / static_cast<float>(length);

	// Smoothen the speed vector using the Gaussian weighted mean
	cv::Point2f smoothedSpeed = (weightedMean - pt1) / static_cast<float>(length);

	return smoothedSpeed;
}
int TrackUtils::getDistancePassed(int trackId)
{
	if (TracksHistory.trackHistory.find(trackId) != TracksHistory.trackHistory.end())
	{
		int last = TracksHistory.trackHistory[trackId].size() - 1;
		if (last > 0)
		{
			auto& lastPoint = TracksHistory.trackHistory[trackId][last];
			auto& preLastPoint = TracksHistory.trackHistory[trackId][0];
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