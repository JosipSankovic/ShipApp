#include "TrackUtils.h"



TrackUtils::TrackUtils()
{
	tracker = new BYTETracker(10, 1000);
}

TrackUtils::~TrackUtils()
{
	for (const auto& track : TracksHistory.trackHistory) {
	std::ofstream outfile;
	outfile.open("CSV_files_tracks/"+std::to_string(track.first) + "_track.csv");
	outfile << "time,x,y\n";
	for (const auto& point : track.second) {
		outfile << point.frameNumber <<","<< point.bboxCenter.x<<","<< point.bboxCenter.y << "\n";
	}
	outfile.close();
	}
	if (tracker != nullptr) {
		delete tracker;
		tracker = nullptr;
	}
}

void TrackUtils::track(std::vector<Result>& results, float confThresh, int frameNumber)
{
	std::vector<Object> objectsForTracks;

	for (auto& result : results)
	{
		if (result.confidence > confThresh)
			objectsForTracks.push_back({ (cv::Rect2f)result.boundingBox,result.classId,result.confidence });
	}
	detectedTracks = tracker->update(objectsForTracks);
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

void TrackUtils::drawPastPoints(cv::Mat& frame, int frameNumber, int numberOfTrackedPoints)
{
	//uzmi sve trackove iz trenutnog frame-a
	if (TracksHistory.pastPoints.find(frameNumber) == TracksHistory.pastPoints.end()) return;

	for (const auto& track : TracksHistory.pastPoints[frameNumber])
	{
		//za svaki track iz trenutnog frame-a uzmi 20 prethodnih pozicija
		//if(notMovingTrack(track.Id,5,50))
			//continue;
		std::vector<cv::Point> points;
		int j = 0;
		for (int i = TracksHistory.trackHistory[track.Id].size() - 1; i >= 0; i--) {
			// u slucaju da je i<0 znaci da je to kraj tracka
			if (i<0 || j>numberOfTrackedPoints)
				break;
			if (numberOfTrackedPoints > 0)
				j++;
			auto point = TracksHistory.trackHistory[track.Id][i].bboxCenter;
			points.push_back(point);
		}
		drawSpeedVector(frame, track.Id, 50);
		cv::polylines(frame, points, false, ColorsClassId[track.label], 2);
		cv::putText(frame, std::to_string(track.Id), points[0], cv::FONT_HERSHEY_SIMPLEX, 1, ColorsClassId[track.label], 2);
	}

	findPossibleCollisions(frame, frameNumber);
}
// Function to generate a Gaussian kernel
std::vector<float> generateGaussianKernel(int size, float sigma) {
	std::vector<float> kernel(size);
	float sum = 0.0;
	int halfSize = size / 2;
	for (int i = -halfSize; i < halfSize; ++i) {
		float x = static_cast<float>(i);
		kernel[i + halfSize] = std::exp(-0.5f * (x * x) / (sigma * sigma));
		sum += kernel[i + halfSize];
	}
	// Normalize the kernel
	for (int i = 0; i < size; ++i) {
		kernel[i] /= sum;
	}
	return kernel;
}
void TrackUtils::drawSpeedVector(cv::Mat& frame, int trackId, int time)
{
	//get track
	auto& track = TracksHistory.trackHistory[trackId][TracksHistory.trackHistory[trackId].size() - 1];
	// draw speed vector
	cv::Point2f second_point = (cv::Point2f)track.bboxCenter + track.speed_vector;
	cv::arrowedLine(frame, track.bboxCenter, second_point, { 255,255,255 }, 2);
	// draw line where track should be in 100 frames
	auto second_future_point = (cv::Point2f)track.bboxCenter + track.speed_vector * time;
	cv::arrowedLine(frame, second_point, second_future_point, { 25,0,255 }, 2);
}
void TrackUtils::findPossibleCollisions(cv::Mat& frame,int frameNumber)
{
	for (size_t i = 0; i < TracksHistory.pastPoints[frameNumber].size(); ++i) {
		const auto& object1 = TracksHistory.pastPoints[frameNumber][i];
		for (size_t j = i + 1; j < TracksHistory.pastPoints[frameNumber].size(); ++j) {
			const auto& object2 = TracksHistory.pastPoints[frameNumber][j];
			if (object1.Id == object2.Id) continue;
			cv::Point2f r01{ object1.bboxCenter }, r02{ object2.bboxCenter };
			cv::Point2f v1{ object1.speed_vector }, v2{ object2.speed_vector };

			for (int time = 1; time < 700; time++) {
				// r(t)=r0+v*t
				//r1(t)=r2(t) -> sudar
				cv::Point r1 = r01 + v1 * time;
				cv::Point r2 = r02 + v2 * time;

				if (isInsideCircle(r1, r2, 5)) {
					cv::putText(frame, to_string(object1.Id) + "," + to_string(object2.Id) + "fr:" + to_string(time), r1, 2, 1, { 255,0,20 });
					cv::circle(frame, r1, 4, { 255,0,0 }, -1);
					float dist=sqrt(pow(r1.x - r2.x, 2) + pow(r1.y - r2.y, 2));
					TracksHistory.collision_pairs.push_back({ object1.Id,object1.speed_vector,object2.Id,object2.speed_vector,time,frameNumber,dist });
					break;
				}


			}
		}

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
			float distance = std::sqrt((x - ref_x) * (x - ref_x) + (y - ref_y) * (y - ref_y) * 4);
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
	return image;
}
bool TrackUtils::notMovingTrack(int trackId, int speed_thresh, int distance_thresh)
{
	if (getSpeed(trackId) < speed_thresh && getDistancePassed(trackId) < distance_thresh)
		return true;
	else
		return false;
}
float TrackUtils::getSpeed(int trackId)
{
	if (TracksHistory.trackHistory.find(trackId) != TracksHistory.trackHistory.end())
	{
		int last = TracksHistory.trackHistory[trackId].size() - 1;
		int numberOfPoints = 3;
		if (last > numberOfPoints)
		{
			auto lastPoint = TracksHistory.trackHistory[trackId][last];
			auto preLastPoint = TracksHistory.trackHistory[trackId][0];
			auto distance = cv::norm(lastPoint.bboxCenter - preLastPoint.bboxCenter);
			auto time = lastPoint.frameNumber - preLastPoint.frameNumber;
			return (distance / time) * 100;
		}
	}
	return 0.0f;
}
inline bool TrackUtils::isInsideCircle(cv::Point2f pt1, cv::Point2f pt2, float radius)
{
	if (sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2)) < radius)
		return true;
	return false;
}
cv::Point2f TrackUtils::getSpeedVector(int trackId) {
	std::vector<Track>& track = TracksHistory.trackHistory[trackId];
	if (track.size() < 3) return cv::Point2f(0.0f, 0.0f);

	int length = 60;
	int selected_track_id = track.size() - length;
	if (selected_track_id < 0) {
		selected_track_id = 0;
		length = track.size();
	}

	// Generate a Gaussian kernel
	float sigma = 15.0f; // Standard deviation for Gaussian kernel, adjust as needed
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
	length = track[selected_track_id + length - 1].frameNumber-track[selected_track_id].frameNumber;
	// Calculate speed vector
	cv::Point2f pt1 = track[selected_track_id].bboxCenter;
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
		
		cv::Point2f speed_vector = getSpeedVector(track.track_id);
		Track tr = { frameNumber,tlwhToRect(track.tlwh),tlwhToCenter(track.tlwh),track.score,track.label,track.track_id,speed_vector };



		Eigen::Matrix<float, 2, 1> z(tr.bboxCenter.x, tr.bboxCenter.y);
		if (TracksHistory.trackHistory[track.track_id].size() == 0) {
	
			trackSpeedVector.emplace(track.track_id,TrackKalman(tr.bboxCenter.x, tr.bboxCenter.y));
		}
		else if(TracksHistory.trackHistory[track.track_id].size()>2){
			auto it = trackSpeedVector.find(track.track_id);
			TrackKalman& tracker = it->second;
			float dt= TracksHistory.trackHistory[track.track_id][TracksHistory.trackHistory[track.track_id].size() - 1].frameNumber -
				TracksHistory.trackHistory[track.track_id][TracksHistory.trackHistory[track.track_id].size() - 2].frameNumber;
			tracker.A(0, 2) = dt;
			tracker.A(1, 3) = dt;
			tracker.State = tracker.A * tracker.State;
			tracker.P = tracker.A.cast<double>() * tracker.P * tracker.A.transpose().cast<double>() + tracker.Q;


			// Measurement update
			Eigen::Matrix<float, 2, 1> y = z - tracker.H * tracker.State; // Measurement residual
			Eigen::Matrix<double, 2, 2> S = tracker.H.cast<double>() * tracker.P * tracker.H.transpose().cast<double>() + tracker.R; // Residual covariance
			Eigen::Matrix<double, 4, 2> K = tracker.P * tracker.H.transpose().cast<double>() * S.inverse(); // Kalman gain

			tracker.State = tracker.State + K.cast<float>() * y;
			tracker.P = (Eigen::Matrix<double, 4, 4>::Identity() - K * tracker.H.cast<double>()) * tracker.P;
			tr.bboxCenter.x = tracker.State(0, 0);
			tr.bboxCenter.y = tracker.State(1, 0);
			tr.speed_vector.x = tracker.State(2, 0);
			tr.speed_vector.y = tracker.State(3, 0);
		
		}
		TracksHistory.pastPoints[frameNumber].push_back(tr);
		TracksHistory.trackHistory[track.track_id].push_back(tr);
	}

}