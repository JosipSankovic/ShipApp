#include "CollisionTracker.h"


void CollisionTracker::update_track(std::vector<Result> objects, float conf_thresh)
{
	std::vector<Object> objectsForTracks;

	for (auto& result : objects)
	{
		if (result.confidence > conf_thresh)
			objectsForTracks.push_back({ (cv::Rect2f)result.boundingBox,result.classId,result.confidence });
	}

	std::vector<STrack> detectedTracks=tracker.update(objectsForTracks);
	frameNumber++;
	add_tracks(detectedTracks);
}

void CollisionTracker::show_speed_vector(cv::Mat& frame,int dt)
{
	if(_Tracks.track_frames.find(frameNumber-1)!=_Tracks.track_frames.end())
		for (const auto& track : _Tracks.track_frames[frameNumber-1]) {
			//all points of track
			std::vector<TrackPoint> track_points = _Tracks.track_history[track];
			if (track_stoped(track_points))continue;
			TrackPoint last_track = track_points[track_points.size() - 1];
			cv::Point next_point =(cv::Point2f)last_track.center+last_track.speed_vector*dt;
			arrowedLine(frame, last_track.center, next_point, ColorsClassId[track_points[0].label_id], 3);
			cv::putText(frame, std::to_string(track_points[0].track_id), last_track.center -cv::Point(10,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, ColorsClassId[track_points[0].label_id], 1);

		}
}
void CollisionTracker::show_track(cv::Mat& frame,int number_of_past_points) {
	if (_Tracks.track_frames.find(frameNumber - 1) != _Tracks.track_frames.end())
		for (const auto& track : _Tracks.track_frames[frameNumber - 1]) {
			std::vector<TrackPoint> track_points = _Tracks.track_history[track];
			if (track_stoped(track_points))continue;
			std::vector<cv::Point> points;
			int j = 0;
			for (int i = track_points.size() - 1; i >= 0; i--) {

				if (i<0 || j>number_of_past_points)
					break;
				if (number_of_past_points > 0)
					j++;

				auto point = track_points[i].center;
				points.push_back(point);
			}
			cv::polylines(frame, points, false, ColorsClassId[track_points[0].label_id], 2);


		}
}
void CollisionTracker::find_colision(cv::Mat& frame,int dt) {
	if (_Tracks.track_frames.find(frameNumber - 1) != _Tracks.track_frames.end())
		for (int i = 0; i < _Tracks.track_frames[frameNumber-1].size(); i++) {
			int track_id=_Tracks.track_frames[frameNumber-1][i];
			std::vector<TrackPoint>& track_past_o1 = _Tracks.track_history[track_id];
			TrackPoint object1 = track_past_o1[track_past_o1.size() - 1];
			for (int j = i + 1; j < _Tracks.track_frames[frameNumber-1].size(); j++) {
				int track_id2 = _Tracks.track_frames[frameNumber-1][j];
				std::vector<TrackPoint>& track_past_o2 = _Tracks.track_history[track_id2];
				TrackPoint object2 = track_past_o2[track_past_o2.size() - 1];
				if (object1.track_id == object2.track_id) continue;
				cv::Point2f r01{ object1.center }, r02{ object2.center };
				cv::Point2f v1{ object1.speed_vector }, v2{ object2.speed_vector };
				if (track_stoped(track_past_o1) ||track_stoped(track_past_o2))continue;
				for (int time = 1; time < 700; time++) {
					// r(t)=r0+v*t
					//r1(t)=r2(t) -> sudar
					cv::Point r1 = r01 + v1 * time;
					cv::Point r2 = r02 + v2 * time;

					if (isInsideCircle(r1, r2, 5)) {
						cv::putText(frame, to_string(object1.track_id) + "," + to_string(object2.track_id), r1,0,1, { 255,0,20 },2);
						cv::circle(frame, r1, 4, { 255,0,0 }, -1);
						break;
					}
				}

			}
		}
}
inline bool CollisionTracker::isInsideCircle(cv::Point2f pt1, cv::Point2f pt2, float radius)
{
	if (sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2)) < radius)
		return true;
	return false;
}
void CollisionTracker::get_speed_vector(TrackPoint& track_point)
{
	Eigen::Matrix<float, 2, 1> z(track_point.center.x, track_point.center.y);
	if (_Tracks.track_history.find(track_point.track_id) == _Tracks.track_history.end()) {
		_Tracks.KS.emplace(track_point.track_id, KalmanState(track_point.center.x, track_point.center.y));
	}
	else if(_Tracks.track_history[track_point.track_id].size()>=2)
	{
		KalmanState& KS = _Tracks.KS[track_point.track_id];
		int dt = _Tracks.track_history[track_point.track_id][_Tracks.track_history[track_point.track_id].size() - 1].frameNumber-
			_Tracks.track_history[track_point.track_id][_Tracks.track_history[track_point.track_id].size() - 2].frameNumber;
		KS.A(0, 2) = dt;
		KS.A(1, 3) = dt;
		KS.State = KS.A * KS.State;
		KS.P = KS.A.cast<double>() * KS.P * KS.A.transpose().cast<double>() + KS.Q;

		// Measurement update
		Eigen::Matrix<float, 2, 1> y = z - KS.H * KS.State; // Measurement residual
		Eigen::Matrix<double, 2, 2> S = KS.H.cast<double>() * KS.P * KS.H.transpose().cast<double>() + KS.R; // Residual covariance
		Eigen::Matrix<double, 4, 2> K = KS.P * KS.H.transpose().cast<double>() * S.inverse(); // Kalman gain


		KS.State = KS.State + K.cast<float>() * y;
		KS.P = (Eigen::Matrix<double, 4, 4>::Identity() - K * KS.H.cast<double>()) * KS.P;
		track_point.center.x = KS.State(0, 0);
		track_point.center.y = KS.State(1, 0);
		track_point.speed_vector.x = KS.State(2, 0);
		track_point.speed_vector.y = KS.State(3, 0);
	}

}
bool CollisionTracker::track_stoped(std::vector<TrackPoint>& track){
	if (track.size() < 10) return false;
	
	if (get_distance(track[track.size() - 1].center, track[0].center) < 5)return true;
	else
		return false;
}
void CollisionTracker::add_tracks(std::vector<STrack>& tracks)
{
	for (const auto& track : tracks) {

		TrackPoint track_point{ frameNumber,tlwhToRect(track.tlwh),tlwhToCenter(track.tlwh),track.score, track.label,track.track_id };
		get_speed_vector(track_point);
		_Tracks.track_history[track_point.track_id].push_back(track_point);
		_Tracks.track_frames[frameNumber].push_back(track_point.track_id);
	}
}
