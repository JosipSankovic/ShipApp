#include "CollisionTracker.h"


void CollisionTracker::update_track(std::vector<Result> objects, float conf_thresh)
{
	std::vector<Track> detectedTracks = tracker.update(objects,conf_thresh);
	frameNumber++;
	add_tracks(detectedTracks);
}

void CollisionTracker::show_speed_vector(cv::Mat& frame, int dt)
{
	if (_Tracks.track_frames.find(frameNumber - 1) != _Tracks.track_frames.end())
		for (const auto& track : _Tracks.track_frames[frameNumber - 1]) {
			//all points of track
			std::vector<Track> track_points = _Tracks.track_history[track];
			if (track_stoped(track_points))continue;
			Track last_track = track_points[track_points.size() - 1];
			cv::Point next_point = rect_to_point(last_track.box) + last_track.speed * dt;
			arrowedLine(frame, rect_to_point(last_track.box), next_point, ColorsClassId[track_points[0].label], 3);
			cv::putText(frame, std::to_string(track_points[0].track_id), rect_to_point(last_track.box) - cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 1.5, ColorsClassId[track_points[0].label],2);

		}
}
void CollisionTracker::show_track(cv::Mat& frame, int number_of_past_points) {
	if (_Tracks.track_frames.find(frameNumber - 1) != _Tracks.track_frames.end())
		for (const auto& track : _Tracks.track_frames[frameNumber - 1]) {
			std::vector<Track> track_points = _Tracks.track_history[track];
			if (track_stoped(track_points))continue;
			std::vector<cv::Point> points;
			int j = 0;
			for (int i = track_points.size() - 1; i >= 0; i--) {

				if (i<0 || j>number_of_past_points)
					break;
				if (number_of_past_points > 0)
					j++;

				auto point = rect_to_point(track_points[i].box);
				points.push_back(point);
			}
			cv::polylines(frame, points, false, ColorsClassId[track_points[0].label], 2);
			cv::putText(frame, to_string(track_points[track_points.size()-1].track_id), rect_to_point(track_points[track_points.size() - 1].box), 0, 1, {25,0,200}, 2);
		}
}
void CollisionTracker::find_collision(int dt) {
	if (_Tracks.track_frames.find(frameNumber - 1) != _Tracks.track_frames.end())
		for (int i = 0; i < _Tracks.track_frames[frameNumber - 1].size(); i++) {
			int track_id = _Tracks.track_frames[frameNumber - 1][i];
			std::vector<Track>& track_past_o1 = _Tracks.track_history[track_id];
			Track object1 = track_past_o1[track_past_o1.size() - 1];
			for (int j = i + 1; j < _Tracks.track_frames[frameNumber - 1].size(); j++) {
				int track_id2 = _Tracks.track_frames[frameNumber - 1][j];
				std::vector<Track>& track_past_o2 = _Tracks.track_history[track_id2];
				Track object2 = track_past_o2[track_past_o2.size() - 1];
				if (object1.track_id == object2.track_id) continue;
				cv::Point2f r01{ rect_to_point(object1.box) }, r02{ rect_to_point(object2.box) };
				cv::Point2f v1{ object1.speed }, v2{ object2.speed };
				if (track_stoped(track_past_o1) || track_stoped(track_past_o2))continue;
				for (int time = 1; time < dt; time++) {
					// r(t)=r0+v*t
					//r1(t)=r2(t) -> sudar
					cv::Point r1 = r01 + v1 * time;
					cv::Point r2 = r02 + v2 * time;
					if (isInsideCircle(r1, r2, 5)) {
						collision_points.assign(1, { object1.track_id,object2.track_id,r1 });
						break;
					}
				}
			}
		}
}
void CollisionTracker::draw_collision_points(cv::Mat& frame)
{
		for (const auto& collision : collision_points) {
			cv::putText(frame, to_string(collision.track_id_1) + "," + to_string(collision.track_id_2), collision.collision_point, 0, 1, {255,0,20}, 2);
			cv::circle(frame, collision.collision_point, 4, { 255,0,0 }, -1);
		}
}
inline bool CollisionTracker::isInsideCircle(cv::Point2f pt1, cv::Point2f pt2, float radius)
{
	if (sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2)) < radius)
		return true;
	return false;
}
bool CollisionTracker::track_stoped(std::vector<Track>& track) {
	if (track.size() < 10) return false;

	if (get_distance(rect_to_point(track[track.size() - 1].box), rect_to_point(track[0].box)) < 5)return true;
	else
		return false;
}
void CollisionTracker::add_tracks(std::vector<Track>& tracks)
{
	for (const auto& track : tracks) {
		_Tracks.track_history[track.track_id].push_back(track);
		_Tracks.track_frames[frameNumber].push_back(track.track_id);
	}
}