#include "CustomTracker.h"

CustomTracker::CustomTracker()
{
    low_score = 0.2;
    high_score = 0.2;
}

std::vector<Track> CustomTracker::update(std::vector<Result> detected_objects,float conf_thresh)
{
    std::vector<Track> high_score_boxs;
    std::vector<Track> low_score_boxs;
    std::vector<Track> tracks_return;
    fr_num++;
    if (conf_thresh == -1)
        conf_thresh = high_score;
    for (const auto& object : detected_objects) {
        if (object.confidence > conf_thresh) {
            high_score_boxs.push_back({ object.boundingBox,{0,0},object.confidence,-1,fr_num,object.classId });
        }
    }
    //trackId,nextPoint
    std::map<int, cv::Point2f> next_points;
    std::vector<int> inactive_ids;
    for (auto& track_pair : tracks) {
        if (!use_object_id(track_pair.first)) {
            if (!active_id(track_pair.first)) 
                inactive_ids.push_back(track_pair.first);
            continue;
        }
        Track& track = track_pair.second[track_pair.second.size()-1];
        get_speed_vector(track);
        next_points[track_pair.first] = (cv::Point2f)rect_center(track.box);
    }
    //remove inactive tracks
    for (const auto& tr_id : inactive_ids) {
        objects.erase(tr_id);
        tracks.erase(tr_id);
    }
    
    for (auto& object : high_score_boxs) {
        cv::Point first_point, second_point;
        first_point = rect_center(object.box);
        float min_distance = 9999;
        int id = -1;
        for (auto& point : next_points) {
            second_point = point.second;
            float distance = get_distance(first_point, second_point);
            if (distance < min_distance) {
                min_distance = distance;
                id = point.first;
            }
        }
        if (min_distance < 50)
            object.track_id = id;
            if (object.track_id != -1) {
                object.track_id = id;
                next_points.erase(id);
            }
        
    }
    for (auto& object : high_score_boxs) {
        if (object.track_id != -1) {
            object.speed = tracks[object.track_id][tracks[object.track_id].size() - 1].speed;
            tracks[object.track_id].push_back(object);
            if(tracks[object.track_id].size()>3)
                tracks_return.push_back(object);
        }
        else {
            object.track_id = track_id_counter;
            tracks[track_id_counter].push_back(object);
            objects.emplace(track_id_counter, KalmanState(rect_center(object.box)));
            if (tracks[object.track_id].size() > 3)
            tracks_return.push_back(object);
            track_id_counter++;
        }

    }

    return tracks_return;
}

void CustomTracker::get_speed_vector(Track& track)
{
    cv::Point z_point = rect_center(track.box);
    Eigen::Matrix<float, 2, 1> z(z_point.x, z_point.y);
    int dt = 1;
    KalmanState& KS = objects[track.track_id];
    KS.A(0, 2) = 1;
    KS.A(1, 3) = 1;
    KS.State = KS.A * KS.State;
    KS.P = KS.A.cast<double>() * KS.P * KS.A.transpose().cast<double>() + KS.Q;

    Eigen::Matrix<float, 2, 1> y = z - KS.H * KS.State;
    Eigen::Matrix<double, 2, 2> S = KS.H.cast<double>() * KS.P * KS.H.transpose().cast<double>() + KS.R;
    Eigen::Matrix<double, 4, 2> K = KS.P * KS.H.transpose().cast<double>() * S.inverse();


    KS.State = KS.State + K.cast<float>() * y;
    KS.P = (Eigen::Matrix<double, 4, 4>::Identity() - K * KS.H.cast<double>()) * KS.P;
    cv::Point2f speed_vector;
    speed_vector.x = KS.State(2, 0);
    speed_vector.y = KS.State(3, 0);
    track.speed.x = KS.State(2, 0);
    track.speed.y = KS.State(3, 0);
    track.box.x= KS.State(0, 0)-track.box.width/2+KS.State(2,0);
    track.box.y= KS.State(1, 0)-track.box.height/2+KS.State(3,0);
}

bool CustomTracker::use_object_id(int track_id)
{
    std::vector<Track>& track = tracks[track_id];
    int from_last_frame = abs(tracks[track_id][tracks[track_id].size() - 1].frameNumber - fr_num);
    // nije aktivan u slucaju da je proslo predugo od zadnjeg prikaza
    // nije aktivan u slucaju da je malen a proslo je neko vrijeme od zadnje detekcije
    if (from_last_frame > 50 ||
        (from_last_frame>2&&tracks[track_id].size()<4)) {
        return false;
    }else
        return true;
}
bool CustomTracker::active_id(int track_id)
{
    std::vector<Track>& track = tracks[track_id];
    int from_last_frame = abs(tracks[track_id][tracks[track_id].size() - 1].frameNumber - fr_num);
    if (from_last_frame < 50)return true;
    else return false;
}
