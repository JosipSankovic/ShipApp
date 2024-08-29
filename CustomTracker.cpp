#include "CustomTracker.h"

CustomTracker::CustomTracker()
{
    low_score = 0.05;
    high_score = 0.5;
}

std::vector<Track> CustomTracker::update(std::vector<Result> detected_objects, cv::Mat& frame, float conf_thresh,float low_conf_thresh)
{
    std::vector<Track> high_score_boxs,low_score_boxs;
    std::vector<Track> tracks_return;
    fr_num++;
    if (conf_thresh == -1)
        conf_thresh = high_score;
    if (low_conf_thresh == -1)
        low_conf_thresh = low_score;
    for (const auto& object : detected_objects) {
        if (object.confidence > conf_thresh) {
            high_score_boxs.push_back({ object.boundingBox,{0,0},object.confidence,-1,fr_num,object.classId });
        }
        else if(object.confidence>=low_conf_thresh){
            low_score_boxs.push_back({ object.boundingBox,{0,0},object.confidence,-1,fr_num,object.classId });
        }
    }
    //trackId,nextPoint
    std::map<int, cv::Point2f> next_points;
    std::vector<int> inactive_ids;
    //pretrazi sve aktivne id-ove i dodaj ih u next_points kako bi se za njih 
    //pretražila buduæa toèka gdje bi trebao biti objekt iz high_score_boxs
    for (auto& track_pair : tracks) {
        if (!use_object_id(track_pair.first)) {
            if (!active_id(track_pair.first)) 
                inactive_ids.push_back(track_pair.first);
            continue;
        }
        Track& track = track_pair.second[track_pair.second.size()-1];
        //izraèunaj brzinu i sljedeæu poziciju ako je objekt aktivan
        get_speed_vector(track);
        next_points[track_pair.first] = (cv::Point2f)rect_center(track.box);
        cv::circle(frame, next_points[track_pair.first], 3, { 255,255,255 }, -1);
        int from_last_frame = abs(tracks[track_pair .first][tracks[track_pair.first].size() - 1].frameNumber - fr_num);
        double cir = (from_last_frame / (double)track_lost_tresh) * 40;
        cv::circle(frame, next_points[track_pair.first],30+cir, {244,0,10}, 1);

    }
    //remove inactive tracks
    for (const auto& tr_id : inactive_ids) {
        objects.erase(tr_id);
        tracks.erase(tr_id);
    }

    //prolaz kroz sve pronaðene objekte
    for (auto& object : high_score_boxs) {
        cv::Point first_point, second_point;
        first_point = rect_center(object.box);
        float min_distance = 9999;

        //pronaði trasu kojoj je objekt najbliži
        // i da je u radijusu od 50 piksela
        for (auto& point : next_points) {
            second_point = point.second;
            float distance = get_distance(first_point, second_point);
            int from_last_frame = abs(tracks[point.first][tracks[point.first].size() - 1].frameNumber - fr_num);
            double cir = (from_last_frame / (double)track_lost_tresh) * 40;
            if (distance < min_distance&&distance< 30 + cir) {
                min_distance = distance;
                object.track_id = point.first;
            }
        }
        if (object.track_id != -1) {
            next_points.erase(object.track_id);
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

    for (auto& point : next_points) {
        cv::Point first_point, second_point;
        first_point = point.second;
        float min_distance = 99999;
        int from_last_frame = abs(tracks[point.first][tracks[point.first].size() - 1].frameNumber - fr_num);
        double cir = (from_last_frame / (double)track_lost_tresh) * 40;
        int i = 0;
        int id = -1;
        for (auto& object : low_score_boxs) {
            second_point = rect_center(object.box);
            float distance = get_distance(first_point, second_point);
            if (distance < min_distance && distance < 30 + cir) {
                min_distance = distance;
                id = i;
            }
            i++;
        }

        if (id!=-1) {
            Track ob=low_score_boxs[id];
            ob.speed= tracks[point.first][tracks[point.first].size() - 1].speed;
            ob.track_id = point.first;
            tracks[point.first].push_back(ob);
            tracks_return.push_back(ob);
            low_score_boxs.erase(std::remove(low_score_boxs.begin(), low_score_boxs.end(), low_score_boxs[id]), low_score_boxs.end());
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
    if (from_last_frame > track_lost_tresh ||
        (from_last_frame>2&&tracks[track_id].size()<5)) {
        return false;
    }else
        return true;
}
bool CustomTracker::active_id(int track_id)
{
    std::vector<Track>& track = tracks[track_id];
    int from_last_frame = abs(tracks[track_id][tracks[track_id].size() - 1].frameNumber - fr_num);
    if (from_last_frame < track_lost_tresh)return true;
    else return false;
}
