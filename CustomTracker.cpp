#include "CustomTracker.h"

CustomTracker::CustomTracker()
{
    low_score = 0.1;
    high_score = 0.5;
    track_lost_tresh = 60;
    DISTANCE_THRESH = 20;
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
            cv::rectangle(frame, object.boundingBox, { 0,255,0 }, 1);
        }
        else if(object.confidence>=low_conf_thresh){
            low_score_boxs.push_back({ object.boundingBox,{0,0},object.confidence,-1,fr_num,object.classId });
            cv::rectangle(frame, object.boundingBox, { 0,0,255 }, 1);
        }
    }
    //trackId,nextPoint
    std::map<int, cv::Point2f> next_points;
    std::vector<int> inactive_ids;
    //pretrazi sve aktivne id-ove i dodaj ih u next_points kako bi se za njih 
    //pretra�ila budu�a to�ka gdje bi trebao biti objekt iz high_score_boxs
    for (auto& track_pair : tracks) {
        if (!use_object_id(track_pair.first)|| !active_id(track_pair.first)) {
            inactive_ids.push_back(track_pair.first);
            continue;
        }
        Track& track = track_pair.second.first;
        //izra�unaj brzinu i sljede�u poziciju ako je objekt aktivan
        get_speed_vector(track);
        next_points[track_pair.first] = (cv::Point2f)rect_center(track.box);
        cv::circle(frame, next_points[track_pair.first], 3, { 255,255,255 }, -1);
        int from_last_frame = abs(track.last_active_frNum - fr_num);
        double cir = (from_last_frame / (double)track_lost_tresh) * (DISTANCE_THRESH+10) +
            get_speed(track.speed*5);
        cv::circle(frame, next_points[track_pair.first], DISTANCE_THRESH+cir , {244,0,10}, 1);

    }
    for (const auto& tr_id : inactive_ids) {
        tracks.erase(tr_id);
    }
    //prolaz kroz sve prona�ene objekte
    for (auto& object : high_score_boxs) {
        cv::Point first_point, second_point;
        first_point = rect_center(object.box);
        float min_distance = std::numeric_limits<float>().max();
        //prona�i trasu kojoj je objekt najbli�i
        // i da je u odre�enom radijusu
        for (auto& point : next_points) {
            second_point = point.second;
            float distance = get_distance(first_point, second_point);
            int from_last_frame = abs(tracks[point.first].first.last_active_frNum - fr_num);
            double cir = (from_last_frame / (double)track_lost_tresh) * (10+DISTANCE_THRESH)
                +get_speed(tracks[point.first].first.speed*5);
            if (distance < min_distance&&distance< DISTANCE_THRESH + cir) {
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
            object.speed = tracks[object.track_id].first.speed;
            tracks[object.track_id].first = object;
            tracks[object.track_id].first.count++;
            if(tracks[object.track_id].first.count>3)
                tracks_return.push_back(object);
        }
        else {
            object.track_id = track_id_counter;
            tracks.emplace(track_id_counter, std::pair<Track, KalmanState>(object,KalmanState(rect_center(object.box))));
            tracks[track_id_counter].first.count++;
            if (tracks[object.track_id].first.count > 3)
            tracks_return.push_back(object);
            track_id_counter++;
        }

    }

    //udaljenost,low_score_index,next_point_index
    std::vector<std::tuple<float, int, int>> distance_list;

    //dodaju se sve koje zadovoljavaju uvjet
    for (int i = 0; i < low_score_boxs.size(); ++i) {
        cv::Point first_point = rect_center(low_score_boxs[i].box);
        for (auto& point : next_points) {
            int next_point_index = point.first;
            cv::Point second_point = point.second;
            float distance = get_distance(first_point, second_point);
            int from_last_frame = abs(tracks[point.first].first.last_active_frNum - fr_num);
            double cir = (from_last_frame / (double)track_lost_tresh) * (10 + DISTANCE_THRESH)
                + get_speed(tracks[point.first].first.speed * 5);

            if (distance < DISTANCE_THRESH + cir) {
                distance_list.push_back(std::make_tuple(distance, i, next_point_index));
            }
        }
    }
    //sortiranje od najmanje do najvece
    std::sort(distance_list.begin(), distance_list.end(),
        [](const std::tuple<float, int, int>& a, const std::tuple<float, int, int>& b) {
        return std::get<0>(a) < std::get<0>(b);
    });
    std::set<int> assigned_points;
    std::set<int> assigned_detection;
    //dodajemo najbli�e vrijednosti pripadaju�im next_pointsima
    for (const auto& entry : distance_list) {
        float distance = std::get<0>(entry);
        int low_score_box_index = std::get<1>(entry);
        int next_point_index = std::get<2>(entry);

        if (assigned_points.find(next_point_index) != assigned_points.end()||assigned_detection.find(low_score_box_index)!=assigned_detection.end())
            continue;
        low_score_boxs[low_score_box_index].track_id = next_point_index;
        assigned_points.insert(next_point_index);
        assigned_detection.insert(low_score_box_index);
        low_score_boxs[low_score_box_index].speed = tracks[next_point_index].first.speed;
        tracks[next_point_index].first = low_score_boxs[low_score_box_index];
        tracks[next_point_index].first.count++;
        if (tracks[next_point_index].first.count > 3) {
            tracks_return.push_back(low_score_boxs[low_score_box_index]);
        }
    }
   
    return tracks_return;
}

void CustomTracker::get_speed_vector(Track& track)
{
    cv::Point z_point = rect_center(track.box);
    Eigen::Matrix<float, 2, 1> z(z_point.x, z_point.y);
    int dt = 1;
    KalmanState& KS = tracks[track.track_id].second;
    KS.A(0, 2) = 1;
    KS.A(1, 3) = 1;
    KS.State = KS.A * KS.State;
    KS.P = KS.A.cast<double>() * KS.P * KS.A.transpose().cast<double>() + KS.Q;

    Eigen::Matrix<float, 2, 1> y = z - KS.H * KS.State;
    Eigen::Matrix<double, 2, 2> S = KS.H.cast<double>() * KS.P * KS.H.transpose().cast<double>() + KS.R;
    Eigen::Matrix<double, 4, 2> K = KS.P * KS.H.transpose().cast<double>() * S.inverse();


    KS.State = KS.State + K.cast<float>() * y;
    KS.P = (Eigen::Matrix<double, 4, 4>::Identity() - K * KS.H.cast<double>()) * KS.P;

    track.speed.x = KS.State(2, 0);
    track.speed.y = KS.State(3, 0);
    track.box.x= KS.State(0, 0)-track.box.width/2+KS.State(2,0);
    track.box.y= KS.State(1, 0)-track.box.height/2+KS.State(3,0);
}
bool CustomTracker::use_object_id(int track_id)
{
    Track& track = tracks[track_id].first;
    int from_last_frame = fr_num-track.last_active_frNum;
    if (from_last_frame>2&&track.count<4) {
        return false;
    }else
        return true;
}
bool CustomTracker::active_id(int track_id)
{
    Track& track = tracks[track_id].first;
    int from_last_frame = fr_num-track.last_active_frNum;
    if (from_last_frame < track_lost_tresh)return true;
    else return false;
}
