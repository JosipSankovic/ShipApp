#pragma once
#include "Yolov8Detection.h"
#include "Tracker.h"


std::vector<CustomTracker::InputTracks> YoloToTracker(std::vector<Result> yoloResults) {
	std::vector<CustomTracker::InputTracks> output;
	for (const auto& yolo : yoloResults) {
		output.push_back({ yolo.classId,yolo.confidence,yolo.boundingBox });
	}
	return output;
}