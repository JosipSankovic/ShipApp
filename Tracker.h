#pragma once
#include<opencv2/opencv.hpp>
#include "./utils/trackerHelper.h"
namespace CustomTracker {
	struct InputTracks {
		int label;
		float confidence;
		cv::Rect bbox;
	};
	struct Track {
		int frameNumber;
		bool stillActive{ true };
		int trackId;
		int label;
		float confidence;
		cv::Rect bbox;

	};
	class Tracker
	{
	public:
		Tracker();
		~Tracker();
	public:

		std::map<int, std::vector<Track>> _activeTracks;
	private:
		float confThresh{ 0.4 };
		int frameNumber{ 0 };
		int lastTrack = 0;
		//trackId,tracks>
		std::map<int,std::vector<Track>> _tracksHistory;
		std::vector<InputTracks> _highConfidenceTracks, _lowConfidenceTracks;
	public:
		std::vector<Track> track(std::vector<InputTracks>& results);
	public:
		void resetTracker();
	private:
		void predictTracks();

	};
}