#include "Tracker.h"
namespace CustomTracker {
	Tracker::Tracker()
	{
		frameNumber = 0;
	}

	Tracker::~Tracker()
	{
	}

	std::vector<Track> Tracker::track(std::vector<InputTracks>& results)
	{
		std::vector<Track> output;
		if (results.size() == 0)
			return std::vector<Track>();
		frameNumber++;
		std::vector<InputTracks> _hightTr, _lowTr;
		for (const auto& track : results)
		{
			if (track.confidence > confThresh) {
				_highConfidenceTracks.push_back(track);
				_hightTr.push_back(track);
			}
			else {
				_lowConfidenceTracks.push_back(track);
				_lowTr.push_back(track);
			}
		}
		if (!_activeTracks.size()) {
			for (const auto& track : results) {
				Track tr;
				tr.bbox = track.bbox;
				tr.confidence = track.confidence;
				tr.frameNumber = frameNumber;
				tr.label = track.label;
				tr.trackId = lastTrack;
				tr.stillActive = true;
				_activeTracks[++lastTrack].push_back(tr);
				output.push_back(tr);
			}

		}
		else {

			//za high treshold
			std::set<int> used_high;
			for (auto it = _activeTracks.begin(); it != _activeTracks.end(); ) {
				//ako ima vise _activeTracks-a nego rezultata
				if (used_high.size() > _hightTr.size()) {
					//izbrisi samo u slucaju da se nije pojavljiva zadnja 4 frame-a
					auto zadnji = it->second[it->second.size() - 1];
					if (zadnji.frameNumber <= frameNumber - 4) {
						++it; continue;
					}
					_tracksHistory[it->first] = it->second;
					it = _activeTracks.erase(it); // erase returns the iterator following the last removed element
					continue;
				}
				else {
					std::vector<Track>& track = it->second;
					double distanceNewPosition;
					double angle = 0;
					Utils::calculateNextPosition(track, distanceNewPosition, angle);
					int returnedIndex = Utils::closestToPosition(_hightTr, distanceNewPosition, angle, it->second[it->second.size() - 1].bbox, used_high);
					if (returnedIndex == -1) {
						++it;
						continue;
					}
					Track tr;
					tr.bbox = _hightTr[returnedIndex].bbox;
					tr.confidence = _hightTr[returnedIndex].confidence;
					tr.frameNumber = frameNumber; // Assuming `frameNumber` is defined elsewhere
					tr.label = _hightTr[returnedIndex].label;
					tr.trackId = it->first;
					tr.stillActive = true;

					_activeTracks[it->first].push_back(tr);
					output.push_back(tr); // Assuming `output` is defined elsewhere
					used_high.insert(returnedIndex); // Assuming `used` is defined elsewhere

					++it; // Manually increment the iterator only if not erasing
				}
			}
			for (int i = 0; i < _hightTr.size(); i++) {
				if (used_high.find(i) != used_high.end())
					continue;
				Track tr;
				tr.bbox = _hightTr[i].bbox;
				tr.confidence = _hightTr[i].confidence;
				tr.frameNumber = frameNumber;
				tr.label = _hightTr[i].label;
				tr.trackId = lastTrack;
				tr.stillActive = true;
				_activeTracks[++lastTrack].push_back(tr);
				output.push_back(tr);
				used_high.insert(i);
			}
		
		
		}

		
		return output;
	}

	void Tracker::resetTracker()
	{
		frameNumber = 0;
		_tracksHistory.clear();
		_highConfidenceTracks.clear();
		_lowConfidenceTracks.clear();
	}

	void Tracker::predictTracks()
	{

	}


}