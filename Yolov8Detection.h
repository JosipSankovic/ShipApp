
#pragma once
#include <onnxruntime_cxx_api.h>
#include<numeric>
#include<opencv2/opencv.hpp>
using namespace Ort;
struct Result {
	cv::Rect boundingBox;
	std::string label;
	int classId;
	float confidence;
};
class Yolov8Detection
{
	struct {
		std::wstring modelPath;
		int _netWidth{ 0 };
		int _netHeight{ 0 };
		int _netChannels{ 0 };
		int numOfClasses;
		std::vector<std::string> labels = {"Ferry","Boat"};
	}modelInfo;

public:
	Yolov8Detection() :_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {};
	~Yolov8Detection();

public:
	float CONFIDENCE{ 0.05 };
	float NMS_THRESHOLD{ 0.3 };

public:
	bool ReadModel(std::string modelPath);
	bool ReadModel(std::string modelPath,bool useCUDA);
	std::vector<Result> Detect(cv::Mat& image);
private:
	float* Preprocess(cv::Mat& img);
	std::vector<Result> Postprocess(std::vector<Ort::Value>& output);
	void getBestClassInfo(const cv::Mat& p_Mat, const int& numClasses,
		float& bestConf, int& bestClassId);

private:
	Env _OrtEnv;
	RunOptions runOtions;
	SessionOptions _sessionOptions;
	Session* _OrtSession=nullptr;
	MemoryInfo _OrtMemoryInfo;

private:
	std::shared_ptr<char> _inputName, _outputNames;
	ONNXTensorElementDataType _inputNodeTensorType;
	ONNXTensorElementDataType _outputNodeTensorType;
	std::vector<char*> _inputNodeNames;
	std::vector<char*> _outputNodeNames;
	std::vector<int64_t> _inputTensorShape;
	std::vector<int64_t> _outputTensorShape;
	size_t inputTensorSize{ 0 };
	size_t _inputTensorLength{ 0 };
	int _inputNodesNum{0 };
	int _outputNodeNum{ 0 };
	cv::Size _imageSize;
};

