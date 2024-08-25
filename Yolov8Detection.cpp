
#include "Yolov8Detection.h"


Yolov8Detection::~Yolov8Detection()
{
	if (_OrtSession != nullptr) {
		
		delete _OrtSession;
		_OrtSession = nullptr;
	}
}

bool Yolov8Detection::ReadModel(std::string modelPath)
{
	modelInfo.modelPath = std::wstring(modelPath.begin(), modelPath.end());
	_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	_sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
	_OrtSession=new Session(_OrtEnv,modelInfo.modelPath.c_str(), _sessionOptions);
	if (_OrtSession == nullptr)
	{
		return false;
	}

	AllocatorWithDefaultOptions allocator;
	_inputNodesNum = _OrtSession->GetInputCount();
	 _inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
	_inputNodeNames.push_back(_inputName.get());
	TypeInfo typeInfo = _OrtSession->GetInputTypeInfo(0);
	auto input_tensor_info= typeInfo.GetTensorTypeAndShapeInfo();
	_inputNodeTensorType = input_tensor_info.GetElementType();
	_inputTensorShape = input_tensor_info.GetShape();
	modelInfo._netChannels = _inputTensorShape[1];
	modelInfo._netHeight = _inputTensorShape[2];
	modelInfo._netWidth = _inputTensorShape[3];

	_outputNodeNum=_OrtSession->GetOutputCount();
	_outputNames = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
	_outputNodeNames.push_back(_outputNames.get());
	TypeInfo outputTypeInfo = _OrtSession->GetOutputTypeInfo(0);
	auto output_tensor_info = outputTypeInfo.GetTensorTypeAndShapeInfo();
	_outputNodeTensorType= output_tensor_info.GetElementType();
	_outputTensorShape= output_tensor_info.GetShape();
	_inputTensorLength=modelInfo._netChannels*modelInfo._netHeight*modelInfo._netWidth;


	return true;
}

bool Yolov8Detection::ReadModel(std::string modelPath, bool useCUDA)
{
		auto providers = Ort::GetAvailableProviders();
		auto i = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");
		modelInfo.modelPath = std::wstring(modelPath.begin(), modelPath.end());
		if (useCUDA == true && i != providers.end()) {
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = 0;
			_sessionOptions.SetExecutionMode(ORT_PARALLEL);
			cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
			_sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
			_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

		}
		else {
			_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
			_sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
		}
			_OrtSession = new Session(_OrtEnv, modelInfo.modelPath.c_str(), _sessionOptions);
			if (_OrtSession == nullptr)
			{
				return false;
			}
			AllocatorWithDefaultOptions allocator;
			_inputNodesNum = _OrtSession->GetInputCount();
			_inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
			_inputNodeNames.push_back(_inputName.get());
			TypeInfo typeInfo = _OrtSession->GetInputTypeInfo(0);
			auto input_tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
			_inputNodeTensorType = input_tensor_info.GetElementType();
			_inputTensorShape = input_tensor_info.GetShape();
			modelInfo._netChannels = _inputTensorShape[1];
			modelInfo._netHeight = _inputTensorShape[2];
			modelInfo._netWidth = _inputTensorShape[3];
			_outputNodeNum = _OrtSession->GetOutputCount();
			_outputNames = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
			_outputNodeNames.push_back(_outputNames.get());
			TypeInfo outputTypeInfo = _OrtSession->GetOutputTypeInfo(0);
			auto output_tensor_info = outputTypeInfo.GetTensorTypeAndShapeInfo();
			_outputNodeTensorType = output_tensor_info.GetElementType();
			_outputTensorShape = output_tensor_info.GetShape();
			_inputTensorLength = modelInfo._netChannels * modelInfo._netHeight * modelInfo._netWidth;
			if (useCUDA) {
				size_t input_tensor_length = std::accumulate(_inputTensorShape.begin(), _inputTensorShape.end(),1,std::multiplies<int>());
				float* temp = new float[input_tensor_length];
				std::vector<Ort::Value> input_tensor;
				std::vector<Ort::Value> output_tensor;
				input_tensor.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, temp, input_tensor_length,
					_inputTensorShape.data(), _inputTensorShape.size()));
				for (int i = 0; i < 10; i++) {
					output_tensor = _OrtSession->Run(Ort::RunOptions{ nullptr },
						_inputNodeNames.data(),
						input_tensor.data(),	
						_inputNodeNames.size(),
						_outputNodeNames.data(),
						_outputNodeNames.size());
				}
				delete[] temp;
			}
			return true;
}

float* Yolov8Detection::Preprocess(cv::Mat& img)
{
	cv::Mat image;
	cv::resize(img, image, cv::Size(modelInfo._netHeight, modelInfo._netWidth));
	int netHeight = modelInfo._netHeight;
	int netWidth = modelInfo._netWidth;
	float* output = new float[_inputTensorLength];
	cv::Mat channels[3];
	image.convertTo(image, CV_32FC3, 1.0 / 255.0);
	cv::split(image, channels);
	std::memcpy(output, channels[2].ptr<float>(), netHeight * netWidth * sizeof(float));
	std::memcpy(output + netHeight * netWidth, channels[1].ptr<float>(), netHeight * netWidth * sizeof(float));
	std::memcpy(output + (netHeight * netWidth) * 2, channels[0].ptr<float>(), netHeight * netWidth * sizeof(float));
	return output;
}

std::vector<Result> Yolov8Detection::Detect(cv::Mat& image)
{
	_imageSize = image.size();
	float* blob = Preprocess(image);
	std::vector<Ort::Value> inputTensors;
	inputTensors.push_back(Ort::Value::CreateTensor<float>(
		_OrtMemoryInfo, blob, _inputTensorLength,
		_inputTensorShape.data(), _inputTensorShape.size()
	));
	std::vector<Ort::Value> outputTensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
		_inputNodeNames.data(), 
		inputTensors.data(),
		_inputNodeNames.size(),
		_outputNodeNames.data(),
		_outputNodeNames.size());
	delete[] blob;
	std::vector<Result> results=Postprocess(outputTensors);
	return results;
}

std::vector<Result> Yolov8Detection::Postprocess(std::vector<Ort::Value>& output)
{
std::vector<Result> results;
std::vector<cv::Rect> boxes;
std::vector<int> classIds;
std::vector<float> confidences;
cv::Mat l_Mat = cv::Mat(_outputTensorShape[1], _outputTensorShape[2], CV_32FC1, (void*)output[0].GetTensorData<float>());
cv::Mat l_Mat_t = l_Mat.t();
int numOfClasses =l_Mat_t.cols - 4;
int elements_inBatch = int(_outputTensorShape[0] * _outputTensorShape[1] * _outputTensorShape[2]);
for (int n_row = 0; n_row < l_Mat_t.rows; n_row++) {
	Result result;
	float* values = l_Mat_t.ptr<float>(n_row);
	float objConf;
	int classId;
	getBestClassInfo(values, numOfClasses, objConf, classId);
	if (objConf > CONFIDENCE_THRESHOLD) {
		float centerX = (values[0]) * (float)_imageSize.width / modelInfo._netWidth;
		float centerY = (values[1]) * (float)_imageSize.height / modelInfo._netHeight;
		float width = (values[2]) * (float)_imageSize.width / modelInfo._netWidth;
		float height = (values[3]) * (float)_imageSize.height / modelInfo._netHeight;
		float left = centerX - width / 2;
		float top = centerY - height / 2;
		float confidence = objConf;
		boxes.push_back(cv::Rect(left, top, width, height));
		confidences.push_back(confidence);
		classIds.push_back(classId);
	}
}
std::vector<int> selectedIds;
cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, selectedIds);
for (const auto& index : selectedIds)
{
	results.push_back({ boxes[index],modelInfo.labels[classIds[index]],classIds[index],confidences[index]});
}
return results;
}

void Yolov8Detection::getBestClassInfo(const float* values, const int& numClasses, float& bestConf, int& bestClassId)
{
	bestClassId = 0;
	bestConf = 0;
		for (int i = 0; i < numClasses; i++)
		{
			if (values[i + 4] > bestConf)
			{
				bestConf = values[i + 4];
				bestClassId = i;
			}
		}
}
