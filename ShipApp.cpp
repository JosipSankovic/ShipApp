#include "ShipApp.h"
ShipApp::ShipApp(QWidget* parent)
	: QMainWindow(parent), _registry(QSettings::UserScope, "ShipApp")
{
	ui.setupUi(this);

}

ShipApp::~ShipApp()
{

}

QPixmap ShipApp::getPixmap(cv::Mat& Img)
{
	QImage img = QImage((const uchar*)(Img.data),
		Img.cols, Img.rows, QImage::Format_RGB888);
	return QPixmap::fromImage(img.rgbSwapped());
}

void ShipApp::loadVideo()
{
	cap.open(_path.toStdString());
	if (!cap.isOpened())
		return;
	_ShipAppState.videoLoaded = true;
	//uzimanje info u vezi videa
	_VideoInfo.totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
	_VideoInfo.fps = cap.get(cv::CAP_PROP_FPS);
	//postavljanje na prvi frame
	_VideoInfo.frameNumber = 0;
	cap.set(cv::CAP_PROP_POS_FRAMES, _VideoInfo.frameNumber);
	//dohvacanje prvog frame-a i prikazivanje
	cap >> frame;
	_VideoInfo.frameNumber++;
	tracker.resetTracks();
	showImage(frame);
}

void ShipApp::playVideo()
{
//	writer.open("output.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), _VideoInfo.fps/2, cv::Size(frame.cols, frame.rows));
	if (!_ShipAppState.modelLoaded)
		_ShipAppState.modelLoaded = detection.ReadModel("Model/best (3).onnx", ui.check_CUDA->isChecked());
	std::thread([&] {
		while (_ShipAppState.videoPlaying &&
			_VideoInfo.frameNumber < _VideoInfo.totalFrames)
		{
			for (int i = 0; i < 8; i++)
				cap.grab();
			_VideoInfo.frameNumber++;

			if (!_ShipAppState.detectionRunning) {
				_ShipAppState.detectionRunning = true;
				cap.retrieve(frame);

				detectAndTrack();
				showImage(frame);
			//	writer.write(frame);
				_ShipAppState.detectionRunning = false;

			}


		}
		//writer.release();
	}).detach();
}

void ShipApp::detectAndTrack()
{
	auto start = std::chrono::high_resolution_clock::now();
	// detektiraj objekte
	results = detection.Detect(frame);
	// dodjeli svakom objektu Id
	tracker.track(results, _CONFIDENCE_THRESHOLD, _VideoInfo.frameNumber);
	//nacrtaj trace od svakog detektiranog id-a
	tracker.drawPastPoints(frame, _VideoInfo.frameNumber,80);
	auto end = std::chrono::high_resolution_clock::now();
	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cv::putText(frame, to_string(dur.count())+"fr:"+to_string(_VideoInfo.frameNumber), {20,100},1, 5, {0,5,0},3);
	results.clear();
}  

void ShipApp::showImage(cv::Mat& img)
{
	ui.video->setPixmap(getPixmap(img).scaled(ui.video->size(), Qt::KeepAspectRatio, Qt::TransformationMode::SmoothTransformation));
}

void ShipApp::on_btn_Stop_clicked()
{
	_ShipAppState.videoPlaying = false;
	while (_ShipAppState.detectionRunning)
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	//writer.release();
}

void ShipApp::on_btn_LoadVideo_clicked()
{
	QString lastFileName = _registry.value("lastFileName").toString();
	QString fileName = QFileDialog::getOpenFileName(this, tr("Select video"), lastFileName, tr("Video Files (*.avi *.mp4)"));
	if (fileName.isEmpty())
		return;
	_registry.setValue("lastFileName", fileName);
	_path = fileName;
	on_btn_Stop_clicked();
	loadVideo();
}
void ShipApp::resizeEvent(QResizeEvent* event)
{
	if (frame.empty())
		return;
	showImage(frame);

}
void ShipApp::on_btn_Start_clicked() {
	if (!_ShipAppState.videoLoaded)
		return;
	if (_ShipAppState.detectionRunning)
		return;
	_ShipAppState.videoPlaying = true;
	std::thread([&] {
		playVideo();
	}).detach();

}
void ShipApp::closeEvent(QCloseEvent* event)
{
	_ShipAppState.videoPlaying = false;
	while (_ShipAppState.detectionRunning)
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));


	QMainWindow::closeEvent(event);
}
