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
	_ShipAppState.videoLoaded = false;
	video.open(_path.toStdString());
	if (!video.isOpened())
		return;
	_ShipAppState.videoLoaded = true;
	//uzimanje info u vezi videa
	_VideoInfo.totalFrames = video.get(cv::CAP_PROP_FRAME_COUNT);
	_VideoInfo.fps = video.get(cv::CAP_PROP_FPS);
	//postavljanje na prvi frame
	_VideoInfo.frameNumber = 0;
	video.set(cv::CAP_PROP_POS_FRAMES, _VideoInfo.frameNumber);
	tracker.resetTracker();
	//dohvacanje prvog frame-a i prikazivanje
	video >> frame;
	_VideoInfo.frameNumber++;
	if (!_ShipAppState.modelLoaded)
		_ShipAppState.modelLoaded = detection_model.ReadModel("Model/best (8).onnx", ui.check_CUDA->isChecked());
	showImage(frame);
}

void ShipApp::playVideo()
{
	std::thread([&] {
		while (_ShipAppState.videoPlaying &&
			_VideoInfo.frameNumber < _VideoInfo.totalFrames)
		{
			for (int i = 0; i < 10; i++) {
				video.grab();
				_VideoInfo.frameNumber++;
			}
			if (!_ShipAppState.detectionRunning) {
				_ShipAppState.detectionRunning = true;
				video.retrieve(frame);
				detectAndTrack();
				showImage(frame);
				_ShipAppState.detectionRunning = false;
			}
		}
	}).detach();
}

void ShipApp::detectAndTrack()
{
	auto start = std::chrono::system_clock::now();
	results = detection_model.Detect(frame);
	tracker.update_track(results, _CONFIDENCE_THRESHOLD);
	tracker.find_collision(80);
	tracker.draw_collision_points(frame);
	tracker.show_track(frame, 10000);
	results.clear();
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cv::putText(frame, "MS:" + std::to_string(elapsed.count()), { 20,80 }, 0, 2, { 255,255,10 });
}  

void ShipApp::showImage(cv::Mat& img)
{
	ui.video->setPixmap(getPixmap(img).scaled(ui.video->size(), Qt::KeepAspectRatio, Qt::TransformationMode::SmoothTransformation));
}

void ShipApp::on_btn_Stop_clicked()
{
	_ShipAppState.videoPlaying = false;
	int i = 0;
	while (_ShipAppState.detectionRunning&&i++<5)
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
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
	if (!_ShipAppState.videoLoaded|| _ShipAppState.detectionRunning||_ShipAppState.videoPlaying)
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
