#include "ShipApp.h"
ShipApp::ShipApp(QWidget *parent)
    : QMainWindow(parent), _registry(QSettings::UserScope, "ShipApp")
{
    ui.setupUi(this);
    
}

ShipApp::~ShipApp()
{}

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
    //dohva�anje prvog frame-a i prikazivanje
	cap >> frame;
    _VideoInfo.frameNumber++;
   // tracker.resetTracks();
    //_tracker.resetTracker();
    showImage(frame);
}

void ShipApp::playVideo()
{
    //writer.open("output.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), _VideoInfo.fps/2, cv::Size(frame.cols, frame.rows));
    if(!_ShipAppState.modelLoaded)
        _ShipAppState.modelLoaded = detection.ReadModel("Model/ShipModel.onnx",ui.check_CUDA->isChecked());
    std::thread([&] {
    while (_ShipAppState.videoPlaying &&
        _VideoInfo.frameNumber<_VideoInfo.totalFrames)
    {
        //u slu�aju nevaljanog frame-a
        if (!cap.grab())
            continue;
        if (!cap.grab())
            continue;
        if (!cap.grab())
            continue;
        _VideoInfo.frameNumber++;
       
        if (!_ShipAppState.detectionRunning) {
            _ShipAppState.detectionRunning = true;
            cap.retrieve(frame);
            
                
                detectAndTrack();
                showImage(frame);
                
                _ShipAppState.detectionRunning = false;

        }
       

	}   
            }).detach();
}

void ShipApp::detectAndTrack()
{
    
    auto start = std::chrono::high_resolution_clock::now();
    results = detection.Detect(frame);
    tracker.track(results, _CONFIDENCE_THRESHOLD,_VideoInfo.frameNumber);
    tracker.drawPastPoints(frame,_VideoInfo.frameNumber);
   /* std::vector<CustomTracker::InputTracks> track;
    Utils::yoloToTrackerInput(results, track);
    auto start = std::chrono::high_resolution_clock::now();
    auto tracks=_tracker.track(track);
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cv::putText(frame, to_string(dur.count()), { 20,100 }, 1, 5, { 100,255,0 });
    for (const auto& tr : tracks) {
        cv::rectangle(frame, tr.bbox, { 255,0,0 }, 3,5);
        cv::putText(frame, to_string(tr.trackId),Utils::boxToPoint(tr.bbox),1,3,{0,255,100});
    }*/
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cv::putText(frame, to_string(dur.count()), { 20,100 }, 1, 5, { 100,255,0 });
   /* for (const auto& tr : _tracker._activeTracks) {
        std::vector<cv::Point> points;
        auto tracks=tr.second;
        for (const auto& track : tracks) {
            points.push_back(Utils::boxToPoint(track.bbox));
        }
        cv::polylines(frame, points, false, { 255,0,0 },4);
    }*/
    /*writer.write(frame);*/
    results.clear();
}

void ShipApp::showImage(cv::Mat& img)
{
    ui.video->setPixmap(getPixmap(img).scaled(ui.video->size(),Qt::KeepAspectRatio,Qt::TransformationMode::SmoothTransformation));
}

void ShipApp::on_btn_Stop_clicked()
{
    _ShipAppState.videoPlaying = false;
    while(_ShipAppState.detectionRunning)
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //writer.release();
}

void ShipApp::on_btn_LoadVideo_clicked()
{
    QString lastFileName=_registry.value("lastFileName").toString();
    QString fileName= QFileDialog::getOpenFileName(this, tr("Select video"), lastFileName, tr("Video Files (*.avi *.mp4)"));
    if(fileName.isEmpty())
		return;
    _registry.setValue("lastFileName", fileName);
	_path = fileName;
    on_btn_Stop_clicked();
    loadVideo();
}
void ShipApp::resizeEvent(QResizeEvent* event)
{   if(frame.empty())
		return;
    showImage(frame);

}
void ShipApp::on_btn_Start_clicked() {
    if (!_ShipAppState.videoLoaded)
        return;
    if(_ShipAppState.detectionRunning)
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
