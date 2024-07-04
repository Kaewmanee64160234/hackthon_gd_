import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';
import 'camera_viewForHeight.dart';
import 'gallery_view.dart';

enum DetectorViewMode { liveFeed, gallery }

class DetectorViewForHeight extends StatefulWidget {
  DetectorViewForHeight({
    Key? key,
    required this.title,
    required this.onImage,
    this.customPaint,
    this.text,
    this.initialDetectionMode = DetectorViewMode.liveFeed,
    this.initialCameraLensDirection = CameraLensDirection.back,
    this.onCameraFeedReady,
    this.onDetectorViewModeChanged,
    this.onCameraLensDirectionChanged,
  }) : super(key: key);

  final String title;
  final CustomPaint? customPaint;
  final String? text;
  final DetectorViewMode initialDetectionMode;
  final Function(InputImage inputImage) onImage;
  final Function()? onCameraFeedReady;
  final Function(DetectorViewMode mode)? onDetectorViewModeChanged;
  final Function(CameraLensDirection direction)? onCameraLensDirectionChanged;
  final CameraLensDirection initialCameraLensDirection;

  @override
  State<DetectorViewForHeight> createState() => _DetectorViewState();
}

class _DetectorViewState extends State<DetectorViewForHeight> {
  late DetectorViewMode _mode;

  @override
  void initState() {
    _mode = widget.initialDetectionMode;
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return _mode == DetectorViewMode.liveFeed
        ? CameraViewForHeight(
            customPaint: widget.customPaint,
            onImage: widget.onImage,
            onCameraFeedReady: widget.onCameraFeedReady,
            onDetectorViewModeChanged: _onDetectorViewModeChanged,
            initialCameraLensDirection: widget.initialCameraLensDirection,
            onCameraLensDirectionChanged: widget.onCameraLensDirectionChanged,
            onImageCaptured: _onImageCaptured,
          )
        : GalleryView(
            title: widget.title,
            text: widget.text,
            onImage: widget.onImage,
            onDetectorViewModeChanged: _onDetectorViewModeChanged,
          );
  }

  void _onDetectorViewModeChanged() {
    setState(() {
      _mode = _mode == DetectorViewMode.liveFeed
          ? DetectorViewMode.gallery
          : DetectorViewMode.liveFeed;
      if (widget.onDetectorViewModeChanged != null) {
        widget.onDetectorViewModeChanged!(_mode);
      }
    });
  }

  void _onImageCaptured(InputImage inputImage, XFile file) async {
    final inputImage = InputImage.fromFilePath(file.path);
    widget.onImage(inputImage);
  }
}
