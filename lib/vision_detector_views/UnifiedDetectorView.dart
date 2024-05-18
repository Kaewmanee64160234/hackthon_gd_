import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:hackathon/vision_detector_views/painters/object_detector_painter.dart';
import 'package:hackathon/vision_detector_views/painters/pose_painter.dart';
import 'package:hackathon/vision_detector_views/painters/segmentation_painter.dart';
import 'package:hackathon/vision_detector_views/utils.dart';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'detector_view.dart';

class UnifiedDetectorView extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _UnifiedDetectorViewState();
}

class _UnifiedDetectorViewState extends State<UnifiedDetectorView> {
  final PoseDetector _poseDetector =
      PoseDetector(options: PoseDetectorOptions());
  ObjectDetector? _objectDetector;

  bool _isBusy = false;
  CustomPaint? _customPaint;
  var _cameraLensDirection = CameraLensDirection.back;

  final double _ballDiameterCm = 24;
  double _pixelsPerCm = 0.0;
  double _personHeightCm = 0.00;
  @override
  void initState() {
    super.initState();
    _initializeDetectors();
  }

  @override
  void dispose() {
    _poseDetector.close();
    _objectDetector?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(children: [
      DetectorView(
        title: 'Unified Detector',
        customPaint: _customPaint,
        text: 'Detection Results',
        onImage: _processImage,
        initialCameraLensDirection: _cameraLensDirection,
        onCameraLensDirectionChanged: (value) => setState(() {
          _cameraLensDirection = value;
        }),
      ),
      Positioned(
          bottom: 0,
          left: 0,
          top: 100,
          child: Container(
              padding: EdgeInsets.all(10),
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Person Height: ${_personHeightCm.toStringAsFixed(2)} cm',
                      style: TextStyle(color: Colors.white, fontSize: 10),
                    ),
                    Text(
                      'Ball Diameter: $_ballDiameterCm cm',
                      style: TextStyle(color: Colors.white, fontSize: 10),
                    ),
                  ]))),
    ]);
  }

  Future<void> _initializeDetectors() async {
    final modelPath = await getAssetPath('assets/ml/object_labeler.tflite');
    final options = LocalObjectDetectorOptions(
        mode: DetectionMode.stream,
        classifyObjects: true,
        multipleObjects: true,
        modelPath: modelPath);
    _objectDetector = ObjectDetector(options: options);
  }

  Future<void> _processImage(InputImage inputImage) async {
    if (!_isBusy) {
      _isBusy = true;
      try {
        // Process image for object detection
        final objects = await _objectDetector!.processImage(inputImage);
        final objectPainter = ObjectDetectorPainter(
          objects,
          inputImage.metadata!.size,
          inputImage.metadata!.rotation,
          _cameraLensDirection,
        );

        // Process image for pose detection
        final poses = await _poseDetector.processImage(inputImage);
        final posePainter = PosePainter(
          poses,
          inputImage.metadata!.size,
          inputImage.metadata!.rotation,
          _cameraLensDirection,
        );

        // Calculate the height of the person
        _calculatePersonHeight(poses, objects, inputImage);

        // Combine the painters
        setState(() {
          _customPaint =
              CustomPaint(painter: CombinedPainter(objectPainter, posePainter));
        });
      } catch (e) {
        print('Error processing image: $e');
      } finally {
        _isBusy = false;
      }
    }
  }

  void _calculatePersonHeight(
      List<Pose> poses, List<DetectedObject> objects, InputImage inputImage) {
    _personHeightCm = 0.0; // Reset height to zero initially

    if (poses.isNotEmpty) {
      final pose = poses.first;
      final landmarks = pose.landmarks;

      // Check if required landmarks are present
      if (landmarks.containsKey(PoseLandmarkType.leftHeel) &&
          landmarks.containsKey(PoseLandmarkType.leftAnkle) &&
          landmarks.containsKey(PoseLandmarkType.leftHip) &&
          landmarks.containsKey(PoseLandmarkType.leftShoulder) &&
          landmarks.containsKey(PoseLandmarkType.leftMouth) &&
          landmarks.containsKey(PoseLandmarkType.leftEye)) {
        final leftToe = landmarks[PoseLandmarkType.leftHeel]!;
        final leftAnkle = landmarks[PoseLandmarkType.leftAnkle]!;
        final leftHip = landmarks[PoseLandmarkType.leftHip]!;
        final leftShoulder = landmarks[PoseLandmarkType.leftShoulder]!;
        final mouthCenter = landmarks[PoseLandmarkType.leftMouth]!;
        final leftEye = landmarks[PoseLandmarkType.leftEye]!;

        // Calculate distances in pixels using absolute values
        final toeToAnkle = (leftAnkle.y - leftToe.y).abs();
        final ankleToHip = (leftHip.y - leftAnkle.y).abs();
        final hipToShoulder = (leftShoulder.y - leftHip.y).abs();
        final shoulderToMouthCenter = (mouthCenter.y - leftShoulder.y).abs();
        final mouthToLeftEye = (leftEye.y - mouthCenter.y).abs();

        bool foundBall = false;

        // Find an object to use as a scale reference, e.g., a known ball size
        for (var object in objects) {
          for (var label in object.labels) {
            if (label.text.toLowerCase().contains('ball')) {
              foundBall = true;
              final ballBoundingBox = object.boundingBox;
              final ballHeightInPixels = ballBoundingBox.height;

              // Calculate the scaling factor (real-world size to pixel size)
              _pixelsPerCm = ballHeightInPixels / _ballDiameterCm;

              // Calculate heights in cm
              final toeToAnkleCm = toeToAnkle / _pixelsPerCm;
              final ankleToHipCm = ankleToHip / _pixelsPerCm;
              final hipToShoulderCm = hipToShoulder / _pixelsPerCm;
              final shoulderToMouthCenterCm =
                  shoulderToMouthCenter / _pixelsPerCm;
              final mouthToLeftEyeCm = mouthToLeftEye / _pixelsPerCm;

              // Summing up the parts to get the total height
              _personHeightCm = toeToAnkleCm +
                  ankleToHipCm +
                  hipToShoulderCm +
                  shoulderToMouthCenterCm +
                  (mouthToLeftEyeCm *
                      2); // Assuming distance to top of head is similar to eye to mouth distance

              break; // Break after calculating with the found ball
            }
          }
          if (foundBall) break; // Exit outer loop if ball found
        }
      }
    }
    setState(() {});
  }

  Map<String, dynamic> cropImageFromBoundingBoxAndCalculateHeight(
      Uint8List imageData, Rect boundingBox) {
    img.Image originalImage = img.decodeImage(imageData)!;
    img.Image croppedImage = img.copyCrop(
      originalImage,
      boundingBox.left.toInt(),
      boundingBox.top.toInt(),
      boundingBox.width.toInt(),
      boundingBox.height.toInt(),
    );

    // Calculate the height in pixels of the bounding box
    int boundingBoxHeightInPixels = boundingBox.height.toInt();

    return {
      'croppedImage': croppedImage,
      'boundingBoxHeightInPixels': boundingBoxHeightInPixels
    };
  }
}

class CombinedPainter extends CustomPainter {
  final ObjectDetectorPainter objectPainter;
  final PosePainter posePainter;

  CombinedPainter(this.objectPainter, this.posePainter);

  @override
  void paint(Canvas canvas, Size size) {
    objectPainter.paint(canvas, size);
    posePainter.paint(canvas, size);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
