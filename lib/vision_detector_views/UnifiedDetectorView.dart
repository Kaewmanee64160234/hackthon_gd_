import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:hackathon/vision_detector_views/painters/object_detector_painter.dart';
import 'package:hackathon/vision_detector_views/painters/pose_painter.dart';
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
  Map<String, double> distancesInCm = {};

  final double _ballDiameterCm = 23.8;
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
    return Scaffold(
      body: Stack(
        children: [
          // Background camera and detections view

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
          // Measurements display
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 200, // Adjust height as necessary
              padding: EdgeInsets.all(10),
              color: Colors.black.withOpacity(0.7), // Semi-transparent
              child: ListView(
                children: distancesInCm.entries
                    .map((entry) => Text(
                          '${entry.key}: ${entry.value.toStringAsFixed(2)} cm',
                          style: TextStyle(color: Colors.white, fontSize: 16),
                        ))
                    .toList(),
              ),
            ),
          ),
        ],
      ),
    );
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
        calculateSpecificDistances(poses, objects, inputImage);

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

  double calculateEuclideanDistance(
      Point<double> point1, Point<double> point2) {
    return sqrt(pow(point2.x - point1.x, 2) + pow(point2.y - point1.y, 2));
  }

  void calculateSpecificDistances(
      List<Pose> poses, List<DetectedObject> objects, InputImage inputImage) {
    if (poses.isNotEmpty) {
      final pose = poses.first;
      final landmarks = pose.landmarks;

      // Check for the scaling object first
      for (var object in objects) {
        for (var label in object.labels) {
          if (label.text.toLowerCase().contains('ball')) {
            final ballBoundingBox = object.boundingBox;
            final ballDiameterInPixels =
                min(ballBoundingBox.height, ballBoundingBox.width);
            _pixelsPerCm = ballDiameterInPixels / _ballDiameterCm;
            print('Calculated _pixelsPerCm: $_pixelsPerCm ');
            break;
          }
        }
      }

      // Extract landmark coordinates
      Point<double> leftAnkle = Point(landmarks[PoseLandmarkType.leftAnkle]!.x,
          landmarks[PoseLandmarkType.leftAnkle]!.y);
      Point<double> rightAnkle = Point(
          landmarks[PoseLandmarkType.rightAnkle]!.x,
          landmarks[PoseLandmarkType.rightAnkle]!.y);
      Point<double> leftKnee = Point(landmarks[PoseLandmarkType.leftKnee]!.x,
          landmarks[PoseLandmarkType.leftKnee]!.y);
      Point<double> rightKnee = Point(landmarks[PoseLandmarkType.rightKnee]!.x,
          landmarks[PoseLandmarkType.rightKnee]!.y);
      Point<double> leftHip = Point(landmarks[PoseLandmarkType.leftHip]!.x,
          landmarks[PoseLandmarkType.leftHip]!.y);
      Point<double> rightHip = Point(landmarks[PoseLandmarkType.rightHip]!.x,
          landmarks[PoseLandmarkType.rightHip]!.y);
      Point<double> leftShoulder = Point(
          landmarks[PoseLandmarkType.leftShoulder]!.x,
          landmarks[PoseLandmarkType.leftShoulder]!.y);
      Point<double> rightShoulder = Point(
          landmarks[PoseLandmarkType.rightShoulder]!.x,
          landmarks[PoseLandmarkType.rightShoulder]!.y);
      Point<double> nose = Point(landmarks[PoseLandmarkType.nose]!.x,
          landmarks[PoseLandmarkType.nose]!.y);
      Point<double> leftMouth = Point(landmarks[PoseLandmarkType.leftMouth]!.x,
          landmarks[PoseLandmarkType.leftMouth]!.y);
      Point<double> leftEye = Point(landmarks[PoseLandmarkType.leftEye]!.x,
          landmarks[PoseLandmarkType.leftEye]!.y);

      // Calculate distances directly between relevant points
      double distanceLeftAnkleKnee =
          calculateEuclideanDistance(leftAnkle, leftKnee);
      double distanceRightAnkleKnee =
          calculateEuclideanDistance(rightAnkle, rightKnee);
      double distanceLeftKneeHip =
          calculateEuclideanDistance(leftKnee, leftHip);
      double distanceRightKneeHip =
          calculateEuclideanDistance(rightKnee, rightHip);
      double distanceLeftHipShoulder =
          calculateEuclideanDistance(leftHip, leftShoulder);
      double distanceRightHipShoulder =
          calculateEuclideanDistance(rightHip, rightShoulder);
      double distanceNoseShoulder = calculateEuclideanDistance(
          nose, leftShoulder); // Assuming left shoulder as reference
      double distanceMouthEye = calculateEuclideanDistance(leftMouth, leftEye);

      // Calculate average distances for symmetry
      double averageAnkleKnee =
          (distanceLeftAnkleKnee + distanceRightAnkleKnee) / 2;
      double averageKneeHip = (distanceLeftKneeHip + distanceRightKneeHip) / 2;
      double averageHipShoulder =
          (distanceLeftHipShoulder + distanceRightHipShoulder) / 2;

      // Total height estimation
      double totalDistance = averageAnkleKnee +
          averageKneeHip +
          averageHipShoulder +
          distanceNoseShoulder +
          distanceMouthEye;

      // Convert to real-world units
      totalDistance /= _pixelsPerCm;

      // Save distances in cm
      distancesInCm['Left Ankle to Left Knee'] =
          distanceLeftAnkleKnee / _pixelsPerCm;
      distancesInCm['Right Ankle to Right Knee'] =
          distanceRightAnkleKnee / _pixelsPerCm;
      distancesInCm['Left Knee to Left Hip'] =
          distanceLeftKneeHip / _pixelsPerCm;
      distancesInCm['Right Knee to Right Hip'] =
          distanceRightKneeHip / _pixelsPerCm;
      distancesInCm['Left Hip to Left Shoulder'] =
          distanceLeftHipShoulder / _pixelsPerCm;
      distancesInCm['Right Hip to Right Shoulder'] =
          distanceRightHipShoulder / _pixelsPerCm;
      distancesInCm['Nose to Shoulder'] = distanceNoseShoulder / _pixelsPerCm;
      distancesInCm['Mouth to Eye'] = distanceMouthEye / _pixelsPerCm;
      distancesInCm['Total Body Height'] = totalDistance;

      // Print or update state with the total distance
      _personHeightCm = totalDistance;
      print("Estimated height: ${_personHeightCm.toStringAsFixed(2)} cm");
    } else {
      print("No poses or objects detected.");
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
