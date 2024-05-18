import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
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
    return Stack(
      children: [
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
          right: 0,
          top: 100,
          child: Container(
            height: 200, // Adjust as necessary
            padding: EdgeInsets.all(10),
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
        //show Height(
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          top: 100,
          child: Container(
            height: 200, // Adjust as necessary
            padding: EdgeInsets.all(10),
            child: Text(
              'Person Height: ${_personHeightCm.toStringAsFixed(2)} cm',
              style: TextStyle(color: Colors.white, fontSize: 16),
            ),
          ),
        ),
      ],
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
    if (poses.isNotEmpty && objects.isNotEmpty) {
      final pose = poses.first;
      final landmarks = pose.landmarks;

      Map<String, List<PoseLandmarkType>> landmarkPairs = {
        'Ankle to Knee': [PoseLandmarkType.leftHeel, PoseLandmarkType.leftKnee],
        'Knee to Hip': [PoseLandmarkType.leftKnee, PoseLandmarkType.leftHip],
        'Hip to Shoulder': [
          PoseLandmarkType.leftHip,
          PoseLandmarkType.leftShoulder
        ],
        'Shoulder to Ear': [
          PoseLandmarkType.leftShoulder,
          PoseLandmarkType.leftEar
        ],
        'Mouth to Eye': [PoseLandmarkType.leftMouth, PoseLandmarkType.leftEye],
      };

      // Reset total height in pixels
      double totalHeightInPixels = 0;

      // Check for the object used for scaling
      for (var object in objects) {
        for (var label in object.labels) {
          if (label.text.toLowerCase().contains('ball')) {
            final ballBoundingBox = object.boundingBox;
            final ballDiameterInPixels =
                min(ballBoundingBox.height, ballBoundingBox.width);
            _pixelsPerCm = ballDiameterInPixels / _ballDiameterCm;
            print('Calculated _pixelsPerCm: $_pixelsPerCm');
            break;
          }
        }
      }

      // Calculate the distances and sum them in pixels
      landmarkPairs.forEach((description, landmarksList) {
        if (landmarks.containsKey(landmarksList[0]) &&
            landmarks.containsKey(landmarksList[1])) {
          final point1 = landmarks[landmarksList[0]]!;
          final point2 = landmarks[landmarksList[1]]!;
          double distanceInPixels = calculateEuclideanDistance(
              Point(point1.x, point1.y), Point(point2.x, point2.y));

          // Accumulate total height in pixels
          totalHeightInPixels += distanceInPixels;

          // Store distances in cm for display
          distancesInCm[description] = distanceInPixels / _pixelsPerCm;
        } else {
          print("Required landmarks not found for $description.");
        }
      });

      // Convert total pixels to centimeters and update state
      if (_pixelsPerCm > 0) {
        double totalHeightInCm = totalHeightInPixels / _pixelsPerCm;
        setState(() {
          _personHeightCm = totalHeightInCm;
          print(
              "Total height calculated: ${_personHeightCm.toStringAsFixed(2)} cm");
        });
      } else {
        print(
            "Scaling factor (pixels per cm) not set. Cannot compute total height in cm.");
      }
    } else {
      print("No poses or objects detected.");
    }
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
