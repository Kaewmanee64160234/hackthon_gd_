import 'dart:io';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:hackathon/vision_detector_views/detector_viewForHeight.dart';
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

  final double _ballDiameterCm = 22;
  double _pixelsPerCm = 0.0;
  double _personHeightCm = 0.00;
  String posePosition = 'front';

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
          DetectorViewForHeight(
            title: 'Unified Detector',
            customPaint: _customPaint,
            text: 'Detection Results',
            initialCameraLensDirection: _cameraLensDirection,
            onCameraLensDirectionChanged: (value) => setState(() {
              _cameraLensDirection = value;
            }),
            onImage: _processImage,
          ),
          Positioned(
            left: 0,
            top: 100,
            right: 0,
            child: Container(
              height: 60,
              padding: EdgeInsets.all(10),
              color: Colors.black.withOpacity(0.7),
              child: Center(
                child: Text(
                  'Estimated Height: ${_personHeightCm.toStringAsFixed(2)} cm ${posePosition}',
                  style: TextStyle(color: Colors.white, fontSize: 20),
                ),
              ),
            ),
          ),
          Positioned(
            top: 16,
            right: 16,
            child: ElevatedButton(
              onPressed: () {
                setState(() {
                  posePosition = posePosition == 'front' ? 'side' : 'front';
                });
              },
              child: Text('Toggle Pose Position'),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _initializeDetectors() async {
    final modelPath = await getAssetPath('assets/ml/object_labeler.tflite');
    final options = LocalObjectDetectorOptions(
        mode: DetectionMode.single,
        classifyObjects: true,
        multipleObjects: true,
        modelPath: modelPath);
    _objectDetector = ObjectDetector(options: options);
  }

  Future<void> _onImageCaptured(InputImage inputImage, XFile file) async {
    if (!_isBusy) {
      _isBusy = true;
      try {
        final objects = await _objectDetector!.processImage(inputImage);
        final objectPainter = ObjectDetectorPainter(
          objects,
          inputImage.metadata!.size,
          inputImage.metadata!.rotation,
          _cameraLensDirection,
        );

        final poses = await _poseDetector.processImage(inputImage);
        final posePainter = PosePainter(
          poses,
          inputImage.metadata!.size,
          inputImage.metadata!.rotation,
          _cameraLensDirection,
        );

        calculateSpecificDistances(poses, objects, inputImage);

        setState(() {
          _customPaint =
              CustomPaint(painter: CombinedPainter(objectPainter, posePainter));
        });

        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (context) => Scaffold(
              appBar: AppBar(
                title: Text('Detection Results'),
              ),
              body: Column(
                children: [
                  Expanded(
                    child: Image.file(File(file.path)),
                  ),
                  Container(
                    height: 60,
                    padding: EdgeInsets.all(10),
                    color: Colors.black.withOpacity(0.7),
                    child: Center(
                      child: Text(
                        'Estimated Height: ${_personHeightCm.toStringAsFixed(2)} cm',
                        style: TextStyle(color: Colors.white, fontSize: 20),
                      ),
                    ),
                  ),
                  ElevatedButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: Text('Back to Camera'),
                  ),
                ],
              ),
            ),
          ),
        );
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

  double calculateVerticalDistance(Point<double> point1, Point<double> point2) {
    return (point2.y - point1.y).abs();
  }

  void calculateSpecificDistances(
      List<Pose> poses, List<DetectedObject> objects, InputImage inputImage) {
    if (poses.isNotEmpty) {
      final pose = poses.first;
      final landmarks = pose.landmarks;

      // Check for the scaling object first
      for (var object in objects) {
        for (var label in object.labels) {
          if (label.text.toLowerCase() == 'ball') {
            // ตรวจสอบเฉพาะคำว่า 'ball' เท่านั้น
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
      Point<double> rightMouth = Point(
          landmarks[PoseLandmarkType.rightMouth]!.x,
          landmarks[PoseLandmarkType.rightMouth]!.y);
      Point<double> leftEye = Point(landmarks[PoseLandmarkType.leftEye]!.x,
          landmarks[PoseLandmarkType.leftEye]!.y);
      Point<double> rightEye = Point(landmarks[PoseLandmarkType.rightEye]!.x,
          landmarks[PoseLandmarkType.rightEye]!.y);

      // Extract landmark coordinates for side pose
      Point<double> leftWrist = Point(landmarks[PoseLandmarkType.leftWrist]!.x,
          landmarks[PoseLandmarkType.leftWrist]!.y);
      Point<double> rightWrist = Point(
          landmarks[PoseLandmarkType.rightWrist]!.x,
          landmarks[PoseLandmarkType.rightWrist]!.y);
      Point<double> leftElbow = Point(landmarks[PoseLandmarkType.leftElbow]!.x,
          landmarks[PoseLandmarkType.leftElbow]!.y);
      Point<double> rightElbow = Point(
          landmarks[PoseLandmarkType.rightElbow]!.x,
          landmarks[PoseLandmarkType.rightElbow]!.y);

      // Calculate distances directly between relevant points
      //
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
      double distanceNoseShoulder = calculateVerticalDistance(
          nose, leftShoulder); // Using vertical distance
      double distanceMouthEye = calculateVerticalDistance(
          leftMouth, leftEye); // Using vertical distance
      double distanceRightMouthEye =
          calculateVerticalDistance(rightMouth, rightEye);
      // double distanceNoseShoulder = calculateEuclideanDistance(
      //     nose, leftShoulder); // Assuming left shoulder as reference
      // double distanceMouthEye = calculateEuclideanDistance(leftMouth, leftEye);
      double distanceNoseToEye = calculateVerticalDistance(nose, leftEye);
      double distanceEyeToTopOfHead = distanceNoseToEye * 2;

      // Calculate average distances for symmetry
      double averageAnkleKnee =
          (distanceLeftAnkleKnee + distanceRightAnkleKnee) / 2;
      double averageKneeHip = (distanceLeftKneeHip + distanceRightKneeHip) / 2;
      double averageHipShoulder =
          (distanceLeftHipShoulder + distanceRightHipShoulder) / 2;

      // Calculate distances for side pose
      double distanceLeftWristElbow =
          calculateEuclideanDistance(leftWrist, leftElbow);
      double distanceRightWristElbow =
          calculateEuclideanDistance(rightWrist, rightElbow);
      double distanceLeftElbowHip =
          calculateEuclideanDistance(leftElbow, leftHip);
      double distanceRightElbowHip =
          calculateEuclideanDistance(rightElbow, rightHip);

      // Calculate average distances for symmetry

      // Calculate the height from mouth to eye using either left or right side
      double averageMouthEye = (distanceMouthEye + distanceRightMouthEye) / 2;

      double totalDistance = 0;
      double totalDistanceOnTheSide = 0;
      // Determine if the pose is side or front
      bool isSidePose = posePosition == 'side';
      if (!isSidePose) {
        // Total height estimation
        totalDistance = averageAnkleKnee +
            averageKneeHip +
            averageHipShoulder +
            distanceNoseShoulder +
            distanceMouthEye +
            distanceEyeToTopOfHead; // Add this line
        // Convert to real-world units
        totalDistance /= _pixelsPerCm;
      } else {
        // Total height estimation
        totalDistanceOnTheSide = averageAnkleKnee +
            averageKneeHip +
            averageHipShoulder +
            distanceNoseShoulder +
            averageMouthEye +
            distanceEyeToTopOfHead;

        // Convert to real-world units
        totalDistanceOnTheSide /= _pixelsPerCm;
      }
      double finalHeight = isSidePose ? totalDistanceOnTheSide : totalDistance;

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
      distancesInCm['Eye to Top of Head'] =
          distanceEyeToTopOfHead / _pixelsPerCm; // Add this line
      distancesInCm['Left Mouth to Left Eye'] = distanceMouthEye / _pixelsPerCm;
      distancesInCm['Right Mouth to Right Eye'] =
          distanceRightMouthEye / _pixelsPerCm;
      distancesInCm['Eye to Top of Head'] =
          distanceEyeToTopOfHead / _pixelsPerCm;
      distancesInCm['Left Wrist to Left Elbow'] =
          distanceLeftWristElbow / _pixelsPerCm;
      distancesInCm['Right Wrist to Right Elbow'] =
          distanceRightWristElbow / _pixelsPerCm;
      distancesInCm['Left Elbow to Left Hip'] =
          distanceLeftElbowHip / _pixelsPerCm;
      distancesInCm['Right Elbow to Right Hip'] =
          distanceRightElbowHip / _pixelsPerCm;
      distancesInCm['Total Body Height'] = finalHeight;

      if (finalHeight > 0 && finalHeight < 300) {
        distancesInCm['Total Body Height'] = finalHeight;
      } else if (!isSidePose &&
          totalDistanceOnTheSide > 0 &&
          totalDistanceOnTheSide < 300) {
        // Fallback to side pose calculation if front pose fails
        distancesInCm['Total Body Height'] = totalDistanceOnTheSide;
      }

      _personHeightCm = distancesInCm['Total Body Height'] ?? 0.0;
      print("Estimated height: ${_personHeightCm.toStringAsFixed(2)} cm, ");
    } else {
      print("No poses or objects detected.");
    }
    setState(() {});
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
