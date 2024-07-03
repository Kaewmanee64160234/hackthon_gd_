import 'dart:io';
import 'dart:math';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:hackathon/vision_detector_views/painters/coordinates_translator.dart';
import 'package:hackathon/vision_detector_views/painters/object_detector_painter.dart';
import 'package:hackathon/vision_detector_views/utils.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

class ObjectDetectionApp extends StatelessWidget {
  final XFile? imageFile;

  ObjectDetectionApp({this.imageFile});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: ObjectDetectionHome(imageFile: imageFile),
    );
  }
}

class ObjectDetectionHome extends StatefulWidget {
  final XFile? imageFile;

  ObjectDetectionHome({this.imageFile});

  @override
  _ObjectDetectionHomeState createState() => _ObjectDetectionHomeState();
}

class _ObjectDetectionHomeState extends State<ObjectDetectionHome> {
  final ImagePicker _picker = ImagePicker();
  XFile? _image;
  List<DetectedObject>? _objects;
  List<Pose>? _poses;
  final PoseDetector _poseDetector =
      PoseDetector(options: PoseDetectorOptions());
  ObjectDetector? _objectDetector;
  bool _isBusy = false;
  double _pixelsPerCm = 0.0;
  double _personHeightCm = 0.00;
  final double _ballDiameterCm = 22;
  Map<String, double> distancesInCm = {};

  String posePosition = 'front';

  @override
  void initState() {
    super.initState();
    _initializeDetectors();
    if (widget.imageFile != null) {
      _image = widget.imageFile;
      _processImage(InputImage.fromFilePath(_image!.path));
    }
  }

  @override
  void dispose() {
    _poseDetector.close();
    _objectDetector?.close();
    super.dispose();
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

  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = pickedFile;
      });
      _processImage(InputImage.fromFilePath(pickedFile.path));
    }
  }

  Future<void> _processImage(InputImage inputImage) async {
    if (!_isBusy) {
      _isBusy = true;
      try {
        _objects = await _objectDetector!.processImage(inputImage);
        _poses = await _poseDetector.processImage(inputImage);
        calculateSpecificDistances(_poses!, _objects!, inputImage);
        setState(() {});
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Object and Pose Detection'),
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (_image != null)
            Image.file(
              File(_image!.path),
              fit: BoxFit.contain, // Set to BoxFit.contain to ensure full width
            ),
          if (_objects != null && _poses != null)
            CustomPaint(
              painter: CombinedPainter(
                ObjectDetectorPainter(_objects!, _imageSize,
                    InputImageRotation.rotation0deg, CameraLensDirection.back),
                PosePainter(_poses!, _imageSize,
                    InputImageRotation.rotation0deg, CameraLensDirection.back),
              ),
            ),
          if (_personHeightCm > 0)
            Positioned(
              top: 16,
              left: 16,
              child: Container(
                padding: const EdgeInsets.all(8),
                color: Colors.black.withOpacity(0.5),
                child: Text(
                  "Estimated height: ${_personHeightCm.toStringAsFixed(2)} cm",
                  style: TextStyle(color: Colors.white, fontSize: 18),
                ),
              ),
            ),
          Positioned(
            bottom: 16,
            left: 16,
            child: ElevatedButton(
              onPressed: _pickImage,
              child: Text('Pick Image'),
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 16),
                textStyle: TextStyle(fontSize: 18),
              ),
            ),
          ),
          Positioned(
            bottom: 16,
            right: 16,
            child: ElevatedButton(
              onPressed: _refreshDetection,
              child: Text('Refresh Detection'),
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 16),
                textStyle: TextStyle(fontSize: 18),
              ),
            ),
          ),
          Positioned(
            top: 50,
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

  Size get _imageSize {
    if (_image == null) return Size.zero;
    final decodedImage = img.decodeImage(File(_image!.path).readAsBytesSync())!;
    return Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
  }

  void _refreshDetection() {
    if (_image != null) {
      _processImage(InputImage.fromFilePath(_image!.path));
    }
  }
}

class CombinedPainter extends CustomPainter {
  final ObjectDetectorPainter objectPainter;
  final PosePainter posePainter;

  CombinedPainter(this.objectPainter, this.posePainter);

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / posePainter.imageSize.width;
    final double scaleY = size.height / posePainter.imageSize.height;
    canvas.scale(scaleX, scaleY);

    objectPainter.paint(canvas, posePainter.imageSize);
    posePainter.paint(canvas, posePainter.imageSize);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}

class PosePainter extends CustomPainter {
  PosePainter(
    this.poses,
    this.imageSize,
    this.rotation,
    this.cameraLensDirection,
  );

  final List<Pose> poses;
  final Size imageSize;
  final InputImageRotation rotation;
  final CameraLensDirection cameraLensDirection;

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..color = Colors.green;

    final leftPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.yellow;

    final rightPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.blueAccent;

    canvas.save();
    canvas.scale(scaleX, scaleY);

    for (final pose in poses) {
      pose.landmarks.forEach((_, landmark) {
        final x = translateX(
          landmark.x,
          size,
          imageSize,
          rotation,
          cameraLensDirection,
        );
        final y = translateY(
          landmark.y,
          size,
          imageSize,
          rotation,
          cameraLensDirection,
        );

        if (x >= 0 && x <= size.width && y >= 0 && y <= size.height) {
          canvas.drawCircle(
            Offset(x, y),
            1,
            paint,
          );
        }
      });

      void paintLine(
          PoseLandmarkType type1, PoseLandmarkType type2, Paint paintType) {
        final PoseLandmark joint1 = pose.landmarks[type1]!;
        final PoseLandmark joint2 = pose.landmarks[type2]!;

        final x1 = translateX(
          joint1.x,
          size,
          imageSize,
          rotation,
          cameraLensDirection,
        );
        final y1 = translateY(
          joint1.y,
          size,
          imageSize,
          rotation,
          cameraLensDirection,
        );
        final x2 = translateX(
          joint2.x,
          size,
          imageSize,
          rotation,
          cameraLensDirection,
        );
        final y2 = translateY(
          joint2.y,
          size,
          imageSize,
          rotation,
          cameraLensDirection,
        );

        if (x1 >= 0 &&
            x1 <= size.width &&
            y1 >= 0 &&
            y1 <= size.height &&
            x2 >= 0 &&
            x2 <= size.width &&
            y2 >= 0 &&
            y2 <= size.height) {
          canvas.drawLine(
            Offset(x1, y1),
            Offset(x2, y2),
            paintType,
          );
        }
      }

      // Draw arms
      paintLine(
          PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow, leftPaint);
      paintLine(
          PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist, leftPaint);
      paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow,
          rightPaint);
      paintLine(
          PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist, rightPaint);

      // Draw body
      paintLine(
          PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip, leftPaint);
      paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip,
          rightPaint);

      // Draw legs
      paintLine(PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee, leftPaint);
      paintLine(
          PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle, leftPaint);
      paintLine(
          PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee, rightPaint);
      paintLine(
          PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle, rightPaint);
    }

    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant PosePainter oldDelegate) {
    return oldDelegate.imageSize != imageSize || oldDelegate.poses != poses;
  }
}
