import 'dart:io';
import 'dart:math';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:hackathon/vision_detector_views/camera_view.dart';
import 'package:hackathon/vision_detector_views/camera_viewForHeight.dart';
import 'package:hackathon/vision_detector_views/painters/coordinates_translator.dart';
import 'package:hackathon/vision_detector_views/painters/object_detector_painter.dart';
import 'package:hackathon/vision_detector_views/utils.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img; // Add this line

import 'package:google_mlkit_commons/google_mlkit_commons.dart';

void main() => runApp(ObjectDetectionApp());

class ObjectDetectionApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: ObjectDetectionHome(),
    );
  }
}

class ObjectDetectionHome extends StatefulWidget {
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
        // Process image for object detection
        _objects = await _objectDetector!.processImage(inputImage);

        // Process image for pose detection
        _poses = await _poseDetector.processImage(inputImage);

        // Calculate the height of the person
        calculateSpecificDistances(_poses!, _objects!, inputImage);

        // Update state
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
      double distanceNoseShoulder = calculateVerticalDistance(
          nose, leftShoulder); // Using vertical distance
      double distanceMouthEye = calculateVerticalDistance(
          leftMouth, leftEye); // Using vertical distance
      double distanceNoseToEye = calculateVerticalDistance(nose, leftEye);
      double distanceEyeToTopOfHead = distanceNoseToEye * 2;

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
          distanceMouthEye +
          distanceEyeToTopOfHead;

      // Convert to real-world units
      totalDistance /= _pixelsPerCm;

      // Print or update state with the total distance
      _personHeightCm = totalDistance;
      print("Estimated height: ${_personHeightCm.toStringAsFixed(2)} cm");
    } else {
      print("No poses or objects detected.");
    }
    setState(() {});
  }

  void _refreshDetection() {
    if (_image != null) {
      _processImage(InputImage.fromFilePath(_image!.path));
    }
  }

  void _onImageTaken(XFile image) {
    setState(() {
      _image = image;
    });
    _processImage(InputImage.fromFilePath(image.path));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Object and Pose Detection'),
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          return Column(
            children: [
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 16),
                child: ElevatedButton(
                  onPressed: _pickImage,
                  child: Text('Pick Image'),
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(vertical: 16),
                    textStyle: TextStyle(fontSize: 18),
                  ),
                ),
              ),
              if (_image != null)
                Container(
                  height: constraints.maxHeight * 0.5,
                  width: double.infinity,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.black),
                  ),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.file(
                        File(_image!.path),
                        fit: BoxFit.contain,
                      ),
                      if (_objects != null && _poses != null)
                        CustomPaint(
                          size: Size.infinite,
                          painter: CombinedPainter(
                            ObjectDetectorPainter(
                                _objects!,
                                _imageSize,
                                InputImageRotation.rotation0deg,
                                CameraLensDirection.back),
                            PosePainter(
                                _poses!,
                                _imageSize,
                                InputImageRotation.rotation0deg,
                                CameraLensDirection.back),
                          ),
                        ),
                    ],
                  ),
                ),
              if (_image != null)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  child: ElevatedButton(
                    onPressed: _refreshDetection,
                    child: Text('Refresh Detection'),
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(vertical: 16),
                      textStyle: TextStyle(fontSize: 18),
                    ),
                  ),
                ),
              if (_image != null)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => CameraViewForHeight(
                            customPaint: null,
                            onImageCaptured: (XFile file) {
                              _onImageTaken(file);
                            },
                          ),
                        ),
                      );
                    },
                    child: Text('Take Photo'),
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(vertical: 16),
                      textStyle: TextStyle(fontSize: 18),
                    ),
                  ),
                ),
              if (_personHeightCm > 0)
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Text(
                    "Estimated height: ${_personHeightCm.toStringAsFixed(2)} cm",
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }

  Size get _imageSize {
    if (_image == null) return Size.zero;
    final decodedImage = img.decodeImage(File(_image!.path).readAsBytesSync())!;
    return Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
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

    for (final pose in poses) {
      pose.landmarks.forEach((_, landmark) {
        canvas.drawCircle(
            Offset(
              translateX(
                landmark.x,
                size,
                imageSize,
                rotation,
                cameraLensDirection,
              ),
              translateY(
                landmark.y,
                size,
                imageSize,
                rotation,
                cameraLensDirection,
              ),
            ),
            1,
            paint);
      });

      void paintLine(
          PoseLandmarkType type1, PoseLandmarkType type2, Paint paintType) {
        final PoseLandmark joint1 = pose.landmarks[type1]!;
        final PoseLandmark joint2 = pose.landmarks[type2]!;
        canvas.drawLine(
          Offset(
            translateX(
              joint1.x,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
            translateY(
              joint1.y,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
          ),
          Offset(
            translateX(
              joint2.x,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
            translateY(
              joint2.y,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
          ),
          paintType,
        );
      }

      //Draw arms
      paintLine(
          PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow, leftPaint);
      paintLine(
          PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist, leftPaint);
      paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow,
          rightPaint);
      paintLine(
          PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist, rightPaint);

      //Draw Body
      paintLine(
          PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip, leftPaint);
      paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip,
          rightPaint);

      //Draw legs
      paintLine(PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee, leftPaint);
      paintLine(
          PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle, leftPaint);
      paintLine(
          PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee, rightPaint);
      paintLine(
          PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle, rightPaint);
    }
  }

  @override
  bool shouldRepaint(covariant PosePainter oldDelegate) {
    return oldDelegate.imageSize != imageSize || oldDelegate.poses != poses;
  }
}
