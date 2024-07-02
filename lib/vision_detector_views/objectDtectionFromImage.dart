import 'dart:io';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:hackathon/vision_detector_views/utils.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img; // Add this line

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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Object and Pose Detection'),
      ),
      body: Stack(
        children: [
          if (_image != null)
            Image.file(
              File(_image!.path),
              fit: BoxFit.cover,
              height: double.infinity,
              width: double.infinity,
              alignment: Alignment.center,
            ),
          if (_image != null && _objects != null && _poses != null)
            CustomPaint(
              size: Size.infinite,
              painter: CombinedPainter(
                ObjectDetectorPainter(_objects!, _imageSize,
                    InputImageRotation.rotation0deg, CameraLensDirection.back),
                PosePainter(_poses!, _imageSize,
                    InputImageRotation.rotation0deg, CameraLensDirection.back),
              ),
            ),
          Center(
            child: ElevatedButton(
              onPressed: _pickImage,
              child: Text('Pick Image'),
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

class ObjectDetectorPainter extends CustomPainter {
  final List<DetectedObject> objects;
  final Size imageSize;
  final InputImageRotation rotation;
  final CameraLensDirection cameraLensDirection;

  ObjectDetectorPainter(
    this.objects,
    this.imageSize,
    this.rotation,
    this.cameraLensDirection,
  );

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    for (var object in objects) {
      final rect = object.boundingBox;
      canvas.drawRect(
          Rect.fromLTRB(
            translateX(rect.left, imageSize, size),
            translateY(rect.top, imageSize, size),
            translateX(rect.right, imageSize, size),
            translateY(rect.bottom, imageSize, size),
          ),
          paint);
    }
  }

  @override
  bool shouldRepaint(ObjectDetectorPainter oldDelegate) {
    return oldDelegate.objects != objects;
  }

  double translateX(double x, Size imageSize, Size size) {
    double scaleX = size.width / imageSize.width;
    return x * scaleX;
  }

  double translateY(double y, Size imageSize, Size size) {
    double scaleY = size.height / imageSize.height;
    return y * scaleY;
  }
}

class PosePainter extends CustomPainter {
  final List<Pose> poses;
  final Size imageSize;
  final InputImageRotation rotation;
  final CameraLensDirection cameraLensDirection;

  PosePainter(
    this.poses,
    this.imageSize,
    this.rotation,
    this.cameraLensDirection,
  );

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    for (var pose in poses) {
      for (var landmark in pose.landmarks.values) {
        canvas.drawCircle(
            Offset(
              translateX(landmark.x, imageSize, size),
              translateY(landmark.y, imageSize, size),
            ),
            2.0,
            paint);
      }
    }
  }

  @override
  bool shouldRepaint(PosePainter oldDelegate) {
    return oldDelegate.poses != poses;
  }

  double translateX(double x, Size imageSize, Size size) {
    double scaleX = size.width / imageSize.width;
    return x * scaleX;
  }

  double translateY(double y, Size imageSize, Size size) {
    double scaleY = size.height / imageSize.height;
    return y * scaleY;
  }
}
