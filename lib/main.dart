import 'package:flutter/material.dart';
import 'package:hackathon/vision_detector_views/UnifiedDetectorView.dart';
import 'package:hackathon/vision_detector_views/objectDtectionFromImage.dart';
import 'package:hackathon/vision_detector_views/object_detector_view.dart';
import 'package:hackathon/vision_detector_views/pose_detector_view.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Home(),
    );
  }
}

class Home extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Google ML Kit Demo App'),
        centerTitle: true,
        elevation: 0,
      ),
      // body: Center(
      //   child: Padding(
      //     padding: EdgeInsets.all(10),
      //     child: Container(
      //         child: CustomCard('Height Measurement', UnifiedDetectorView())),
      //   ),
      // ),

      body: Center(
        child: Padding(
          padding: EdgeInsets.all(10),
          child: Column(
            children: [
              CustomCard('Object Detection', ObjectDetectorView()),
              CustomCard('Pose Detection', PoseDetectorView()),
              CustomCard('Height Measurement', ObjectDetectionApp()),
              // UnifiedDetectorView
              CustomCard("Height Measurement real", UnifiedDetectorView()),
            ],
          ),
        ),
      ),
    );
  }
}

class CustomCard extends StatelessWidget {
  final String _label;
  final Widget _viewPage;
  final bool featureCompleted;

  const CustomCard(this._label, this._viewPage, {this.featureCompleted = true});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 5,
      margin: EdgeInsets.only(bottom: 10),
      child: ListTile(
        tileColor: Theme.of(context).primaryColor,
        title: Text(
          _label,
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        onTap: () {
          if (!featureCompleted) {
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                content:
                    const Text('This feature has not been implemented yet')));
          } else {
            Navigator.push(
                context, MaterialPageRoute(builder: (context) => _viewPage));
          }
        },
      ),
    );
  }
}
