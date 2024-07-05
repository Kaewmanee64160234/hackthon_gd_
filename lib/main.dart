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
    final double screenWidth = MediaQuery.of(context).size.width;
    final double screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Google ML Kit Demo App',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.purple,
      ),
      body: Center(
        child: Padding(
          padding: EdgeInsets.all(screenWidth * 0.03),
          child: LayoutBuilder(
            builder: (context, constraints) {
              return Column(
                children: [
                  CustomCard(
                    'Live Measurement',
                    UnifiedDetectorView(),
                    imageUrl:
                        'https://img5.pic.in.th/file/secure-sv1/16fade08659b8e6b0.jpg',
                    maxWidth: constraints.maxWidth,
                  ),
                  CustomCard(
                    'Pick a Photo',
                    ObjectDetectionApp(),
                    imageUrl:
                        'https://img5.pic.in.th/file/secure-sv1/200d5e9119519cfc3.jpg',
                    maxWidth: constraints.maxWidth,
                  ),
                ],
              );
            },
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
  final String imageUrl;
  final double maxWidth;

  const CustomCard(this._label, this._viewPage,
      {this.featureCompleted = true,
      required this.imageUrl,
      required this.maxWidth});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 5,
      margin: EdgeInsets.only(bottom: 10),
      child: InkWell(
        onTap: () {
          if (!featureCompleted) {
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
              content: const Text('This feature has not been implemented yet'),
            ));
          } else {
            Navigator.push(
                context, MaterialPageRoute(builder: (context) => _viewPage));
          }
        },
        child: Column(
          children: [
            Image.network(
              imageUrl,
              width: maxWidth,
              height: maxWidth *
                  0.6, // Adjust height based on the width to maintain aspect ratio
              fit: BoxFit.cover,
            ),
            Row(
              children: [
                Expanded(
                  child: Container(
                    color: Theme.of(context).primaryColor,
                    padding:
                        EdgeInsets.symmetric(vertical: 20.0, horizontal: 10.0),
                    child: Text(
                      _label,
                      style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 20),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
