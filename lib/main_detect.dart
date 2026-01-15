import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;

const int _minProcessIntervalMs = 120;
const double _scoreThreshold = 0.55;
const double _nmsIou = 0.45;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cams = await availableCameras();
  runApp(App(cameras: cams));
}

class App extends StatelessWidget {
  final List<CameraDescription> cameras;
  const App({super.key, required this.cameras});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Detect Benchmark',
        debugShowCheckedModeBanner: false,
        home: DetectPage(cameras: cameras),
      );
}

class DetectPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const DetectPage({super.key, required this.cameras});
  @override
  State<DetectPage> createState() => _DetectPageState();
}

class _DetectPageState extends State<DetectPage> {
  CameraController? _controller;
  tfl.Interpreter? _det;
  List<String> _labels = [];
  String _status = 'Init...';
  bool _isProcessing = false;
  int _lastMs = 0;

  double _fps = 0.0;
  int _phaseStart = 0;

  List<_Det> _dets = [];

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final cam = widget.cameras.first;
      _controller = CameraController(
        cam,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _controller!.initialize();

      try {
        final txt = await rootBundle.loadString('assets/models/labelmap.txt');
        _labels = txt.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty && e != '???').toList();
      } catch (_) {}

      _det = await tfl.Interpreter.fromAsset('assets/models/detect.tflite');

      await _controller!.startImageStream(_onFrame);
      setState(() => _status = 'Running ✅');
    } catch (e) {
      setState(() => _status = 'Init error: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _det?.close();
    super.dispose();
  }

  Future<void> _onFrame(CameraImage camImg) async {
    final now = DateTime.now().millisecondsSinceEpoch;
    if (_isProcessing || now - _lastMs < _minProcessIntervalMs) return;
    _isProcessing = true;
    _lastMs = now;
    _phaseStart = now;

    try {
      final rgb = _yuv420ToImage(camImg);
      if (rgb == null) return;

      final size = _controller!.value.previewSize!;
      final dets = await _runDet(rgb, previewW: size.width, previewH: size.height);

      if (!mounted) return;
      setState(() {
        _dets = dets;
        final dt = (DateTime.now().millisecondsSinceEpoch - _phaseStart).clamp(1, 1 << 30);
        final inst = 1000.0 / dt;
        _fps = _fps == 0.0 ? inst : (_fps * 0.85 + inst * 0.15);
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = 'Runtime error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Future<List<_Det>> _runDet(img.Image image, {required double previewW, required double previewH}) async {
    final i = _det!;
    final inT = i.getInputTensor(0);
    final H = inT.shape[1], W = inT.shape[2];
    final resized = img.copyResize(image, width: W, height: H);
    final isFloat = inT.type.toString().toLowerCase().contains('float');
    final input = isFloat ? _imageToFloat(resized) : _imageToUint8(resized);

    final out0 = _zeros(i.getOutputTensor(0).shape, true);
    final out1 = _zeros(i.getOutputTensor(1).shape, true);
    final out2 = _zeros(i.getOutputTensor(2).shape, true);
    final outMap = {0: out0, 1: out1, 2: out2};
    if (i.getOutputTensors().length == 4) {
      outMap[3] = _zeros(i.getOutputTensor(3).shape, true);
    }
    i.runForMultipleInputs([input], outMap);

    final boxes = out0[0] as List;
    final classes = out1[0] as List;
    final scores = out2[0] as List;

    final area = previewW * previewH;
    const double minFrac = 0.01, maxFrac = 0.9;

    final raw = <_Det>[];
    for (int k = 0; k < scores.length; k++) {
      final s = (scores[k] as num).toDouble();
      if (s < _scoreThreshold) continue;

      final b = boxes[k] as List;
      final ymin = (b[0] as num).toDouble().clamp(0.0, 1.0);
      final xmin = (b[1] as num).toDouble().clamp(0.0, 1.0);
      final ymax = (b[2] as num).toDouble().clamp(0.0, 1.0);
      final xmax = (b[3] as num).toDouble().clamp(0.0, 1.0);
      if (xmax <= xmin || ymax <= ymin) continue;

      final l = xmin * previewW, t = ymin * previewH, r = xmax * previewW, btm = ymax * previewH;
      final w = (r - l).abs(), h = (btm - t).abs();
      final frac = (w * h) / area;
      if (frac < minFrac || frac > maxFrac) continue;

      final rawCls = (classes[k] as num).toInt();
      final clsIdx = rawCls > 0 ? rawCls - 1 : rawCls;
      final label = (clsIdx >= 0 && clsIdx < _labels.length) ? _labels[clsIdx] : 'id:$rawCls';

      raw.add(_Det(Rect.fromLTRB(l, t, r, btm), label, s));
    }
    return _nms(raw, iouThreshold: _nmsIou); // ✅ ใช้ named parameter
  }

  @override
  Widget build(BuildContext context) {
    final ready = _controller?.value.isInitialized == true;
    return Scaffold(
      backgroundColor: Colors.black,
      body: ready
          ? Stack(
              fit: StackFit.expand,
              children: [
                Center(
                  child: AspectRatio(
                    aspectRatio: _controller!.value.aspectRatio,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        CameraPreview(_controller!),
                        IgnorePointer(child: CustomPaint(painter: _DetPainter(_dets))),
                      ],
                    ),
                  ),
                ),
                SafeArea(
                  child: Align(
                    alignment: Alignment.topCenter,
                    child: Container(
                      margin: const EdgeInsets.only(top: 8),
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(12)),
                      child: Text('$_status • FPS: ${_fps.toStringAsFixed(1)}',
                          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                    ),
                  ),
                ),
              ],
            )
          : const Center(child: CircularProgressIndicator()),
    );
  }

  // --- utils ---
  img.Image? _yuv420ToImage(CameraImage image) {
    if (image.format.group != ImageFormatGroup.yuv420) return null;
    final w = image.width, h = image.height;
    final y = image.planes[0], u = image.planes[1], v = image.planes[2];
    final im = img.Image(width: w, height: h);
    final yRow = y.bytesPerRow, uvRow = u.bytesPerRow, uvPix = u.bytesPerPixel ?? 1;
    for (int yy = 0; yy < h; yy++) {
      final yOff = yy * yRow;
      final uvRowOff = (yy ~/ 2) * uvRow;
      for (int xx = 0; xx < w; xx++) {
        final Y = y.bytes[yOff + xx] & 0xFF;
        final uvCol = (xx ~/ 2) * uvPix;
        final U = u.bytes[uvRowOff + uvCol] & 0xFF;
        final V = v.bytes[uvRowOff + uvCol] & 0xFF;
        double yf = Y.toDouble(), uf = U - 128.0, vf = V - 128.0;
        int r = (yf + 1.402 * vf).round();
        int g = (yf - 0.344136 * uf - 0.714136 * vf).round();
        int b = (yf + 1.772 * uf).round();
        im.setPixelRgb(xx, yy, r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255));
      }
    }
    return im;
  }

  List _imageToFloat(img.Image im) => [
        List.generate(im.height, (y) => List.generate(im.width, (x) {
              final p = im.getPixel(x, y);
              return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
            }))
      ];
  List _imageToUint8(img.Image im) => [
        List.generate(im.height, (y) => List.generate(im.width, (x) {
              final p = im.getPixel(x, y);
              return [p.r, p.g, p.b];
            }))
      ];
  List _zeros(List<int> s, bool f) {
    dynamic z(List<int> a) => a.isEmpty ? (f ? 0.0 : 0) : List.generate(a.first, (_) => z(a.sublist(1)));
    return z(s);
  }
}

// === NMS & IOU ===
List<_Det> _nms(List<_Det> dets, {double iouThreshold = 0.5}) {
  final src = [...dets]..sort((a, b) => b.score.compareTo(a.score));
  final keep = <_Det>[];
  for (final d in src) {
    bool ok = true;
    for (final k in keep) {
      if (_iou(d.rect, k.rect) > iouThreshold) { ok = false; break; }
    }
    if (ok) keep.add(d);
  }
  return keep;
}

double _iou(Rect a, Rect b) {
  final inter = Rect.fromLTRB(
    math.max(a.left, b.left),
    math.max(a.top, b.top),
    math.min(a.right, b.right),
    math.min(a.bottom, b.bottom),
  );
  final interArea = math.max(0.0, inter.width) * math.max(0.0, inter.height);
  final union = a.width * a.height + b.width * b.height - interArea;
  return union <= 0 ? 0 : interArea / union;
}

// === model result ===
class _Det {
  final Rect rect;
  final String label;
  final double score;
  _Det(this.rect, this.label, this.score);
}

class _DetPainter extends CustomPainter {
  final List<_Det> dets;
  _DetPainter(this.dets);
  @override
  void paint(Canvas c, Size s) {
    final box = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.greenAccent;
    final bg = Paint()..color = Colors.black54;
    for (final d in dets) {
      c.drawRect(d.rect, box);
      final text = '${d.label} ${d.score.toStringAsFixed(2)}';
      final tp = TextPainter(
        text: TextSpan(text: text, style: const TextStyle(color: Colors.white, fontSize: 12)),
        textDirection: TextDirection.ltr,
      )..layout();
      const pad = 4.0;
      final r = Rect.fromLTWH(d.rect.left, math.max(0, d.rect.top - tp.height - 2 * pad),
          tp.width + 2 * pad, tp.height + 2 * pad);
      c.drawRect(r, bg);
      tp.paint(c, Offset(r.left + pad, r.top + pad));
    }
  }
  @override
  bool shouldRepaint(covariant _DetPainter old) => old.dets != dets;
}
