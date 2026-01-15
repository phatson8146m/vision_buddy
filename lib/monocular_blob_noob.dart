import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;

// ========================= CONFIG =========================
const bool kEnableNNAPI       = false; // เปิดถ้าอุปกรณ์เสถียร
const bool kEnableGPUForDepth = false; // true ถ้า depth เป็น float32 และเครื่องรองรับ

// รอบส่งเฟรมไป pipeline (มิลลิวินาที)
const int kDepthIntervalMs = 90;   // ~11 FPS

// Depth → โหมดสัมพัทธ์ 0..1 (มาก=ใกล้). ตั้ง >0 ถ้าคาลิเบรตเป็นเมตรได้
double _metersPerUnit = 0.0; // เช่น ใส่ k ถ้าใช้สูตร meters ≈ k/(rel+eps)

// โซนสนใจหาอุปสรรค: ส่วนล่างของเฟรม depth (0..1)
const double kRoiTopFrac = 0.55; // ใช้ช่วงล่าง ~45% ของภาพ
// เปอร์เซ็นไทล์เพื่อคัด "ใกล้" (ค่าสูง=ใกล้ เพราะ normalize มาก=ใกล้)
const double kNearPercentile = 0.20; // 20th percentile
// เกณฑ์พื้นที่ blob ขั้นต่ำ (เป็นสัดส่วนของภาพทั้งหมด) เพื่อตัด noise
const double kMinBlobAreaFrac = 0.010; // >=1% ของภาพ

// ===== กัน “ฉากโล่ง” ไม่ให้มั่ว =====
const double kMinNearFrac      = 0.030; // >=3% ของพิกเซลใน ROI ต้องถูกจัดว่า "ใกล้"
const double kMinDepthContrast = 0.08;  // spread ระหว่างเปอร์เซ็นไทล์ 90 กับ 10 อย่างน้อย 0.08

// ========================= APP =========================
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Depth-only Obstacle Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.blue),
      home: CameraPage(cameras: cameras),
    );
  }
}

class CameraPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const CameraPage({super.key, required this.cameras});
  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  // Camera
  CameraController? _controller;
  late final bool _isFront;

  // Depth isolate & ports
  Isolate? _depIso;
  SendPort? _depInPort;
  late final ReceivePort _depOutPort;
  bool _depBusy = false;
  int _lastDepSendMs = 0;

  // FPS / timing
  final Stopwatch _fpsWindow = Stopwatch()..start();
  int _framesInWindow = 0;
  double _fpsNow = 0.0;
  int _lastDepDurMs = 0;

  // UI state
  String _status = "Initializing...";
  bool _showHeatmap = false;
  Uint8List? _depthOverlayPng;

  // ผลลัพธ์ depth ล่าสุด
  _DepthField? _lastDepthField;

  // ผล blob ล่าสุด
  _BlobResult? _lastBlob;

  @override
  void initState() {
    super.initState();
    _initAll();
  }

  @override
  void dispose() {
    _controller?.dispose();
    _depIso?.kill(priority: Isolate.immediate);
    super.dispose();
  }

  Future<void> _initAll() async {
    try {
      // 1) Camera
      final camera = widget.cameras.isNotEmpty ? widget.cameras.first : null;
      if (camera == null) {
        setState(() => _status = "No camera found");
        return;
      }
      _isFront = camera.lensDirection == CameraLensDirection.front;

      _controller = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _controller!.initialize();

      // 2) DEPTH isolate
      _depOutPort = ReceivePort();
      _depOutPort.listen(_onDepthMessage, onError: (e) {
        setState(() => _status = "Depth isolate error: $e");
      });
      final depthBytes = await rootBundle.load('assets/models/model_opt.tflite');
      _depIso = await Isolate.spawn(
        _depthIsolateEntry,
        _IsoInit(
          replyPort: _depOutPort.sendPort,
          modelBytes: TransferableTypedData.fromList([depthBytes.buffer.asUint8List()]),
          labels: const [],
          threads: 4,
          useNNAPI: kEnableNNAPI,
          preferGPU: kEnableGPUForDepth,
          scoreThreshold: 0.0,
        ),
        errorsAreFatal: true,
        debugName: "depth_isolate",
      );

      // 3) Camera stream (latest-only)
      await _controller!.startImageStream((frame) {
        final now = DateTime.now().millisecondsSinceEpoch;

        if (!_depBusy && (now - _lastDepSendMs) >= kDepthIntervalMs && _depInPort != null) {
          _depInPort!.send(_packYuvFrame(frame, makeHeatmap: _showHeatmap));
          _depBusy = true;
          _lastDepSendMs = now;
        }

        // FPS overlay (~0.5s)
        _framesInWindow++;
        if (_fpsWindow.elapsedMilliseconds >= 500) {
          _fpsNow = (_framesInWindow * 1000) / _fpsWindow.elapsedMilliseconds;
          _framesInWindow = 0;
          _fpsWindow..reset()..start();
          if (mounted) setState(() {});
        }
      });

      setState(() => _status = "Depth-only running ✅");
    } catch (e) {
      setState(() => _status = "Init error: $e");
    }
  }

  // ====== Depth messages ======
  void _onDepthMessage(dynamic message) {
    if (message is Map && message['type'] == 'ready') {
      _depInPort = message['port'] as SendPort;
      setState(() {});
      return;
    }
    if (message is Map && message['type'] == 'depth') {
      _depBusy = false;
      _lastDepDurMs = message['durMs'] as int? ?? 0;

      final int w = message['w'] as int;
      final int h = message['h'] as int;
      final double targetW = (message['targetW'] as num).toDouble();
      final double targetH = (message['targetH'] as num).toDouble();

      final TransferableTypedData ttd = message['norm'] as TransferableTypedData;
      final Float32List grid = ttd.materialize().asFloat32List();

      _lastDepthField = _DepthField(
        norm: grid, w: w, h: h, targetW: targetW, targetH: targetH,
      );

      if (_showHeatmap) {
        _depthOverlayPng = message['png'] as Uint8List?;
      }

      // วิเคราะห์ blob ใกล้สุดใน ROI ล่าง (มีตัวกรองฉากโล่ง)
      _lastBlob = (_lastDepthField != null)
          ? _extractNearestBlob(_lastDepthField!)
          : null;

      setState(() {
        final zTxt = (_lastBlob?.medianRel != null)
            ? (_metersPerUnit > 0
                ? "≈${(_metersPerUnit / ((_lastBlob!.medianRel ?? 0) + 1e-3)).toStringAsFixed(2)} m"
                : "z=${(_lastBlob!.medianRel ?? 0).toStringAsFixed(2)} (rel)")
            : "—";
        _status =
            "Depth-only ✅ • depth:${_lastDepDurMs}ms/$kDepthIntervalMs • FPS:${_fpsNow.toStringAsFixed(1)} • blob:${_lastBlob?.areaFrac.toStringAsFixed(3) ?? 0} • $zTxt";
      });
      return;
    }
  }

  // ส่งเฟรม YUV → Isolate
  Map _packYuvFrame(CameraImage cam, {bool makeHeatmap = false}) {
    final y = TransferableTypedData.fromList([cam.planes[0].bytes]);
    final u = TransferableTypedData.fromList([cam.planes[1].bytes]);
    final v = TransferableTypedData.fromList([cam.planes[2].bytes]);
    final size = _controller?.value.previewSize;

    final previewPortrait = (size?.height ?? cam.height.toDouble()) > (size?.width ?? cam.width.toDouble());
    final bufferPortrait  = cam.height > cam.width;
    final rotateCW = previewPortrait != bufferPortrait;

    return {
      'type': 'frame',
      'w': cam.width,
      'h': cam.height,
      'previewW': size?.width ?? cam.width.toDouble(),
      'previewH': size?.height ?? cam.height.toDouble(),
      'yRowStride': cam.planes[0].bytesPerRow,
      'uRowStride': cam.planes[1].bytesPerRow,
      'vRowStride': cam.planes[2].bytesPerRow,
      'uvPixelStride': cam.planes[1].bytesPerPixel ?? 1,
      'y': y,
      'u': u,
      'v': v,
      'makeHeatmap': makeHeatmap,
      'rotateCW': rotateCW,
    };
  }

  // ====== วิเคราะห์ blob ใกล้สุดจาก depth (กัน "ฉากโล่ง") ======
  _BlobResult? _extractNearestBlob(_DepthField f) {
    if (f.w == 0 || f.h == 0 || f.norm.isEmpty) return null;

    final int w = f.w, h = f.h;
    final int y0 = (kRoiTopFrac * h).clamp(0, h - 1).toInt();
    final int y1 = h - 1;

    // 1) เก็บค่า depth ใน ROI
    final values = <double>[];
    for (int y = y0; y <= y1; y++) {
      final base = y * w;
      for (int x = 0; x < w; x++) {
        values.add(f.norm[base + x]); // 0..1 (มาก=ใกล้)
      }
    }
    if (values.isEmpty) return null;

    // 1.1) เช็คคอนทราสต์ (กันกรณีภาพแบน/ฉากโล่ง)
    values.sort();
    double p10 = values[(0.10 * (values.length - 1)).toInt()];
    double p90 = values[(0.90 * (values.length - 1)).toInt()];
    final double spread = (p90 - p10).clamp(0.0, 1.0);
    if (spread < kMinDepthContrast) {
      return null; // ฉากโล่งหรือ depth แบนเกินไป
    }

    // 2) หาค่า threshold เปอร์เซ็นไทล์ "ใกล้"
    final int idx = (kNearPercentile * (values.length - 1)).clamp(0, values.length - 1).toInt();
    final double thr = values[idx];

    // 3) สร้าง nearMask และนับ near ทั้ง ROI
    final mask = Uint8List(w * h); // 0/1 เฉพาะใน ROI
    int nearCount = 0;
    for (int y = y0; y <= y1; y++) {
      final base = y * w;
      for (int x = 0; x < w; x++) {
        final v = f.norm[base + x];
        final isNear = (v >= thr) ? 1 : 0;
        mask[base + x] = isNear;
        nearCount += isNear;
      }
    }

    // 3.1) ต้องมี near อย่างน้อย kMinNearFrac ของ ROI ทั้งหมด (กันฉากโล่ง)
    final int roiPixels = (y1 - y0 + 1) * w;
    if (nearCount < (kMinNearFrac * roiPixels)) {
      return null; // ใกล้น้อยเกินไป => ตัด blob เทียมในฉากโล่ง
    }

    // 4) Connected components ใน ROI
    final visited = Uint8List(w * h);
    int bestArea = 0;
    int bestMinX=0, bestMinY=0, bestMaxX=0, bestMaxY=0;
    final coords = <int>[];

    void bfs(int sx, int sy) {
      final qx = <int>[sx];
      final qy = <int>[sy];
      visited[sy * w + sx] = 1;
      int minX = sx, maxX = sx, minY = sy, maxY = sy;
      int area = 0;
      final tempIdxs = <int>[];

      while (qx.isNotEmpty) {
        final x = qx.removeLast();
        final y = qy.removeLast();
        final idx0 = y * w + x;
        area++;
        tempIdxs.add(idx0);

        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;

        // 4-conn
        if (x > 0) {
          final i = y * w + (x - 1);
          if (mask[i] == 1 && visited[i] == 0) { visited[i] = 1; qx.add(x - 1); qy.add(y); }
        }
        if (x + 1 < w) {
          final i = y * w + (x + 1);
          if (mask[i] == 1 && visited[i] == 0) { visited[i] = 1; qx.add(x + 1); qy.add(y); }
        }
        if (y > y0) {
          final i = (y - 1) * w + x;
          if (mask[i] == 1 && visited[i] == 0) { visited[i] = 1; qx.add(x); qy.add(y - 1); }
        }
        if (y + 1 <= y1) {
          final i = (y + 1) * w + x;
          if (mask[i] == 1 && visited[i] == 0) { visited[i] = 1; qx.add(x); qy.add(y + 1); }
        }
      }

      if (area > bestArea) {
        bestArea = area;
        bestMinX = minX; bestMaxX = maxX; bestMinY = minY; bestMaxY = maxY;
        coords..clear()..addAll(tempIdxs);
      }
    }

    for (int y = y0; y <= y1; y++) {
      final base = y * w;
      for (int x = 0; x < w; x++) {
        final i = base + x;
        if (mask[i] == 1 && visited[i] == 0) {
          bfs(x, y);
        }
      }
    }

    if (bestArea <= 0) return null;

    // 5) เช็คสัดส่วนพื้นที่ขั้นต่ำ (กัน noise/ก้อนเล็ก)
    final areaFrac = bestArea / (w * h);
    if (areaFrac < kMinBlobAreaFrac) return null;

    // 6) median depth ของ blob
    final vals = <double>[];
    for (final i in coords) {
      vals.add(f.norm[i]);
    }
    vals.sort();
    final med = vals[vals.length ~/ 2];

    // bbox in preview
    final sx = f.targetW / w;
    final sy = f.targetH / h;
    Rect bbox = Rect.fromLTWH(
      bestMinX * sx,
      bestMinY * sy,
      (bestMaxX - bestMinX + 1) * sx,
      (bestMaxY - bestMinY + 1) * sy,
    );

    return _BlobResult(
      bboxInPreview: bbox,
      medianRel: med,
      areaFrac: areaFrac,
    );
  }

  @override
  Widget build(BuildContext context) {
    final camReady = _controller?.value.isInitialized == true;
    final previewSize = _controller?.value.previewSize;

    final medianText = (_lastBlob?.medianRel != null)
        ? (_metersPerUnit > 0
            ? "Depth median in blob ≈ ${(_metersPerUnit / ((_lastBlob!.medianRel ?? 0)+1e-3)).toStringAsFixed(2)} m"
            : "Depth median in blob = ${(_lastBlob!.medianRel ?? 0).toStringAsFixed(3)} (rel)")
        : "Depth median in blob = —";

    return Scaffold(
      backgroundColor: Colors.black,
      body: camReady && previewSize != null
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
                        if (_showHeatmap && _depthOverlayPng != null)
                          IgnorePointer(child: Image.memory(_depthOverlayPng!, fit: BoxFit.cover)),
                        // วาดกรอบ blob (พิกัด preview → canvas + mirror ถ้ากล้องหน้า)
                        IgnorePointer(
                          child: CustomPaint(
                            painter: _BlobPainter(
                              blob: _lastBlob,
                              mirrorX: _isFront,
                              srcW: previewSize.width,
                              srcH: previewSize.height,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                // Status
                SafeArea(
                  child: Align(
                    alignment: Alignment.topCenter,
                    child: Container(
                      margin: const EdgeInsets.only(top: 8),
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.55),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        _status,
                        style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600),
                      ),
                    ),
                  ),
                ),
                // FPS/งบเวลา
                SafeArea(
                  child: Align(
                    alignment: Alignment.topLeft,
                    child: Container(
                      margin: const EdgeInsets.only(top: 8, left: 8),
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.55),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        "FPS: ${_fpsNow.toStringAsFixed(1)} • depth:$_lastDepDurMs ms/$kDepthIntervalMs",
                        style: const TextStyle(color: Colors.white, fontSize: 12),
                      ),
                    ),
                  ),
                ),
                // ค่า median
                SafeArea(
                  child: Align(
                    alignment: Alignment.bottomCenter,
                    child: Container(
                      margin: const EdgeInsets.only(bottom: 10),
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.45),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        medianText,
                        style: const TextStyle(color: Colors.white, fontSize: 14),
                      ),
                    ),
                  ),
                ),
                // Toggle Heatmap
                SafeArea(
                  child: Align(
                    alignment: Alignment.topRight,
                    child: IconButton(
                      onPressed: () {
                        setState(() {
                          _showHeatmap = !_showHeatmap;
                          _depthOverlayPng = null;
                        });
                      },
                      icon: Icon(
                        Icons.heat_pump_outlined,
                        color: _showHeatmap ? Colors.orange : Colors.white,
                      ),
                      tooltip: "Toggle heatmap",
                    ),
                  ),
                ),
                if (_metersPerUnit == 0.0)
                  SafeArea(
                    child: Align(
                      alignment: Alignment.bottomLeft,
                      child: Padding(
                        padding: const EdgeInsets.all(10.0),
                        child: Text(
                          "Tip: ตั้ง _metersPerUnit > 0 เพื่อนำ z(rel) → เมตร (≈ k/(z+eps))",
                          style: const TextStyle(color: Colors.white70, fontSize: 12),
                        ),
                      ),
                    ),
                  ),
              ],
            )
          : const Center(child: CircularProgressIndicator()),
    );
  }
}

// ======================= Painter & Data =======================
class _BlobResult {
  final Rect bboxInPreview; // พิกัดบนพื้นที่ preview (ยังไม่ scale ตาม canvas)
  final double? medianRel;  // 0..1 (มาก=ใกล้)
  final double areaFrac;    // สัดส่วนพื้นที่ blob / ภาพทั้งหมด
  _BlobResult({required this.bboxInPreview, required this.medianRel, required this.areaFrac});
}

class _BlobPainter extends CustomPainter {
  final _BlobResult? blob;
  final bool mirrorX; // วาดส่องกระจกถ้ากล้องหน้า
  final double srcW;  // ความกว้างของพรีวิวจากกล้อง
  final double srcH;  // ความสูงของพรีวิวจากกล้อง
  const _BlobPainter({
    required this.blob,
    this.mirrorX = false,
    required this.srcW,
    required this.srcH,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (blob == null) return;

    final scaleX = size.width  / srcW;
    final scaleY = size.height / srcH;

    Rect rect = Rect.fromLTWH(
      blob!.bboxInPreview.left   * scaleX,
      blob!.bboxInPreview.top    * scaleY,
      blob!.bboxInPreview.width  * scaleX,
      blob!.bboxInPreview.height * scaleY,
    );

    if (mirrorX) {
      rect = Rect.fromLTRB(
        size.width - rect.right,
        rect.top,
        size.width - rect.left,
        rect.bottom,
      );
    }

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.greenAccent;

    final bg = Paint()..color = Colors.black.withOpacity(0.55);

    canvas.drawRect(rect, boxPaint);

    final zText = (blob!.medianRel != null)
        ? (_metersPerUnit > 0
            ? "≈${(_metersPerUnit / ((blob!.medianRel ?? 0)+1e-3)).toStringAsFixed(2)} m"
            : "z=${(blob!.medianRel ?? 0).toStringAsFixed(2)}")
        : "—";
    final label = "Obstacle • $zText";
    final tp = _tp(label);
    const pad = 4.0;
    final bgRect = Rect.fromLTWH(
      rect.left,
      math.max(0, rect.top - tp.height - 2 * pad),
      tp.width + 2 * pad,
      tp.height + 2 * pad,
    );
    canvas.drawRect(bgRect, bg);
    tp.paint(canvas, Offset(bgRect.left + pad, bgRect.top + pad));
  }

  TextPainter _tp(String s) => TextPainter(
        text: TextSpan(text: s, style: const TextStyle(color: Colors.white, fontSize: 12)),
        textDirection: TextDirection.ltr,
        maxLines: 1,
      )..layout();

  @override
  bool shouldRepaint(covariant _BlobPainter old) =>
      old.blob != blob || old.mirrorX != mirrorX || old.srcW != srcW || old.srcH != srcH;
}

class _DepthField {
  final Float32List norm; // length = w*h, 0..1 (มาก=ใกล้)
  final int w;
  final int h;
  final double targetW; // ขนาดพรีวิว
  final double targetH;
  _DepthField({required this.norm, required this.w, required this.h, required this.targetW, required this.targetH});
}

// ======================= Isolate init payload =======================
class _IsoInit {
  final SendPort replyPort;
  final TransferableTypedData modelBytes;
  final List<String> labels;
  final int threads;
  final bool useNNAPI;
  final bool preferGPU;
  final double scoreThreshold;
  const _IsoInit({
    required this.replyPort,
    required this.modelBytes,
    required this.labels,
    required this.threads,
    required this.useNNAPI,
    required this.preferGPU,
    required this.scoreThreshold,
  });
}

// ======================= DEPTH ISOLATE =======================
void _depthIsolateEntry(_IsoInit init) {
  final SendPort reply = init.replyPort;
  final inbox = ReceivePort();
  reply.send({'type': 'ready', 'port': inbox.sendPort});

  final Uint8List model = init.modelBytes.materialize().asUint8List();
  final opts = tfl.InterpreterOptions()
    ..threads = init.threads
    ..useNnApiForAndroid = init.useNNAPI;
  if (init.preferGPU) {
    try {
      final gpu = tfl.GpuDelegateV2(options: tfl.GpuDelegateOptionsV2(isPrecisionLossAllowed: true));
      opts.addDelegate(gpu);
    } catch (_) {}
  }
  final interpreter = tfl.Interpreter.fromBuffer(model, options: opts);
  final inT = interpreter.getInputTensor(0);
  final H = inT.shape[1];
  final W = inT.shape[2];

  List? depthOut;
  void ensureOutBuf() {
    if (depthOut != null) return;
    depthOut = _zerosByShape(interpreter.getOutputTensor(0).shape, asFloat: true); // [1,h,w,1] or [1,h,w]
  }

  inbox.listen((msg) {
    if (msg is Map && msg['type'] == 'frame') {
      final sw = Stopwatch()..start();

      final camW = msg['w'] as int;
      final camH = msg['h'] as int;
      final targetW = (msg['previewW'] as num).toDouble();
      final targetH = (msg['previewH'] as num).toDouble();
      final rotateCW = msg['rotateCW'] as bool? ?? false;
      final makeHeatmap = msg['makeHeatmap'] as bool? ?? false;

      final yTTD = msg['y'] as TransferableTypedData;
      final uTTD = msg['u'] as TransferableTypedData;
      final vTTD = msg['v'] as TransferableTypedData;
      final yBytes = yTTD.materialize().asUint8List();
      final uBytes = uTTD.materialize().asUint8List();
      final vBytes = vTTD.materialize().asUint8List();

      final yRowStride = msg['yRowStride'] as int;
      final uvRowStride = msg['uRowStride'] as int;
      final uvPixelStride = msg['uvPixelStride'] as int;

      // YUV → RGB (bilinear เฉพาะ Luma)
      final flat = _yuvToRgbFlatBilinearLuma(
        yBytes: yBytes,
        uBytes: uBytes,
        vBytes: vBytes,
        srcW: camW,
        srcH: camH,
        yRowStride: yRowStride,
        uvRowStride: uvRowStride,
        uvPixelStride: uvPixelStride,
        dstW: W,
        dstH: H,
        asFloat01: true,
        rotateCW: rotateCW,
      ) as Float32List;

      final input4D = _build4DFloat(flat, H, W);

      ensureOutBuf();
      interpreter.run(input4D, depthOut!);

      // อ่านผล + normalize 0..1 (มาก=ใกล้)
      final outShape = interpreter.getOutputTensor(0).shape;
      final hOut = outShape[1];
      final wOut = outShape[2];
      final is4D = outShape.length == 4;

      double minV = double.infinity, maxV = -double.infinity;
      final tmp = Float32List(wOut * hOut);
      int k = 0;
      for (int y = 0; y < hOut; y++) {
        final row = depthOut![0][y];
        for (int x = 0; x < wOut; x++) {
          final v = is4D ? (row[x][0] as num?)?.toDouble() ?? 0.0
                         : (row[x]     as num?)?.toDouble() ?? 0.0;
          tmp[k++] = v;
          if (v < minV) minV = v;
          if (v > maxV) maxV = v;
        }
      }
      final range = (maxV - minV).abs() < 1e-6 ? 1e-6 : (maxV - minV);
      for (int i = 0; i < tmp.length; i++) {
        tmp[i] = (tmp[i] - minV) / range; // 0..1 (มาก=ใกล้)
      }

      // heatmap PNG (เฉพาะเมื่อเปิด)
      Uint8List? png;
      if (makeHeatmap) {
        final heat = img.Image(width: wOut, height: hOut);
        for (int y = 0; y < hOut; y++) {
          final base = y * wOut;
          for (int x = 0; x < wOut; x++) {
            final vv = tmp[base + x].clamp(0.0, 1.0);
            final c = _colormapJet(vv);
            heat.setPixelRgba(x, y, c[0], c[1], c[2], 140);
          }
        }
        final up = img.copyResize(
          heat,
          width: targetW.toInt(),
          height: targetH.toInt(),
          interpolation: img.Interpolation.linear,
        );
        png = Uint8List.fromList(img.encodePng(up));
      }

      reply.send({
        'type': 'depth',
        'durMs': sw.elapsedMilliseconds,
        'w': wOut,
        'h': hOut,
        'targetW': targetW,
        'targetH': targetH,
        'norm': TransferableTypedData.fromList([Uint8List.view(tmp.buffer)]),
        'png': png,
      });
    }
  });
}

// ======================= Shared helpers =======================
List _zerosByShape(List<int> shape, {required bool asFloat}) {
  dynamic build(List<int> s) {
    if (s.isEmpty) return asFloat ? 0.0 : 0;
    return List.generate(s.first, (_) => build(s.sublist(1)));
  }
  return build(shape);
}

// ---- YUV420 → RGB (bilinear เฉพาะ Luma(Y), Chroma(UV) = nearest) ----
Object _yuvToRgbFlatBilinearLuma({
  required Uint8List yBytes,
  required Uint8List uBytes,
  required Uint8List vBytes,
  required int srcW,
  required int srcH,
  required int yRowStride,
  required int uvRowStride,
  required int uvPixelStride,
  required int dstW,
  required int dstH,
  required bool asFloat01,
  bool rotateCW = false,
}) {
  final virtW = rotateCW ? srcH : srcW;
  final virtH = rotateCW ? srcW : srcH;
  final len = dstH * dstW * 3;

  void mapSrc(int vx, int vy, List<int> out) {
    final sx = rotateCW ? vy : vx;
    final sy = rotateCW ? (srcW - 1 - vx) : vy;
    out[0] = sx; out[1] = sy;
  }

  if (asFloat01) {
    final out = Float32List(len);
    int o = 0;
    final temp = List<int>.filled(2, 0);

    for (int y = 0; y < dstH; y++) {
      final fy = (y * (virtH - 1)) / (dstH - 1);
      final y0 = fy.floor().clamp(0, virtH - 1);
      final y1 = (y0 + 1).clamp(0, virtH - 1);
      final wy = fy - y0;

      for (int x = 0; x < dstW; x++) {
        final fx = (x * (virtW - 1)) / (dstW - 1);
        final x0 = fx.floor().clamp(0, virtW - 1);
        final x1 = (x0 + 1).clamp(0, virtW - 1);
        final wx = fx - x0;

        mapSrc(x0, y0, temp); final sx00 = temp[0], sy00 = temp[1];
        mapSrc(x1, y0, temp); final sx10 = temp[0], sy10 = temp[1];
        mapSrc(x0, y1, temp); final sx01 = temp[0], sy01 = temp[1];
        mapSrc(x1, y1, temp); final sx11 = temp[0], sy11 = temp[1];

        final y00 = yBytes[(sy00 * yRowStride + sx00).clamp(0, yBytes.length - 1).toInt()] & 0xFF;
        final y10 = yBytes[(sy10 * yRowStride + sx10).clamp(0, yBytes.length - 1).toInt()] & 0xFF;
        final y01 = yBytes[(sy01 * yRowStride + sx01).clamp(0, yBytes.length - 1).toInt()] & 0xFF;
        final y11 = yBytes[(sy11 * yRowStride + sx11).clamp(0, yBytes.length - 1).toInt()] & 0xFF;

        final yTop = y00 * (1 - wx) + y10 * wx;
        final yBot = y01 * (1 - wx) + y11 * wx;
        final Yf = yTop * (1 - wy) + yBot * wy;

        mapSrc(fx.round(), fy.round(), temp);
        final csx = temp[0], csy = temp[1];
        final cuRow = (csy >> 1) * uvRowStride;
        final cuCol = (csx >> 1) * uvPixelStride;
        final uIdx = (cuRow + cuCol).clamp(0, uBytes.length - 1).toInt();
        final vIdx = (cuRow + cuCol).clamp(0, vBytes.length - 1).toInt();
        final U = (uBytes[uIdx] & 0xFF) - 128;
        final V = (vBytes[vIdx] & 0xFF) - 128;

        int r = (Yf + ((1436 * V) >> 10)).round();
        int g = (Yf - ((352 * U + 731 * V) >> 10)).round();
        int b = (Yf + ((1815 * U) >> 10)).round();

        if (r < 0) r = 0; else if (r > 255) r = 255;
        if (g < 0) g = 0; else if (g > 255) g = 255;
        if (b < 0) b = 0; else if (b > 255) b = 255;

        out[o++] = r / 255.0;
        out[o++] = g / 255.0;
        out[o++] = b / 255.0;
      }
    }
    return out;
  } else {
    final out = Uint8List(len);
    int o = 0;
    final temp = List<int>.filled(2, 0);

    for (int y = 0; y < dstH; y++) {
      final fy = (y * (virtH - 1)) / (dstH - 1);
      final y0 = fy.floor().clamp(0, virtH - 1);
      final y1 = (y0 + 1).clamp(0, virtH - 1);
      final wy = fy - y0;

      for (int x = 0; x < dstW; x++) {
        final fx = (x * (virtW - 1)) / (dstW - 1);
        final x0 = fx.floor().clamp(0, virtW - 1);
        final x1 = (x0 + 1).clamp(0, virtW - 1);
        final wx = fx - x0;

        mapSrc(x0, y0, temp); final sx00 = temp[0], sy00 = temp[1];
        mapSrc(x1, y0, temp); final sx10 = temp[0], sy10 = temp[1];
        mapSrc(x0, y1, temp); final sx01 = temp[0], sy01 = temp[1];
        mapSrc(x1, y1, temp); final sx11 = temp[0], sy11 = temp[1];

        final y00 = yBytes[(sy00 * yRowStride + sx00).clamp(0, yBytes.length - 1).toInt()] & 0xFF;
        final y10 = yBytes[(sy10 * yRowStride + sx10).clamp(0, yBytes.length - 1).toInt()] & 0xFF;
        final y01 = yBytes[(sy01 * yRowStride + sx01).clamp(0, yBytes.length - 1).toInt()] & 0xFF;
        final y11 = yBytes[(sy11 * yRowStride + sx11).clamp(0, yBytes.length - 1).toInt()] & 0xFF;

        final yTop = y00 * (1 - wx) + y10 * wx;
        final yBot = y01 * (1 - wx) + y11 * wx;
        final Yf = yTop * (1 - wy) + yBot * wy;

        mapSrc(fx.round(), fy.round(), temp);
        final csx = temp[0], csy = temp[1];
        final cuRow = (csy >> 1) * uvRowStride;
        final cuCol = (csx >> 1) * uvPixelStride;
        final uIdx = (cuRow + cuCol).clamp(0, uBytes.length - 1).toInt();
        final vIdx = (cuRow + cuCol).clamp(0, vBytes.length - 1).toInt();
        final U = (uBytes[uIdx] & 0xFF) - 128;
        final V = (vBytes[vIdx] & 0xFF) - 128;

        int r = (Yf + ((1436 * V) >> 10)).round();
        int g = (Yf - ((352 * U + 731 * V) >> 10)).round();
        int b = (Yf + ((1815 * U) >> 10)).round();

        if (r < 0) r = 0; else if (r > 255) r = 255;
        if (g < 0) g = 0; else if (g > 255) g = 255;
        if (b < 0) b = 0; else if (b > 255) b = 255;

        out[o++] = r;
        out[o++] = g;
        out[o++] = b;
      }
    }
    return out;
  }
}

List _build4DFloat(Float32List flat, int H, int W) {
  int idx = 0;
  return [
    List.generate(H, (_) =>
      List.generate(W, (_) {
        final r = flat[idx++]; final g = flat[idx++]; final b = flat[idx++];
        return [r, g, b];
      })
    )
  ];
}

List<int> _colormapJet(double v) {
  v = v.clamp(0.0, 1.0);
  final fourV = 4 * v;
  final r = (255 * (fourV - 1.5).clamp(0.0, 1.0)).toInt();
  final g = (255 * (1.5 - (fourV - 2.0).abs()).clamp(0.0, 1.0)).toInt();
  final b = (255 * (1.5 - (fourV - 0.5).abs()).clamp(0.0, 1.0)).toInt();
  return [r, g, b];
}