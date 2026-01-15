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

// รอบส่งเฟรมไปแต่ละ pipeline (มิลลิวินาที)
const int kDetIntervalMs   = 66;   // ~15 FPS
const int kDepthIntervalMs = 90;   // ~11 FPS

// Detection sensitivity (พื้นฐาน)
const double kScoreThreshold = 0.40; // ลดเพื่อเพิ่ม recall (ยังมี guard ใน isolate)
const int kTfliteThreads = 4;

// Depth → โหมดสัมพัทธ์ 0..1 (มาก=ใกล้). ตั้ง >0 ถ้าคาลิเบรตเป็นเมตรได้
double _metersPerUnit = 0.0;

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
      title: 'AI Depth/Object App (Person-Boost)',
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

  // Labels
  List<String> _labels = [];

  // Isolates & Ports
  Isolate? _detIso;
  Isolate? _depIso;
  SendPort? _detInPort;
  SendPort? _depInPort;
  late final ReceivePort _detOutPort;
  late final ReceivePort _depOutPort;
  bool _detBusy = false;
  bool _depBusy = false;
  int _lastDetSendMs = 0;
  int _lastDepSendMs = 0;

  // FPS / timing (ทันที)
  final Stopwatch _fpsWindow = Stopwatch()..start();
  int _framesInWindow = 0;
  double _fpsNow = 0.0;
  int _lastDetDurMs = 0;
  int _lastDepDurMs = 0;

  // ===== FPS Round Logger (ใหม่) =====
  // เก็บค่า fps รายวินาที เป็นรอบละ 60 วินาที สูงสุด 10 รอบ (เลื่อนหน้าต่าง)
  final List<List<double>> _fpsRounds = <List<double>>[];
  int _currentRoundIndex = 0;     // index ของรอบปัจจุบันใน _fpsRounds
  int _secondsInCurrentRound = 0; // นับ 0..59
  Timer? _fpsRoundTimer;

  // UI state
  String _status = "Initializing...";
  bool _showHeatmap = false; // ปิดเริ่มต้นเพื่อความลื่น
  Uint8List? _depthOverlayPng;

  // ผลลัพธ์ล่าสุด (อิสระ)
  List<_Detection> _lastDetections = [];
  _DepthField? _lastDepthField;

  // สำหรับวาด
  List<_Detection> _drawDetections = [];

  @override
  void initState() {
    super.initState();
    _initAll();
  }

  @override
  void dispose() {
    _controller?.dispose();
    _detIso?.kill(priority: Isolate.immediate);
    _depIso?.kill(priority: Isolate.immediate);
    _fpsRoundTimer?.cancel();
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

      // 2) Labels + Model bytes
      _labels = await _loadLabels('assets/models/labels.txt'); // COCO-80 (หรือ 90/91 ที่มี background)
      final detBytes   = await rootBundle.load('assets/models/1.tflite');
      final depthBytes = await rootBundle.load('assets/models/model_opt.tflite');

      // 3) DETECTOR isolate
      _detOutPort = ReceivePort();
      _detOutPort.listen(_onDetectorMessage, onError: (e) {
        setState(() => _status = "Detector isolate error: $e");
      });
      _detIso = await Isolate.spawn(
        _detectorIsolateEntry,
        _IsoInit(
          replyPort: _detOutPort.sendPort,
          modelBytes: TransferableTypedData.fromList([detBytes.buffer.asUint8List()]),
          labels: _labels,
          threads: kTfliteThreads,
          useNNAPI: kEnableNNAPI,
          preferGPU: false,
          scoreThreshold: kScoreThreshold,
        ),
        errorsAreFatal: true,
        debugName: "detector_isolate",
      );

      // 4) DEPTH isolate
      _depOutPort = ReceivePort();
      _depOutPort.listen(_onDepthMessage, onError: (e) {
        setState(() => _status = "Depth isolate error: $e");
      });
      _depIso = await Isolate.spawn(
        _depthIsolateEntry,
        _IsoInit(
          replyPort: _depOutPort.sendPort,
          modelBytes: TransferableTypedData.fromList([depthBytes.buffer.asUint8List()]),
          labels: const [], // ไม่ใช้
          threads: kTfliteThreads,
          useNNAPI: kEnableNNAPI,
          preferGPU: kEnableGPUForDepth,
          scoreThreshold: 0.0,
        ),
        errorsAreFatal: true,
        debugName: "depth_isolate",
      );

      // 5) Camera stream → ส่งแบบ latest-only (ไม่ทำคิว)
      await _controller!.startImageStream((frame) {
        final now = DateTime.now().millisecondsSinceEpoch;

        if (!_detBusy && (now - _lastDetSendMs) >= kDetIntervalMs && _detInPort != null) {
          _detInPort!.send(_packYuvFrame(frame));
          _detBusy = true;
          _lastDetSendMs = now;
        }

        if (!_depBusy && (now - _lastDepSendMs) >= kDepthIntervalMs && _depInPort != null) {
          _depInPort!.send(_packYuvFrame(frame, makeHeatmap: _showHeatmap));
          _depBusy = true;
          _lastDepSendMs = now;
        }

        // FPS overlay (อัปเดตทุก ~0.5s)
        _framesInWindow++;
        if (_fpsWindow.elapsedMilliseconds >= 500) {
          _fpsNow = (_framesInWindow * 1000) / _fpsWindow.elapsedMilliseconds;
          _framesInWindow = 0;
          _fpsWindow..reset()..start();
          if (mounted) setState(() {});
        }
      });

      // 6) เริ่มตัวจับรอบ FPS (อ่านทุก 1 วินาที, 1 รอบ = 60 วินาที, สูงสุด 10 รอบ)
      _startFpsRoundLogger();

      setState(() => _status = "Real-time running ✅");
    } catch (e) {
      setState(() => _status = "Init error: $e");
    }
  }

  // ===== FPS Round Logger =====
  void _startFpsRoundLogger() {
    // สร้างรอบแรกถ้ายังไม่มี
    if (_fpsRounds.isEmpty) {
      _fpsRounds.add(<double>[]);
      _currentRoundIndex = 0;
      _secondsInCurrentRound = 0;
    }

    _fpsRoundTimer?.cancel();
    _fpsRoundTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      // บันทึกค่า FPS ล่าสุด
      final currentList = _fpsRounds[_currentRoundIndex];
      if (currentList.length < 60) {
        currentList.add(_fpsNow);
      } else {
        currentList[59] = _fpsNow; // กันพลาด
      }

      _secondsInCurrentRound++;

      if (_secondsInCurrentRound >= 60) {
        // ปิดรอบนี้ แล้วเริ่มรอบใหม่
        _secondsInCurrentRound = 0;

        // ถ้ายังไม่ครบ 10 รอบ → เพิ่มใหม่ต่อท้าย
        // ถ้าเกินแล้ว → ลบรอบแรก (เก่าสุด) แล้วเพิ่มใหม่ต่อท้าย (เลื่อนหน้าต่าง)
        if (_fpsRounds.length < 10) {
          _fpsRounds.add(<double>[]);
          _currentRoundIndex = _fpsRounds.length - 1;
        } else {
          _fpsRounds.removeAt(0);
          _fpsRounds.add(<double>[]);
          _currentRoundIndex = _fpsRounds.length - 1;
        }

        // ✅ ครบ 10 รอบแล้ว พิมพ์ค่าทั้งหมดของทั้ง 10 รอบออก Console ทุกครั้งที่ปิดรอบ
        if (_fpsRounds.length == 10) {
          _printAllFpsRounds();
        }

        if (mounted) setState(() {});
      }
    });
  }

  // ✅ พิมพ์ค่าทั้งหมดของ 10 รอบ (CSV: sec,fps) + avg ต่อรอบ
  void _printAllFpsRounds() {
    print("=========== FPS ROUNDS (10 x 60s) ===========");
    for (int i = 0; i < _fpsRounds.length; i++) {
      final l = _fpsRounds[i];
      final avg = l.isEmpty ? 0.0 : (l.reduce((a, b) => a + b) / l.length);
      print("[Round ${i + 1}] seconds=${l.length} avg=${avg.toStringAsFixed(2)}");

      // พิมพ์เป็น CSV ทีละก้อน กัน log ยาวเกิน
      var buf = StringBuffer("sec,fps\n");
      for (int s = 0; s < l.length; s++) {
        buf.write("${s + 1},${l[s].toStringAsFixed(2)}\n");
        // flush ทุกๆ 120 แถว (กันโดนตัดใน log)
        if ((s + 1) % 120 == 0) {
          print(buf.toString());
          buf = StringBuffer("sec,fps\n");
        }
      }
      // flush ส่วนที่เหลือของรอบ
      final left = buf.toString();
      if (left.trim().isNotEmpty) {
        print(left);
      }
    }
    print("============= END FPS ROUNDS =============");
  }

  // ค่าช่วยคำนวณเฉลี่ยของรอบล่าสุด (เพื่อแสดงใน overlay)
  double _avgOfRound(int idx) {
    if (idx < 0 || idx >= _fpsRounds.length) return 0.0;
    final l = _fpsRounds[idx];
    if (l.isEmpty) return 0.0;
    double s = 0.0;
    for (final v in l) { s += v; }
    return s / l.length;
  }

  // ====== Detector messages ======
  void _onDetectorMessage(dynamic message) {
    if (message is Map && message['type'] == 'ready') {
      _detInPort = message['port'] as SendPort;
      setState(() {});
      return;
    }
    if (message is Map && message['type'] == 'det') {
      _detBusy = false;
      _lastDetDurMs = message['durMs'] as int? ?? 0;

      final List detList = message['detections'] as List? ?? const [];
      final parsed = <_Detection>[];
      for (final e in detList) {
        final l = (e[0] as num).toDouble();
        final t = (e[1] as num).toDouble();
        final r = (e[2] as num).toDouble();
        final b = (e[3] as num).toDouble();
        final s = (e[4] as num).toDouble();

        final mappedIdx = (e.length >= 6) ? (e[5] as num).toInt() : -1;
        final nameFromIso = (e.length >= 7 && e[6] is String) ? (e[6] as String) : null;

        String label;
        if (nameFromIso != null && nameFromIso.isNotEmpty) {
          label = nameFromIso;
        } else if (mappedIdx >= 0 && mappedIdx < _labels.length) {
          label = _labels[mappedIdx];
        } else {
          label = "id:${(e.length >= 6) ? e[5] : '??'}";
        }

        final low = label.toLowerCase();
        final isBg = (label == '???') || low.contains('background');
        if (isBg) continue;

        parsed.add(_Detection(rect: Rect.fromLTRB(l, t, r, b), label: label, score: s));
      }
      _lastDetections = parsed;
      _refreshOverlay();
      return;
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

      _refreshOverlay();
      return;
    }
  }

  void _refreshOverlay() {
    if (!mounted) return;

    final enriched = (_lastDepthField != null)
        ? _attachDepthToDetections(_lastDetections, _lastDepthField!)
        : _lastDetections;

    setState(() {
      _drawDetections = enriched;
      _status =
          "Parallel ✅  obj:${enriched.length} • det:$_lastDetDurMs ms/$kDetIntervalMs • depth:$_lastDepDurMs ms/$kDepthIntervalMs";
    });
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

  // === โหลด labels: ใช้ตามไฟล์จริง, ไม่ยัด '???' อัตโนมัติ ===
  Future<List<String>> _loadLabels(String assetPath) async {
    final raw = await rootBundle.loadString(assetPath);
    final lines = raw
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();
    return lines;
  }

  // แนบ median depth (0..1) เข้าไปในกรอบ
  List<_Detection> _attachDepthToDetections(List<_Detection> dets, _DepthField f) {
    if (f.w == 0 || f.h == 0 || f.norm.isEmpty) return dets;
    final sx = f.w / f.targetW;
    final sy = f.h / f.targetH;

    final result = <_Detection>[];
    for (final d in dets) {
      final l = (d.rect.left   * sx).clamp(0, f.w - 1).toDouble();
      final t = (d.rect.top    * sy).clamp(0, f.h - 1).toDouble();
      final r = (d.rect.right  * sx).clamp(0, f.w - 1).toDouble();
      final b = (d.rect.bottom * sy).clamp(0, f.h - 1).toDouble();
      if (r <= l || b <= t) { result.add(d); continue; }

      const int S = 6;
      final stepX = math.max(1.0, (r - l) / (S - 1));
      final stepY = math.max(1.0, (b - t) / (S - 1));
      final samples = <double>[];

      for (double yy = t; yy <= b; yy += stepY) {
        final yi = yy.round().clamp(0, f.h - 1);
        final base = yi * f.w;
        for (double xx = l; xx <= r; xx += stepX) {
          final xi = xx.round().clamp(0, f.w - 1);
          samples.add(f.norm[base + xi]);
        }
      }
      samples.sort();
      final median = samples[samples.length ~/ 2];

      String depthText = "z=${median.toStringAsFixed(2)} (rel)";
      if (_metersPerUnit > 0) {
        const eps = 1e-3;
        depthText = "≈${(_metersPerUnit / (median + eps)).toStringAsFixed(2)} m";
      }
      result.add(d.copyWith(extra: depthText));
    }
    return result;
  }

  @override
  Widget build(BuildContext context) {
    final camReady = _controller?.value.isInitialized == true;
    final previewSize = _controller?.value.previewSize;

    final roundsCount = _fpsRounds.length;
    final currentAvg = _avgOfRound(_currentRoundIndex);
    final roundSummary =
        "Round ${_currentRoundIndex + 1}/$roundsCount • avg ${currentAvg.toStringAsFixed(1)} • ${_secondsInCurrentRound}s/60";

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
                        IgnorePointer(
                          child: CustomPaint(
                            painter: _DetectionsPainter(
                              _drawDetections,
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
                // Status (กลางบน)
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
                // FPS/งบเวลา (ซ้ายบน)
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
                        "FPS: ${_fpsNow.toStringAsFixed(1)} • det:$_lastDetDurMs ms/$kDetIntervalMs • depth:$_lastDepDurMs ms/$kDepthIntervalMs",
                        style: const TextStyle(color: Colors.white, fontSize: 12),
                      ),
                    ),
                  ),
                ),
                // Note depth (ล่างกลาง)
                if (_metersPerUnit == 0.0)
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
                        child: const Text(
                          "Depth = relative (0..1, มาก=ใกล้). คาลิเบรตเพื่อแสดงเป็นเมตรได้",
                          style: TextStyle(color: Colors.white, fontSize: 12),
                        ),
                      ),
                    ),
                  ),
                // Toggle Heatmap (ขวาบน)
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
                // สรุปรอบ FPS (ขวาล่าง)
                SafeArea(
                  child: Align(
                    alignment: Alignment.bottomRight,
                    child: Container(
                      margin: const EdgeInsets.only(right: 10, bottom: 10),
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.45),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        roundSummary,
                        style: const TextStyle(color: Colors.white, fontSize: 12),
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

// ======================= Data classes / Painter =======================
class _Detection {
  final Rect rect; // พิกเซลใน "พรีวิว" (ยังไม่ scale ตาม canvas)
  final String label;
  final double score;
  final String? extra;
  _Detection({required this.rect, required this.label, required this.score, this.extra});

  _Detection copyWith({Rect? rect, String? label, double? score, String? extra}) {
    return _Detection(
      rect: rect ?? this.rect,
      label: label ?? this.label,
      score: score ?? this.score,
      extra: extra ?? this.extra,
    );
  }
}

class _DetectionsPainter extends CustomPainter {
  final List<_Detection> dets;
  final bool mirrorX; // วาดแบบส่องกระจก (สำหรับกล้องหน้า)
  final double srcW;  // ความกว้างของพรีวิวจากกล้อง
  final double srcH;  // ความสูงของพรีวิวจากกล้อง
  _DetectionsPainter(this.dets, {this.mirrorX = false, required this.srcW, required this.srcH});

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width  / srcW;
    final scaleY = size.height / srcH;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.greenAccent;

    final bg = Paint()..color = Colors.black.withOpacity(0.55);

    for (final d in dets) {
      Rect rect = Rect.fromLTRB(
        d.rect.left   * scaleX,
        d.rect.top    * scaleY,
        d.rect.right  * scaleX,
        d.rect.bottom * scaleY,
      );

      if (mirrorX) {
        rect = Rect.fromLTRB(
          size.width - rect.right,
          rect.top,
          size.width - rect.left,
          rect.bottom,
        );
      }

      canvas.drawRect(rect, boxPaint);

      final text = "${d.label} ${d.score.toStringAsFixed(2)}"
          "${d.extra != null ? " • ${d.extra}" : ""}";
      final tp = _tp(text);
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
  }

  TextPainter _tp(String s) => TextPainter(
        text: TextSpan(text: s, style: const TextStyle(color: Colors.white, fontSize: 12)),
        textDirection: TextDirection.ltr,
        maxLines: 1,
      )..layout();

  @override
  bool shouldRepaint(covariant _DetectionsPainter oldDelegate) =>
      oldDelegate.dets != dets || oldDelegate.mirrorX != mirrorX || oldDelegate.srcW != srcW || oldDelegate.srcH != srcH;
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

// ======================= DETECTOR ISOLATE =======================
void _detectorIsolateEntry(_IsoInit init) {
  final SendPort reply = init.replyPort;

  final inbox = ReceivePort();
  reply.send({'type': 'ready', 'port': inbox.sendPort});

  // Interpreter
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

  final inT = interpreter.getInputTensor(0); // [1,H,W,3]
  final detInH = inT.shape[1];
  final detInW = inT.shape[2];
  final isFloatInput = inT.type.toString().toLowerCase().contains('float32');

  List? out0, out1, out2, out3;
  void ensureOutBufs() {
    if (out0 != null) return;
    out0 = _zerosByShape(interpreter.getOutputTensor(0).shape, asFloat: true); // boxes [1,N,4]
    out1 = _zerosByShape(interpreter.getOutputTensor(1).shape, asFloat: true); // classes [1,N]
    out2 = _zerosByShape(interpreter.getOutputTensor(2).shape, asFloat: true); // scores [1,N]
    if (interpreter.getOutputTensors().length >= 4) {
      out3 = _zerosByShape(interpreter.getOutputTensor(3).shape, asFloat: true); // count [1] or [1,1]
    }
  }

  final lowerLabels = init.labels.map((s) => s.toLowerCase()).toList();
  final int personIdx = lowerLabels.indexOf('person');

  int mapClassId(int rawCls) {
    if (init.labels.isEmpty) return rawCls;
    final first = init.labels.first.toLowerCase();
    final hasBg = first == 'background' || first == '???' || first.contains('background');
    int idx = rawCls + (hasBg ? 1 : 0);
    if (idx < 0) idx = 0;
    if (idx >= init.labels.length) idx = init.labels.length - 1;
    return idx;
  }

  inbox.listen((msg) {
    if (msg is Map && msg['type'] == 'frame') {
      final sw = Stopwatch()..start();

      final camW = msg['w'] as int;
      final camH = msg['h'] as int;
      final previewW = (msg['previewW'] as num).toDouble();
      final previewH = (msg['previewH'] as num).toDouble();
      final rotateCW = msg['rotateCW'] as bool? ?? false;

      final yTTD = msg['y'] as TransferableTypedData;
      final uTTD = msg['u'] as TransferableTypedData;
      final vTTD = msg['v'] as TransferableTypedData;
      final yBytes = yTTD.materialize().asUint8List();
      final uBytes = uTTD.materialize().asUint8List();
      final vBytes = vTTD.materialize().asUint8List();

      final yRowStride = msg['yRowStride'] as int;
      final uvRowStride = msg['uRowStride'] as int;
      final uvPixelStride = msg['uvPixelStride'] as int;

      final flat = _yuvToRgbFlatBilinearLuma(
        yBytes: yBytes,
        uBytes: uBytes,
        vBytes: vBytes,
        srcW: camW,
        srcH: camH,
        yRowStride: yRowStride,
        uvRowStride: uvRowStride,
        uvPixelStride: uvPixelStride,
        dstW: detInW,
        dstH: detInH,
        asFloat01: isFloatInput,
        rotateCW: rotateCW,
      );

      final input4D = isFloatInput
          ? _build4DFloat(flat as Float32List, detInH, detInW)
          : _build4DUint8(flat as Uint8List, detInH, detInW);

      ensureOutBufs();
      final outMap = <int, Object>{0: out0!, 1: out1!, 2: out2!, if (out3 != null) 3: out3!};
      interpreter.runForMultipleInputs([input4D], outMap);

      final boxes = out0!;
      final classes = out1!;
      final scores = out2!;
      if (boxes.isEmpty || classes.isEmpty || scores.isEmpty) {
        reply.send({'type': 'det', 'durMs': sw.elapsedMilliseconds, 'detections': const []});
        return;
      }

      int nCand = (scores[0] as List).length;
      if (out3 != null && out3!.isNotEmpty) {
        final c = (out3![0] is List && out3![0].isNotEmpty)
            ? ((out3![0][0] as num?)?.toInt() ?? nCand)
            : ((out3![0] as num?)?.toInt() ?? nCand);
        if (c > 0 && c <= nCand) nCand = c;
      }

      const int    topKBase   = 220;
      const double iouBase    = 0.55;
      const double iouPerson  = 0.70;
      final double scoreBase  = (init.scoreThreshold < 0.10 ? 0.10
                              : (init.scoreThreshold > 0.60 ? 0.60 : init.scoreThreshold));
      const double minFracBase   = 0.0010;
      const double minFracPerson = 0.00035;
      const double maxFracBase   = 0.95;

      final totalArea = previewW * previewH;
      final cand = <List>[];

      for (int i = 0; i < nCand; i++) {
        final s = (scores[0][i] as num?)?.toDouble() ?? 0.0;

        final rawCls = (classes[0][i] as num?)?.toInt() ?? -1;
        final mappedCls = mapClassId(rawCls);
        final label = (mappedCls >= 0 && mappedCls < init.labels.length)
            ? init.labels[mappedCls]
            : 'id:$rawCls';

        final low = label.toLowerCase();
        final isBg = (label == '???') || low.contains('background');

        final bool isPerson = (!isBg && personIdx >= 0 && mappedCls == personIdx);
        final double sTh = isPerson ? (scoreBase * 0.55) : scoreBase;
        if (s < sTh) continue;

        final loc = (boxes[0][i] as List?);
        if (loc == null || loc.length < 4) continue;
        final ymin = (loc[0] as num?)?.toDouble().clamp(0.0, 1.0) ?? 0.0;
        final xmin = (loc[1] as num?)?.toDouble().clamp(0.0, 1.0) ?? 0.0;
        final ymax = (loc[2] as num?)?.toDouble().clamp(0.0, 1.0) ?? 0.0;
        final xmax = (loc[3] as num?)?.toDouble().clamp(0.0, 1.0) ?? 0.0;
        if (xmax <= xmin || ymax <= ymin) continue;

        final left   = xmin * previewW;
        final top    = ymin * previewH;
        final right  = xmax * previewW;
        final bottom = ymax * previewH;

        final w = (right - left).abs();
        final h = (bottom - top).abs();
        final frac = (w * h) / totalArea;

        final double minFrac = isPerson ? minFracPerson : minFracBase;
        if (frac < minFrac || frac > maxFracBase) continue;

        if (isBg) continue;

        cand.add([left, top, right, bottom, s, mappedCls, label]);
      }

      if (cand.length > 1) cand.sort((a, b) => (b[4] as double).compareTo(a[4] as double));
      final preNms = (cand.length > topKBase) ? cand.sublist(0, topKBase) : cand;

      final picked = _nmsPerClassWithCustomIoU(
        preNms,
        defaultIou: iouBase,
        specialClass: personIdx >= 0 ? personIdx : null,
        specialIou: iouPerson,
      );

      reply.send({'type': 'det', 'durMs': sw.elapsedMilliseconds, 'detections': picked});
    }
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

// ======================= Shared helpers (both isolates) =======================
List _zerosByShape(List<int> shape, {required bool asFloat}) {
  dynamic build(List<int> s) {
    if (s.isEmpty) return asFloat ? 0.0 : 0;
    return List.generate(s.first, (_) => build(s.sublist(1)));
  }
  return build(shape);
}

double _iouList(List a, List b) {
  final ax1 = (a[0] as num).toDouble();
  final ay1 = (a[1] as num).toDouble();
  final ax2 = (a[2] as num).toDouble();
  final ay2 = (a[3] as num).toDouble();

  final bx1 = (b[0] as num).toDouble();
  final by1 = (b[1] as num).toDouble();
  final bx2 = (b[2] as num).toDouble();
  final by2 = (b[3] as num).toDouble();

  final ix1 = math.max(ax1, bx1);
  final iy1 = math.max(ay1, by1);
  final ix2 = math.min(ax2, bx2);
  final iy2 = math.min(ay2, by2);

  final iw = math.max(0.0, ix2 - ix1);
  final ih = math.max(0.0, iy2 - iy1);
  final inter = iw * ih;

  final aArea = (ax2 - ax1).abs() * (ay2 - ay1).abs();
  final bArea = (bx2 - bx1).abs() * (by2 - by1).abs();
  final union = aArea + bArea - inter;
  if (union <= 0) return 0.0;
  return inter / union;
}

List<List> _nmsPerClassWithCustomIoU(
  List<List> dets, {
  required double defaultIou,
  int? specialClass,
  double? specialIou,
}) {
  final Map<int, List<List>> byCls = <int, List<List>>{};
  for (final d in dets) {
    final ci = (d[5] as num).toInt();
    (byCls[ci] ??= <List>[]).add(d);
  }
  final picked = <List>[];
  byCls.forEach((ci, list) {
    list.sort((a, b) => (b[4] as double).compareTo(a[4] as double));
    final kept = <List>[];
    final thisIou = (specialClass != null && specialIou != null && ci == specialClass)
        ? specialIou
        : defaultIou;
    for (final d in list) {
      bool keep = true;
      for (final p in kept) {
        if (_iouList(d, p) > thisIou) { keep = false; break; }
      }
      if (keep) kept.add(d);
    }
    picked.addAll(kept);
  });
  return picked;
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
    out[0] = sx;
    out[1] = sy;
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

List _build4DUint8(Uint8List flat, int H, int W) {
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
