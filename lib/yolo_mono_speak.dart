// ignore_for_file: prefer_const_constructors, prefer_const_literals_to_create_immutables

import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;

import 'dart:collection';
import 'package:flutter_tts/flutter_tts.dart';

// =====================================================
// =============== RUNTIME CONFIG / CONSTANTS =========
// =====================================================

// z (0..1, ยิ่งมาก=ยิ่งใกล้) → ถังระยะคร่าวๆ (พูด/โชว์ง่าย)
int zToMetersBucket(double z) {
  if (z >= 0.90) return 1;       // ~0-1 m
  if (z >= 0.50) return 2;       // ~2 m
  if (z >= 0.41) return 3;       // ~3 m
  if (z >= 0.30) return 4;       // ~4 m
  return 5;                      // ≥5 m
}

String metersBucketLabel(int m) =>
    (m >= 5) ? "5 meters or more" : "$m meter${m == 1 ? "" : "s"}";

const bool kEnableNNAPI       = false;
const bool kEnableGPUForDepth = false;

const int kDetIntervalMs   = 66;   // ~15 FPS
const int kDepthIntervalMs = 90;   // ~11 FPS

const double kScoreThreshold = 0.40;
const int kTfliteThreads = 4;

// ตั้ง >0 ถ้าคุณคาลิเบรตได้เป็นเมตรจริง (DET: m = _metersPerUnit / z)
double _metersPerUnit = 0.0;

// ===== Blob meters (ตามสั่ง): meters = 10.22 * z_center - 2.66
double blobMetersFromCenterZ(double z) => (10.22 * z) - 2.66;

// =====================================================
// ====================== APP ==========================
// =====================================================
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
      title: 'AI Depth/Object App (Centers + Blob meters)',
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
  late bool _isFront;

  // Labels
  List<String> _labels = [];

  // Isolates
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

  // Timing / FPS
  final Stopwatch _fpsWindow = Stopwatch()..start();
  int _framesInWindow = 0;
  double _fpsNow = 0.0;

  // TTS
  final FlutterTts _tts = FlutterTts();
  final Queue<String> _sayQ = Queue<String>();
  bool _isSpeaking = false;
  DateTime _lastSpokenAt = DateTime.fromMillisecondsSinceEpoch(0);
  String _lastSpokenKey = "";

  // UI state
  String _status = "Initializing...";
  bool _showHeatmap = false;
  Uint8List? _depthOverlayPng;

  // Results
  List<_Detection> _lastDetections = [];
  _DepthField? _lastDepthField;

  // For paint
  List<_Detection> _drawDetections = [];

  // Blobs always-on
  List<BlobResult> _lastBlobs = [];

  // durations
  int _lastDetDurMs = 0;
  int _lastDepDurMs = 0;

  // Speech knobs
  static const int kSpeakCooldownMs = 1500;

  @override
  void initState() {
    super.initState();
    () async {
      try {
        await _tts.setLanguage("en-US");
        await _tts.setSpeechRate(0.9);
        await _tts.awaitSpeakCompletion(true);
      } catch (_) {}
    }();
    _initAll();
  }

  @override
  void dispose() {
    _controller?.dispose();
    _detIso?.kill(priority: Isolate.immediate);
    _depIso?.kill(priority: Isolate.immediate);
    _tts.stop();
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

      // 2) Labels + Models
      _labels = await _loadLabels('assets/models/labels.txt');
      // ใช้ชื่อไฟล์ detector ตามที่สั่ง: 1.tflite
      final detBytes   = await rootBundle.load('assets/models/1.tflite');
      final depthBytes = await rootBundle.load('assets/models/model_opt.tflite');

      // 3) Detector isolate
      _detOutPort = ReceivePort();
      _detOutPort.listen(_onDetectorMessage, onError: (e) {
        if (mounted) setState(() => _status = "Detector isolate error: $e");
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

      // 4) Depth isolate
      _depOutPort = ReceivePort();
      _depOutPort.listen(_onDepthMessage, onError: (e) {
        if (mounted) setState(() => _status = "Depth isolate error: $e");
      });
      _depIso = await Isolate.spawn(
        _depthIsolateEntry,
        _IsoInit(
          replyPort: _depOutPort.sendPort,
          modelBytes: TransferableTypedData.fromList([depthBytes.buffer.asUint8List()]),
          labels: const [],
          threads: kTfliteThreads,
          useNNAPI: kEnableNNAPI,
          preferGPU: kEnableGPUForDepth,
          scoreThreshold: 0.0,
        ),
        errorsAreFatal: true,
        debugName: "depth_isolate",
      );

      // 5) Camera stream
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

        // FPS overlay
        _framesInWindow++;
        if (_fpsWindow.elapsedMilliseconds >= 500) {
          _fpsNow = (_framesInWindow * 1000) / _fpsWindow.elapsedMilliseconds;
          _framesInWindow = 0;
          _fpsWindow..reset()..start();
          if (mounted) setState(() {});
        }
      });

      if (mounted) setState(() => _status = "Real-time running ✅");
    } catch (e) {
      if (mounted) setState(() => _status = "Init error: $e");
    }
  }

  // ====== Detector messages ======
  void _onDetectorMessage(dynamic message) {
    if (message is Map && message['type'] == 'ready') {
      _detInPort = message['port'] as SendPort;
      if (mounted) setState(() {});
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
      _refreshOverlay(); // attach depth + blobs + draw + speech
    }
  }

  // ====== Depth messages ======
  void _onDepthMessage(dynamic message) {
    if (message is Map && message['type'] == 'ready') {
      _depInPort = message['port'] as SendPort;
      if (mounted) setState(() {});
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
    }
  }

  // ====== Overlay & Speech (ALWAYS compute blobs) ======
  void _refreshOverlay() {
    if (!mounted) return;

    // 1) Always-on blobs (ใช้ center pixel + สูตรเมตรใหม่)
    _lastBlobs = (_lastDepthField != null)
        ? extractNearestBlobs(_lastDepthField!)
        : const [];

    // 2) Attach depth ให้ Object Detections (ใช้ center pixel + สูตรเดิม)
    final enriched = (_lastDepthField != null)
        ? _attachDepthToDetections(_lastDetections, _lastDepthField!)
        : _lastDetections;

    // 3) แปลง blobs → _Detection เพื่อวาด พร้อมจุดกึ่งกลาง
    final blobDets = _lastBlobs.map((b) {
      final mTxt = (b.centerMeters != null) ? "≈${b.centerMeters!.toStringAsFixed(2)}m" : "—";
      return _Detection(
        rect: b.bboxInPreview,
        label: "obstacle",
        score: b.centerZ,     // ใช้ใกล้-ไกลแบบ near-score
        extra: mTxt,
        zRel: b.centerZ,
      );
    }).toList();

    final draw = [...enriched, ...blobDets];

    // 4) Status
    final pathTxt = _lastBlobs.isEmpty ? "Path: clear" : "Path: blocked (${_lastBlobs.length})";

    setState(() {
      _drawDetections = draw;
      _status = "$pathTxt  • objs:${enriched.length} • "
                "det:$_lastDetDurMs/$kDetIntervalMs ms • depth:$_lastDepDurMs/$kDepthIntervalMs ms";
    });

    // 5) Speech (ใช้ค่า blob ตามสูตรใหม่)
    if (_lastDepthField != null) {
      _maybeSpeak(enriched,
        previewSize: Size(_lastDepthField!.targetW, _lastDepthField!.targetH),
        blobs: _lastBlobs,
      );
    }
  }

  // ====== Depth → attach z_center + readable distance (DET ใช้วิธีเดิม) ======
  List<_Detection> _attachDepthToDetections(List<_Detection> dets, _DepthField f) {
    if (f.w == 0 || f.h == 0 || f.norm.isEmpty) return dets;
    final sx = f.w / f.targetW;
    final sy = f.h / f.targetH;

    final result = <_Detection>[];
    for (final d in dets) {
      // จุดกึ่งกลางกรอบ (พิกัด grid)
      final cxGrid = (d.rect.center.dx * sx).clamp(0.0, f.w - 1.0);
      final cyGrid = (d.rect.center.dy * sy).clamp(0.0, f.h - 1.0);
      final ix = cxGrid.round().clamp(0, f.w - 1);
      final iy = cyGrid.round().clamp(0, f.h - 1);
      final z = f.norm[iy * f.w + ix]; // near-score ณ center pixel

      String depthText;
      if (_metersPerUnit > 0) {
        const eps = 1e-3;
        final m = (_metersPerUnit / (z + eps)).clamp(0.0, 999.0);
        depthText = "≈${m.toStringAsFixed(2)}m";
      } else {
        final bucket = zToMetersBucket(z);
        depthText = (bucket >= 5) ? "≥5m" : "≈${bucket}m";
      }
      result.add(d.copyWith(extra: depthText, zRel: z));
    }
    return result;
  }

  // ====== Speech helpers ======
  String _dirFromRect(Rect rect, double previewW, {required bool mirrorX}) {
    final cx = rect.center.dx;
    final double x = mirrorX ? (previewW - cx) : cx;
    final third = previewW / 3.0;
    if (x < third) return "left";
    if (x > 2 * third) return "right";
    return "front";
  }

  Future<void> _say(String msg) async {
    final now = DateTime.now();
    if (now.millisecondsSinceEpoch - _lastSpokenAt.millisecondsSinceEpoch < kSpeakCooldownMs) {
      _sayQ.add(msg);
      return;
    }
    if (_isSpeaking) {
      _sayQ.add(msg);
      return;
    }
    _isSpeaking = true;
    _lastSpokenAt = now;
    _lastSpokenKey = msg;
    try {
      await _tts.speak(msg);
    } finally {
      _isSpeaking = false;
    }
    if (_sayQ.isNotEmpty) {
      final next = _sayQ.removeFirst();
      if (next != _lastSpokenKey) {
        await _say(next);
      }
    }
  }

  void _maybeSpeak(List<_Detection> dets, {required Size previewSize, required List<BlobResult> blobs}) {
    // ถ้ามี object ใกล้ (≤~3m) พูด object ก่อน (ใช้ bucket จาก z center ของ DET)
    _Detection? best;
    int bestBucket = 999;
    for (final d in dets) {
      final z = d.zRel;
      if (z == null) continue;
      final bucket = zToMetersBucket(z);
      if (bucket <= 3 && bucket < bestBucket) {
        bestBucket = bucket; best = d;
      }
    }
    if (best != null) {
      final dir = _dirFromRect(best.rect, previewSize.width, mirrorX: _isFront);
      final txt = "A ${best.label}, about ${metersBucketLabel(bestBucket)}, ${dir}.";
      _say(txt);
      return;
    }

    // พูดสถานะทางเดินจาก blobs (ใช้สูตรเมตรใหม่)
    if (blobs.isEmpty) {
      _say("The path ahead is clear.");
    } else {
      final b = blobs.first; // ใกล้สุด
      final dir = _dirFromRect(b.bboxInPreview, previewSize.width, mirrorX: _isFront);
      final m = b.centerMeters;
      if (m != null) {
        _say("Obstacle ahead ${dir}, about ${m.toStringAsFixed(1)} meters.");
      } else {
        final bucket = zToMetersBucket(b.centerZ);
        final dist = metersBucketLabel(bucket);
        _say("Obstacle ahead ${dir}, about $dist.");
      }
    }
  }

  // ====== Pack YUV frame for isolates ======
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

  // ====== Load labels ======
  Future<List<String>> _loadLabels(String assetPath) async {
    final raw = await rootBundle.loadString(assetPath);
    final lines = raw
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();
    return lines;
  }

  // ====== UI ======
  @override
  Widget build(BuildContext context) {
    final camReady = _controller?.value.isInitialized == true;
    final previewSize = _controller?.value.previewSize;

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
                // Status (top)
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
                // FPS/time (top-left)
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
                // Note depth mode (bottom)
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
                          "Depth = relative (0..1, larger = closer). Calibrate DET to meters by setting _metersPerUnit.",
                          style: TextStyle(color: Colors.white, fontSize: 12),
                        ),
                      ),
                    ),
                  ),
                // Toggle heatmap
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
              ],
            )
          : const Center(child: CircularProgressIndicator()),
    );
  }
}

// =====================================================
// ================= Data classes / painter ============
// =====================================================

class _Detection {
  final Rect rect;       // พิกเซลใน "พรีวิว"
  final String label;
  final double score;
  final String? extra;   // เช่น "≈2m"
  final double? zRel;    // near z (0..1, ยิ่งมาก=ใกล้)

  _Detection({
    required this.rect,
    required this.label,
    required this.score,
    this.extra,
    this.zRel,
  });

  _Detection copyWith({
    Rect? rect,
    String? label,
    double? score,
    String? extra,
    double? zRel,
  }) {
    return _Detection(
      rect: rect ?? this.rect,
      label: label ?? this.label,
      score: score ?? this.score,
      extra: extra ?? this.extra,
      zRel: zRel ?? this.zRel,
    );
  }
}

class _DetectionsPainter extends CustomPainter {
  final List<_Detection> dets;
  final bool mirrorX;
  final double srcW;
  final double srcH;
  _DetectionsPainter(this.dets, {this.mirrorX = false, required this.srcW, required this.srcH});

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width  / srcW;
    final scaleY = size.height / srcH;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.greenAccent;

    final centerDotPaint = Paint()
      ..style = PaintingStyle.fill
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

      // กรอบ
      canvas.drawRect(rect, boxPaint);

      // จุดกึ่งกลางกรอบ (สีเขียว)
      final center = rect.center;
      canvas.drawCircle(center, 3.0, centerDotPaint);

      // ป้าย
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
  final double targetW; // preview size (สำหรับ map กลับไปวาด)
  final double targetH;
  _DepthField({required this.norm, required this.w, required this.h, required this.targetW, required this.targetH});
}

// =====================================================
// =================== BLOB MODULE =====================
// =====================================================

class BlobResult {
  final int id;
  final Rect bboxInPreview;
  final double centerZ;        // near z ณ จุดกึ่งกลางกรอบ (pixel center)
  final double? centerMeters;  // meters จากสูตรใหม่ (10.22*z - 2.66)
  final double areaFrac;
  BlobResult({
    required this.id,
    required this.bboxInPreview,
    required this.centerZ,
    required this.centerMeters,
    required this.areaFrac,
  });
}

// ค่าจูน
const double kRoiTopFrac        = 0.35; // ใช้ช่วงล่าง 65%
const double kMinDepthContrast  = 0.03;
const double kHighPct           = 0.85;
const double kLowPct            = 0.70;
const bool   kUse4Conn          = true;
const double kMinBlobAreaFrac   = 0.003;
const double kMinAspect         = 0.25;
const double kMaxWidthFrac      = 0.95;
const int    kMaxBlobsCapBase   = 4;
const int    kMaxBlobsCapMax    = 8;
const double kDenseSceneBoostAt = 0.10;
const double kIouSuppress       = 0.35;

Uint8List _erode(Uint8List m, int w, int h, int y0, int y1) {
  final out = Uint8List.fromList(m);
  for (int y = y0; y <= y1; y++) {
    final base = y * w;
    for (int x = 0; x < w; x++) {
      if (m[base + x] == 0) { out[base + x] = 0; continue; }
      bool keep = true;
      for (int dy = -1; dy <= 1 && keep; dy++) {
        final yy = y + dy; if (yy < y0 || yy > y1) { keep = false; break; }
        for (int dx = -1; dx <= 1; dx++) {
          final xx = x + dx; if (xx < 0 || xx >= w) { keep = false; break; }
          if (m[yy * w + xx] == 0) { keep = false; break; }
        }
      }
      out[base + x] = keep ? 1 : 0;
    }
  }
  return out;
}

Uint8List _dilate(Uint8List m, int w, int h, int y0, int y1) {
  final out = Uint8List.fromList(m);
  for (int y = y0; y <= y1; y++) {
    final base = y * w;
    for (int x = 0; x < w; x++) {
      if (m[base + x] == 1) { out[base + x] = 1; continue; }
      bool any = false;
      for (int dy = -1; dy <= 1 && !any; dy++) {
        final yy = y + dy; if (yy < y0 || yy > y1) continue;
        for (int dx = -1; dx <= 1; dx++) {
          final xx = x + dx; if (xx < 0 || xx >= w) continue;
          if (m[yy * w + xx] == 1) { any = true; break; }
        }
      }
      out[base + x] = any ? 1 : 0;
    }
  }
  return out;
}

double _iouRect(Rect a, Rect b) {
  final l = math.max(a.left, b.left);
  final t = math.max(a.top,  b.top);
  final r = math.min(a.right,b.right);
  final bt= math.min(a.bottom,b.bottom);
  final inter = math.max(0.0, r-l) * math.max(0.0, bt-t);
  final ua = a.width*a.height + b.width*b.height - inter;
  return ua <= 0 ? 0.0 : inter/ua;
}

List<BlobResult> _autoSelectBlobs({
  required List<BlobResult> blobs,
  required int frameW,
  required int frameH,
  required int roiPixels,
  required int nearCount,
}) {
  if (blobs.isEmpty) return const [];
  final nearFrac = nearCount / (roiPixels.toDouble() + 1e-6);
  int cap = kMaxBlobsCapBase + (nearFrac > kDenseSceneBoostAt ? 2 : 0);
  if (nearFrac > (kDenseSceneBoostAt * 2.2)) cap += 2;
  if (cap > kMaxBlobsCapMax) cap = kMaxBlobsCapMax;

  final picked = <BlobResult>[];
  for (final cand in blobs) {
    bool overlap = false;
    for (final p in picked) {
      if (_iouRect(cand.bboxInPreview, p.bboxInPreview) > kIouSuppress) { overlap = true; break; }
    }
    if (!overlap) {
      picked.add(cand);
      if (picked.length >= cap) break;
    }
  }
  return picked;
}

/// Always-on blob extractor (ใกล้=ค่าสูง). ใช้ "พิกเซลกึ่งกลางกรอบ" เป็น depth ของ blob
List<BlobResult> extractNearestBlobs(_DepthField f) {
  final int w = f.w, h = f.h;
  if (w == 0 || h == 0 || f.norm.isEmpty) return const [];

  final int y0 = (kRoiTopFrac * h).clamp(0, h-1).toInt();
  final int y1 = h-1;

  final vals = <double>[];
  for (int y = y0; y <= y1; y++) {
    final base = y * w;
    for (int x = 0; x < w; x++) vals.add(f.norm[base + x]);
  }
  if (vals.isEmpty) return const [];
  vals.sort();
  final p10 = vals[(0.10 * (vals.length - 1)).toInt()];
  final p90 = vals[(0.90 * (vals.length - 1)).toInt()];
  if ((p90 - p10) < kMinDepthContrast) return const [];

  final highThr = vals[(kHighPct * (vals.length - 1)).toInt()];
  final lowThr  = vals[(kLowPct  * (vals.length - 1)).toInt()];

  final lowMask  = Uint8List(w * h);
  final highMask = Uint8List(w * h);
  int nearCount = 0;
  for (int y = y0; y <= y1; y++) {
    final base = y * w;
    for (int x = 0; x < w; x++) {
      final v = f.norm[base + x];
      final inLow  = v >= lowThr  ? 1 : 0;
      final inHigh = v >= highThr ? 1 : 0;
      lowMask [base + x] = inLow;
      highMask[base + x] = inHigh;
      if (inLow == 1) nearCount++;
    }
  }

  final mask = _erode(_dilate(lowMask, w,h,y0,y1), w,h,y0,y1);

  final visited = Uint8List(w*h);
  final roiPixels = (y1 - y0 + 1) * w;
  final blobs = <BlobResult>[];
  int nextId = 1;

  void tryPush(List<int> qx, List<int> qy, int x, int y) {
    if (x < 0 || x >= w || y < y0 || y > y1) return;
    final i = y * w + x;
    if (visited[i] == 0 && mask[i] == 1) { visited[i] = 1; qx.add(x); qy.add(y); }
  }

  void growFromSeed(int seedX, int seedY) {
    final qx = <int>[seedX], qy = <int>[seedY];
    visited[seedY*w + seedX] = 1;
    int minX=seedX, maxX=seedX, minY=seedY, maxY=seedY;
    int area = 0;
    final idxs = <int>[];

    while (qx.isNotEmpty) {
      final x = qx.removeLast();
      final y = qy.removeLast();
      final i = y * w + x;
      idxs.add(i);
      area++;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;

      if (kUse4Conn) {
        tryPush(qx,qy, x-1,y); tryPush(qx,qy, x+1,y); tryPush(qx,qy, x,y-1); tryPush(qx,qy, x,y+1);
      } else {
        for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++) {
          if (dx==0 && dy==0) continue; tryPush(qx,qy, x+dx, y+dy);
        }
      }
    }

    final bw = (maxX - minX + 1).toDouble();
    final bh = (maxY - minY + 1).toDouble();
    final areaFrac = area / (w.toDouble() * h);
    final aspect = bh / (bw + 1e-6);
    final widthFrac = bw / w;

    if (areaFrac < kMinBlobAreaFrac) return;
    if (aspect < kMinAspect) return;
    if (widthFrac > kMaxWidthFrac) return;

    // จุดกึ่งกลางกรอบ (pixel center) → depth center
    final cxGrid = ((minX + maxX) * 0.5).clamp(0.0, w - 1.0);
    final cyGrid = ((minY + maxY) * 0.5).clamp(0.0, h - 1.0);
    final ci = cyGrid.round().clamp(0, h-1) * w + cxGrid.round().clamp(0, w-1);
    final zCenter = f.norm[ci];

    // แปลงเมตรตามสูตรใหม่
    final meters = blobMetersFromCenterZ(zCenter);

    // สร้าง bbox ในพื้นที่ preview
    final scaleXf = f.targetW / w;
    final scaleYf = f.targetH / h;
    final rect = Rect.fromLTWH(minX*scaleXf, minY*scaleYf, bw*scaleXf, bh*scaleYf);

    blobs.add(BlobResult(
      id: nextId++,
      bboxInPreview: rect,
      centerZ: zCenter,
      centerMeters: meters,
      areaFrac: areaFrac,
    ));
  }

  for (int y = y0; y <= y1; y++) {
    final base = y * w;
    for (int x = 0; x < w; x++) {
      final i = base + x;
      if (highMask[i] == 1 && mask[i] == 1 && visited[i] == 0) growFromSeed(x,y);
    }
  }

  // sort ใกล้สุดก่อน (centerZ สูงก่อน)
  blobs.sort((a,b) => b.centerZ.compareTo(a.centerZ));
  return _autoSelectBlobs(
    blobs: blobs, frameW: w, frameH: h, roiPixels: roiPixels, nearCount: nearCount,
  );
}

// =====================================================
// ================== Isolate payload ==================
// =====================================================

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

// =====================================================
// =================== DETECTOR ISOLATE ================
// =====================================================

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

      // Filtering (person-boost)
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

        // [l,t,r,b,score, mappedCls, labelString]
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

// =====================================================
// ==================== DEPTH ISOLATE ==================
// =====================================================

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
        tmp[i] = (tmp[i] - minV) / range; // 0..1 (normalize)
      }

      // Auto-orient so that "larger = closer"
      {
        final int hh = hOut, ww = wOut;
        final int split = (hh * 0.60).toInt();
        double top = 0, bot = 0; int nt = 0, nb = 0;
        for (int y = 0; y < hh; y++) {
          final base = y * ww;
          for (int x = 0; x < ww; x++) {
            final v = tmp[base + x];
            if (y < split) { top += v; nt++; } else { bot += v; nb++; }
          }
        }
        final topMean = top / (nt + 1e-6);
        final botMean = bot / (nb + 1e-6);
        if (botMean < topMean) {
          for (int i = 0; i < tmp.length; i++) tmp[i] = 1.0 - tmp[i];
        }
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

// =====================================================
// ================== Shared helpers ===================
// =====================================================

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

        final y00 = yBytes[(sy00 * yRowStride + sx00).clamp(0, yBytes.length - 1)] & 0xFF;
        final y10 = yBytes[(sy10 * yRowStride + sx10).clamp(0, yBytes.length - 1)] & 0xFF;
        final y01 = yBytes[(sy01 * yRowStride + sx01).clamp(0, yBytes.length - 1)] & 0xFF;
        final y11 = yBytes[(sy11 * yRowStride + sx11).clamp(0, yBytes.length - 1)] & 0xFF;

        final yTop = y00 * (1 - wx) + y10 * wx;
        final yBot = y01 * (1 - wx) + y11 * wx;
        final yF = yTop * (1 - wy) + yBot * wy;

        mapSrc(fx.round(), fy.round(), temp);
        final csx = temp[0], csy = temp[1];
        final cuRow = (csy >> 1) * uvRowStride;
        final cuCol = (csx >> 1) * uvPixelStride;
        final uIdx = (cuRow + cuCol).clamp(0, uBytes.length - 1);
        final vIdx = (cuRow + cuCol).clamp(0, vBytes.length - 1);
        final U = (uBytes[uIdx] & 0xFF) - 128;
        final V = (vBytes[vIdx] & 0xFF) - 128;

        int r = (yF + ((1436 * V) >> 10)).round();
        int g = (yF - ((352 * U + 731 * V) >> 10)).round();
        int b = (yF + ((1815 * U) >> 10)).round();

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

        final y00 = yBytes[(sy00 * yRowStride + sx00).clamp(0, yBytes.length - 1)] & 0xFF;
        final y10 = yBytes[(sy10 * yRowStride + sx10).clamp(0, yBytes.length - 1)] & 0xFF;
        final y01 = yBytes[(sy01 * yRowStride + sx01).clamp(0, yBytes.length - 1)] & 0xFF;
        final y11 = yBytes[(sy11 * yRowStride + sx11).clamp(0, yBytes.length - 1)] & 0xFF;

        final yTop = y00 * (1 - wx) + y10 * wx;
        final yBot = y01 * (1 - wx) + y11 * wx;
        final yF = yTop * (1 - wy) + yBot * wy;

        mapSrc(fx.round(), fy.round(), temp);
        final csx = temp[0], csy = temp[1];
        final cuRow = (csy >> 1) * uvRowStride;
        final cuCol = (csx >> 1) * uvPixelStride;
        final uIdx = (cuRow + cuCol).clamp(0, uBytes.length - 1);
        final vIdx = (cuRow + cuCol).clamp(0, vBytes.length - 1);
        final U = (uBytes[uIdx] & 0xFF) - 128;
        final V = (vBytes[vIdx] & 0xFF) - 128;

        int r = (yF + ((1436 * V) >> 10)).round();
        int g = (yF - ((352 * U + 731 * V) >> 10)).round();
        int b = (yF + ((1815 * U) >> 10)).round();

        if (r < 0) r = 0; else if (r > 255) r = 255;
        if (g < 0) g = 0; else if (g > 255) g = 255;
        if (b < 0) b = 0; else if (b > 255) b = 255;

        out[o++] = r; out[o++] = g; out[o++] = b;
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