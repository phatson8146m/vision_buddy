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

// ====== Calibrate z (0..1, ยิ่งมาก=ยิ่งใกล้) → meters ด้วยสมการกำลังสอง ======
double zToMetersQuad(double z) {
  // m = a*z^2 + b*z + c  (ค่าจากตารางที่แนบ)
  const double a = 21.723208;
  const double b = 34.598467;
  const double c = -10.082949;

  var m = a * z * z + b * z + c;

  // ป้องกันค่าเพี้ยน และจำกัดช่วง
  if (!m.isFinite) m = 0.0;
  if (m < 0) m = 0.0;
  if (m > 10.0) m = 10.0; // สมมติใช้จริงไม่เกิน ~10 m

  return m;
}

// bucket ระยะจาก "เมตรจริง" (เอาไปใช้สร้างประโยคเสียง)
int zToMetersBucket(double z) {
  final m = zToMetersQuad(z);
  if (m <= 1.5) return 1; // ~0-1.5 m
  if (m <= 2.5) return 2; // ~1.5-2.5 m
  if (m <= 3.5) return 3; // ~2.5-3.5 m
  if (m <= 4.5) return 4; // ~3.5-4.5 m
  return 5; // ≥4.5 m ขึ้นไป
}

String metersBucketLabel(int m) =>
    (m >= 5) ? "5 meters or more" : "$m meter${m == 1 ? "" : "s"}";

const bool kEnableNNAPI = false;
const bool kEnableGPUForDepth = false;

const int kDetIntervalMs = 66; // ~15 FPS
const int kDepthIntervalMs = 90; // ~11 FPS

const double kScoreThreshold = 0.40;
const int kTfliteThreads = 4;

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
      title: 'AI Depth/Object App (Centers + Blob alert)',
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

  // Blobs always-on (no meters)
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
        await _tts.setLanguage("th-TH");
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
      final detBytes = await rootBundle.load('assets/models/1.tflite');
      final depthBytes = await rootBundle.load(
        'assets/models/model_opt.tflite',
      );

      // 3) Detector isolate
      _detOutPort = ReceivePort();
      _detOutPort.listen(
        _onDetectorMessage,
        onError: (e) {
          if (mounted) setState(() => _status = "Detector isolate error: $e");
        },
      );
      _detIso = await Isolate.spawn(
        _detectorIsolateEntry,
        _IsoInit(
          replyPort: _detOutPort.sendPort,
          modelBytes: TransferableTypedData.fromList([
            detBytes.buffer.asUint8List(),
          ]),
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
      _depOutPort.listen(
        _onDepthMessage,
        onError: (e) {
          if (mounted) setState(() => _status = "Depth isolate error: $e");
        },
      );
      _depIso = await Isolate.spawn(
        _depthIsolateEntry,
        _IsoInit(
          replyPort: _depOutPort.sendPort,
          modelBytes: TransferableTypedData.fromList([
            depthBytes.buffer.asUint8List(),
          ]),
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

        if (!_detBusy &&
            (now - _lastDetSendMs) >= kDetIntervalMs &&
            _detInPort != null) {
          _detInPort!.send(_packYuvFrame(frame));
          _detBusy = true;
          _lastDetSendMs = now;
        }

        if (!_depBusy &&
            (now - _lastDepSendMs) >= kDepthIntervalMs &&
            _depInPort != null) {
          _depInPort!.send(_packYuvFrame(frame, makeHeatmap: _showHeatmap));
          _depBusy = true;
          _lastDepSendMs = now;
        }

        // FPS overlay
        _framesInWindow++;
        if (_fpsWindow.elapsedMilliseconds >= 500) {
          _fpsNow = (_framesInWindow * 1000) / _fpsWindow.elapsedMilliseconds;
          _framesInWindow = 0;
          _fpsWindow
            ..reset()
            ..start();
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
        final nameFromIso =
            (e.length >= 7 && e[6] is String) ? (e[6] as String) : null;

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

        parsed.add(
          _Detection(rect: Rect.fromLTRB(l, t, r, b), label: label, score: s),
        );
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

      final TransferableTypedData ttd =
          message['norm'] as TransferableTypedData;
      final Float32List grid = ttd.materialize().asFloat32List();

      _lastDepthField = _DepthField(
        norm: grid,
        w: w,
        h: h,
        targetW: targetW,
        targetH: targetH,
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

    // 1) Always-on blobs (ใช้ center pixel ของ blob เป็น zRel — ไม่แปลงเมตร)
    _lastBlobs =
        (_lastDepthField != null)
            ? extractNearestBlobs(_lastDepthField!)
            : const [];

    // 2) Attach depth ให้ Object Detections (DET ใช้สมการกำลังสอง z→เมตร)
    final enriched =
        (_lastDepthField != null)
            ? _attachDepthToDetections(_lastDetections, _lastDepthField!)
            : _lastDetections;

    // 3) แปลง blobs → _Detection เพื่อวาด พร้อมจุดกึ่งกลาง (แสดง z≈..)
    final blobDets =
        _lastBlobs.map((b) {
          // ลองยืมชื่อคลาสจาก DET ถ้าแมตช์ดีพอ + z ใกล้กัน
          final borrowed = _bestLabelForRect(
            b.bboxInPreview,
            enriched, // enriched dets มี zRel แล้ว
            zBlob: b.centerZ,
          );

          final showLabel =
              borrowed != null ? "${borrowed} (blob)" : "obstacle";
          final mTxt = "z≈${b.centerZ.toStringAsFixed(2)}"; // แสดง z อย่างเดียว
          return _Detection(
            rect: b.bboxInPreview,
            label: showLabel,
            score: b.centerZ, // ใช้ near-score
            extra: mTxt, // ไม่ใช่เมตร
            zRel: b.centerZ,
          );
        }).toList();

    final draw = [...enriched, ...blobDets];

    // 4) Status
    final pathTxt =
        _lastBlobs.isEmpty
            ? "Path: clear"
            : "Path: blocked (${_lastBlobs.length})";

    setState(() {
      _drawDetections = draw;
      _status =
          "$pathTxt  • objs:${enriched.length} • "
          "det:$_lastDetDurMs/$kDetIntervalMs ms • depth:$_lastDepDurMs/$kDepthIntervalMs ms";
    });

    // 5) Speech — BLOB จะพูดแค่ทิศ ไม่บอกเมตร / DET พูดแบบ bucket เป็นเมตร
    if (_lastDepthField != null) {
      _maybeSpeak(
        enriched,
        previewSize: Size(_lastDepthField!.targetW, _lastDepthField!.targetH),
        blobs: _lastBlobs,
      );
    }
  }

  // ====== Depth → attach z_center + readable distance (ใช้สมการกำลังสอง) ======
  List<_Detection> _attachDepthToDetections(
    List<_Detection> dets,
    _DepthField f,
  ) {
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

      final m = zToMetersQuad(z);
      final depthText = "≈${m.toStringAsFixed(2)}m";

      result.add(d.copyWith(extra: depthText, zRel: z));
    }
    return result;
  }

  // ====== Speech queue (แก้ไม่ให้พูดทิศเก่าซ้อน) ======
  Future<void> _say(String msg) async {
    final now = DateTime.now();
    final nowMs = now.millisecondsSinceEpoch;
    final lastMs = _lastSpokenAt.millisecondsSinceEpoch;

    // ยังอยู่ใน cooldown หรือกำลังพูดอยู่ → เก็บเฉพาะข้อความล่าสุด
    if (nowMs - lastMs < kSpeakCooldownMs || _isSpeaking) {
      _sayQ
        ..clear()
        ..add(msg);
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

    // ถ้ามีคิวค้างอยู่ ให้พูดแค่ "อันล่าสุด" เท่านั้น
    if (_sayQ.isNotEmpty) {
      final next = _sayQ.removeLast();
      if (next != _lastSpokenKey) {
        await _say(next);
      }
    }
  }

  // ====== Overlay & Speech (ALWAYS compute & decide via Decision Tree) ======
  final ObstacleDecider _decider = ObstacleDecider();

  void _maybeSpeak(
    List<_Detection> dets, {
    required Size previewSize,
    required List<BlobResult> blobs,
  }) {
    final decision = _decider.decide(
      dets: dets,
      blobs: blobs,
      previewSize: previewSize,
      mirrorX: _isFront,
    );

    if (!decision.shouldAlert || decision.message == null) return;
    _say(decision.message!);
  }

  // ====== Pack YUV frame for isolates ======
  Map _packYuvFrame(CameraImage cam, {bool makeHeatmap = false}) {
    final y = TransferableTypedData.fromList([cam.planes[0].bytes]);
    final u = TransferableTypedData.fromList([cam.planes[1].bytes]);
    final v = TransferableTypedData.fromList([cam.planes[2].bytes]);
    final size = _controller?.value.previewSize;

    final previewPortrait =
        (size?.height ?? cam.height.toDouble()) >
        (size?.width ?? cam.width.toDouble());
    final bufferPortrait = cam.height > cam.width;
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
    final lines =
        raw
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
      body:
          camReady && previewSize != null
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
                            IgnorePointer(
                              child: Image.memory(
                                _depthOverlayPng!,
                                fit: BoxFit.cover,
                              ),
                            ),
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
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.55),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          _status,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
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
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.55),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Text(
                          "FPS: ${_fpsNow.toStringAsFixed(1)} • det:$_lastDetDurMs ms/$kDetIntervalMs • depth:$_lastDepDurMs ms/$kDepthIntervalMs",
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                        ),
                      ),
                    ),
                  ),
                  // Note depth mode (bottom) – อธิบายว่าคาลิเบรตด้วยสมการกำลังสอง
                  SafeArea(
                    child: Align(
                      alignment: Alignment.bottomCenter,
                      child: Container(
                        margin: const EdgeInsets.only(bottom: 10),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.45),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: const Text(
                          "Depth (DET) แปลงเป็นเมตรด้วย m = 21.723208·z² + 34.598467·z − 10.082949",
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
  final Rect rect; // พิกเซลใน "พรีวิว"
  final String label;
  final double score;
  final String? extra; // เช่น "≈2m" หรือ "z≈0.82"
  final double? zRel; // near z (0..1, ยิ่งมาก=ใกล้)

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
  _DetectionsPainter(
    this.dets, {
    this.mirrorX = false,
    required this.srcW,
    required this.srcH,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / srcW;
    final scaleY = size.height / srcH;

    // ====== RED GUIDES (Left / Center / Right) ======
    final red =
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = Colors.redAccent;

    final x1 = size.width / 3;
    final x2 = 2 * size.width / 3;

    // vertical guide lines
    canvas.drawLine(Offset(x1, 0), Offset(x1, size.height), red);
    canvas.drawLine(Offset(x2, 0), Offset(x2, size.height), red);

    // top labels with percentages
    final lPct = (x1 / size.width * 100).round(); // ~33
    final cPct = ((x2 - x1) / size.width * 100).round(); // ~34
    final rPct = (((size.width - x2) / size.width) * 100).round(); // ~33

    void drawTag(String text, double cx) {
      final tp = _tp(
        text,
        color: Colors.redAccent,
        size: 12,
        weight: FontWeight.w700,
      );
      const pad = 3.0;
      final bgRect = Rect.fromLTWH(
        (cx - tp.width / 2) - pad,
        6,
        tp.width + 2 * pad,
        tp.height + 2 * pad,
      );
      canvas.drawRect(bgRect, Paint()..color = Colors.black.withOpacity(0.55));
      tp.paint(canvas, Offset(bgRect.left + pad, bgRect.top + pad));
    }

    drawTag("Left ~${lPct}%", x1 / 2);
    drawTag("Center ~${cPct}%", (x1 + x2) / 2);
    drawTag("Right ~${rPct}%", (x2 + size.width) / 2);

    // ====== DETECTION BOXES & LABELS ======
    final boxPaint =
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3
          ..color = Colors.greenAccent;

    final centerDotPaint =
        Paint()
          ..style = PaintingStyle.fill
          ..color = Colors.greenAccent;

    final bg = Paint()..color = Colors.black.withOpacity(0.55);

    for (final d in dets) {
      Rect rect = Rect.fromLTRB(
        d.rect.left * scaleX,
        d.rect.top * scaleY,
        d.rect.right * scaleX,
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
      final text =
          "${d.label} ${d.score.toStringAsFixed(2)}"
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

  TextPainter _tp(
    String s, {
    Color color = Colors.white,
    double size = 12,
    FontWeight weight = FontWeight.w500,
  }) => TextPainter(
    text: TextSpan(
      style: TextStyle(color: color, fontSize: size, fontWeight: weight),
      text: s,
    ),
    textDirection: TextDirection.ltr,
    maxLines: 1,
  )..layout();

  @override
  bool shouldRepaint(covariant _DetectionsPainter oldDelegate) =>
      oldDelegate.dets != dets ||
      oldDelegate.mirrorX != mirrorX ||
      oldDelegate.srcW != srcW ||
      oldDelegate.srcH != srcH;
}

class _DepthField {
  final Float32List norm; // length = w*h, 0..1 (มาก=ใกล้)
  final int w;
  final int h;
  final double targetW; // preview size (สำหรับ map กลับไปวาด)
  final double targetH;
  _DepthField({
    required this.norm,
    required this.w,
    required this.h,
    required this.targetW,
    required this.targetH,
  });
}

// =====================================================
// =================== BLOB MODULE =====================
// =====================================================

class BlobResult {
  final int id;
  final Rect bboxInPreview;
  final double centerZ; // near z ณ จุดกึ่งกลางกรอบ (0..1, มาก=ใกล้)
  final double areaFrac;
  BlobResult({
    required this.id,
    required this.bboxInPreview,
    required this.centerZ,
    required this.areaFrac,
  });
}

// ค่าจูน
const double kRoiTopFrac = 0.35; // ใช้ช่วงล่าง 65%
const double kMinDepthContrast = 0.03;
const double kHighPct = 0.85;
const double kLowPct = 0.70;
const bool kUse4Conn = true;
const double kMinBlobAreaFrac = 0.003;
const double kMinAspect = 0.25;
const double kMaxWidthFrac = 0.95;
const int kMaxBlobsCapBase = 4;
const int kMaxBlobsCapMax = 8;
const double kDenseSceneBoostAt = 0.10;
const double kIouSuppress = 0.35;

Uint8List _erode(Uint8List m, int w, int h, int y0, int y1) {
  final out = Uint8List.fromList(m);
  for (int y = y0; y <= y1; y++) {
    final base = y * w;
    for (int x = 0; x < w; x++) {
      if (m[base + x] == 0) {
        out[base + x] = 0;
        continue;
      }
      bool keep = true;
      for (int dy = -1; dy <= 1 && keep; dy++) {
        final yy = y + dy;
        if (yy < y0 || yy > y1) {
          keep = false;
          break;
        }
        for (int dx = -1; dx <= 1; dx++) {
          final xx = x + dx;
          if (xx < 0 || xx >= w) {
            keep = false;
            break;
          }
          if (m[yy * w + xx] == 0) {
            keep = false;
            break;
          }
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
      if (m[base + x] == 1) {
        out[base + x] = 1;
        continue;
      }
      bool any = false;
      for (int dy = -1; dy <= 1 && !any; dy++) {
        final yy = y + dy;
        if (yy < y0 || yy > y1) continue;
        for (int dx = -1; dx <= 1; dx++) {
          final xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          if (m[yy * w + xx] == 1) {
            any = true;
            break;
          }
        }
      }
      out[base + x] = any ? 1 : 0;
    }
  }
  return out;
}

double _iouRect(Rect a, Rect b) {
  final l = math.max(a.left, b.left);
  final t = math.max(a.top, b.top);
  final r = math.min(a.right, b.right);
  final bt = math.min(a.bottom, b.bottom);
  final inter = math.max(0.0, r - l) * math.max(0.0, bt - t);
  final ua = a.width * a.height + b.width * b.height - inter;
  return ua <= 0 ? 0.0 : inter / ua;
}

// ================= Decision-Tree Obstacle Decider =================
const double kDecisionScoreTh = 0.40; // ขั้นต่ำคะแนน det
const double kDecisionMinAreaFrac = 0.020; // กรอบ >= 2% ของภาพ
const double kDecisionZCloseTh = 0.50; // z (0..1) ยิ่งมาก=ยิ่งใกล้
const double kDecisionOverlapTh = 0.20; // ทับซ้อน blob ขั้นต่ำ
const int kDecisionNPersist = 2; // ต้องคงอยู่ >= N เฟรม
const Set<String> kRiskLabels = {
  'person',
  'bicycle',
  'car',
  'motorbike',
  'bus',
  'truck',
};

// --------- ยืมชื่อคลาสให้ blob ----------
const double kBorrowMinDetScore = 0.50; // det ต้องมั่นใจอย่างน้อยเท่านี้
const double kBorrowMinIoU = 0.25; // เกณฑ์ IoU ขั้นต่ำ
const double kBorrowMinIoM = 0.45; // หรือ IoM ขั้นต่ำ (inter/minArea)
const double kBorrowMaxZDiff = 0.20; // |z_blob - z_det| ≤ 0.20 ถือว่าใกล้กัน

double _rectArea(Rect r) => math.max(0.0, r.width) * math.max(0.0, r.height);

double _interArea(Rect a, Rect b) {
  final l = math.max(a.left, b.left);
  final t = math.max(a.top, b.top);
  final r = math.min(a.right, b.right);
  final bt = math.min(a.bottom, b.bottom);
  return math.max(0.0, r - l) * math.max(0.0, bt - t);
}

double _IoM(Rect a, Rect b) {
  final inter = _interArea(a, b);
  final m = math.min(_rectArea(a), _rectArea(b));
  return m <= 0 ? 0.0 : inter / m;
}

double _riskW(String label) {
  switch (label.toLowerCase()) {
    case 'person':
      return 1.2;
    case 'bicycle':
    case 'motorbike':
      return 1.1;
    case 'car':
    case 'bus':
    case 'truck':
      return 1.0;
    default:
      return 0.9;
  }
}

/// เลือกชื่อคลาสที่ดีที่สุดให้ blobRect โดยพิจารณา dets ที่ทับซ้อน + z ใกล้กัน
String? _bestLabelForRect(
  Rect blobRect,
  List<_Detection> dets, {
  double minIoU = kBorrowMinIoU,
  double minIoM = kBorrowMinIoM,
  double minDetScore = kBorrowMinDetScore,
  double maxZDiff = kBorrowMaxZDiff,
  double? zBlob,
}) {
  double bestScore = 0.0;
  String? bestLabel;

  for (final d in dets) {
    if (d.score < minDetScore) continue;

    final iou = _iouRect(d.rect, blobRect);
    final iom = _IoM(d.rect, blobRect);
    if (iou < minIoU && iom < minIoM) continue;

    if (zBlob != null && d.zRel != null) {
      if ((zBlob - d.zRel!).abs() > maxZDiff) continue;
    }

    final ov = math.max(iou, iom);
    final w = d.score * ov * _riskW(d.label);
    if (w > bestScore) {
      bestScore = w;
      bestLabel = d.label;
    }
  }
  return bestLabel;
}

class _AlertDecision {
  final bool shouldAlert;
  final String? message;
  final Rect? targetBox;
  final double? z;
  final bool? fromBlob; // ใช้บอกว่าเลือกเป้าจาก blob หรือ det
  _AlertDecision({
    required this.shouldAlert,
    this.message,
    this.targetBox,
    this.z,
    this.fromBlob,
  });
}

class ObstacleDecider {
  final List<Rect> _lastBoxes = [];
  final int _keepLastFrames = 3;

  // ทิศทางจากตำแหน่งกรอบ
  String _dirFromRect(Rect rect, double previewW, {required bool mirrorX}) {
    final cx = rect.center.dx;
    final double x = mirrorX ? (previewW - cx) : cx;
    final third = previewW / 3.0;
    if (x < third) return "ซ้าย";
    if (x > 2 * third) return "ขวา";
    return "ตรงกลาง";
  }

  _AlertDecision decide({
    required List<_Detection> dets,
    required List<BlobResult> blobs,
    required Size previewSize,
    required bool mirrorX,
  }) {
    final double frameArea = previewSize.width * previewSize.height;

    // 1) Candidate from DET (ผ่านเกณฑ์เบื้องต้น + อยู่ใน ROI ล่าง)
    final y0Roi = kRoiTopFrac * previewSize.height;
    final candidates =
        <({Rect rect, String? label, double z, bool fromBlob})>[];

    for (final d in dets) {
      final z = d.zRel;
      if (z == null) continue;
      if (d.score < kDecisionScoreTh) continue;
      final areaFrac = (d.rect.width * d.rect.height) / (frameArea + 1e-6);
      if (areaFrac < kDecisionMinAreaFrac) continue;
      if (!kRiskLabels.contains(d.label)) continue;
      final cy = (d.rect.top + d.rect.bottom) * 0.5;
      if (cy < y0Roi) continue;
      candidates.add((rect: d.rect, label: d.label, z: z, fromBlob: false));
    }

    // 1.2 เพิ่ม BLOB เป็น candidate — ยืมชื่อคลาสจาก det ถ้าผ่าน overlap+z
    for (final b in blobs) {
      final areaFrac =
          (b.bboxInPreview.width * b.bboxInPreview.height) / (frameArea + 1e-6);
      final cy = (b.bboxInPreview.top + b.bboxInPreview.bottom) * 0.5;
      if (areaFrac >= kDecisionMinAreaFrac && cy >= y0Roi) {
        final borrowed = _bestLabelForRect(
          b.bboxInPreview,
          dets,
          zBlob: b.centerZ,
        );
        candidates.add((
          rect: b.bboxInPreview,
          label: borrowed,
          z: b.centerZ,
          fromBlob: true,
        ));
      }
    }

    if (candidates.isEmpty) {
      _rollFrames(null);
      return _AlertDecision(shouldAlert: false);
    }

    // 2) กรองความใกล้ด้วย z
    final near = <({Rect rect, String? label, double z, bool fromBlob})>[];
    for (final c in candidates) {
      if (c.z >= kDecisionZCloseTh) near.add(c);
    }
    if (near.isEmpty) {
      _rollFrames(null);
      return _AlertDecision(shouldAlert: false);
    }

    // 3) ต้องมีความสัมพันธ์กับ blob (overlap) หากมาจาก det (เว้นกรณี person)
    final confirmed = <({Rect rect, String? label, double z, bool fromBlob})>[];
    for (final n in near) {
      if (n.fromBlob) {
        confirmed.add((rect: n.rect, label: n.label, z: n.z, fromBlob: true));
        continue;
      }
      double bestOv = 0.0;
      for (final b in blobs) {
        final ov = _iouRect(n.rect, b.bboxInPreview);
        if (ov > bestOv) bestOv = ov;
      }
      final veryRisky = (n.label == 'person');
      if (bestOv >= kDecisionOverlapTh || veryRisky) {
        confirmed.add((rect: n.rect, label: n.label, z: n.z, fromBlob: false));
      }
    }
    if (confirmed.isEmpty) {
      _rollFrames(null);
      return _AlertDecision(shouldAlert: false);
    }

    // 4) เลือกเป้าหมายที่ "ใกล้สุด" (z มากสุด) และตรวจ persistence
    confirmed.sort((a, b) => b.z.compareTo(a.z));
    final target = confirmed.first;

    final persisted = _isPersisted(target.rect);
    _rollFrames(target.rect);
    if (!persisted) {
      return _AlertDecision(shouldAlert: false);
    }

    // 5) สร้างข้อความ
    final dir = _dirFromRect(target.rect, previewSize.width, mirrorX: mirrorX);
    String msg;
    if (target.fromBlob) {
      // === กรณี BLOB: พูดแค่ว่าขวางฝั่งไหน (ไม่พูดเมตร/ระดับ/ชื่อวัตถุ) ===
      msg = "มีสิ่งกีดขวางด้านหน้า ${dir}";
    } else {
      // === กรณี DET: ใช้ bucket จากเมตรจริง ===
      final label = (target.label ?? 'สิ่งกีดขวาง');
      final bucket = zToMetersBucket(target.z);
      switch (bucket) {
        case 1:
          msg = "ระวัง ${label} ใกล้มาก หนึ่งเมตรด้านหน้า ${dir}";
          break;
        case 2:
          msg = "มี ${label} ประมาณสองเมตรด้านหน้า ${dir}";
          break;
        case 3:
          msg = "มี ${label} ประมาณสามเมตรด้านหน้า ${dir}";
          break;
        default:
          msg = "มี ${label} ด้านหน้า ${dir}";
      }
    }

    return _AlertDecision(
      shouldAlert: true,
      message: msg,
      targetBox: target.rect,
      z: target.z,
      fromBlob: target.fromBlob,
    );
  }

  void _rollFrames(Rect? current) {
    _lastBoxes.add(current ?? Rect.zero);
    if (_lastBoxes.length > _keepLastFrames) {
      _lastBoxes.removeAt(0);
    }
  }

  bool _isPersisted(Rect candidate) {
    if (_lastBoxes.isEmpty) return false;
    int hits = 0;
    for (final r in _lastBoxes) {
      if (r == Rect.zero) continue;
      final iouVal = _iouRect(candidate, r);
      if (iouVal >= 0.30) hits++;
    }
    return hits >= kDecisionNPersist;
  }
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
      if (_iouRect(cand.bboxInPreview, p.bboxInPreview) > kIouSuppress) {
        overlap = true;
        break;
      }
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

  final int y0 = (kRoiTopFrac * h).clamp(0, h - 1).toInt();
  final int y1 = h - 1;

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
  final lowThr = vals[(kLowPct * (vals.length - 1)).toInt()];

  final lowMask = Uint8List(w * h);
  final highMask = Uint8List(w * h);
  int nearCount = 0;
  for (int y = y0; y <= y1; y++) {
    final base = y * w;
    for (int x = 0; x < w; x++) {
      final v = f.norm[base + x];
      final inLow = v >= lowThr ? 1 : 0;
      final inHigh = v >= highThr ? 1 : 0;
      lowMask[base + x] = inLow;
      highMask[base + x] = inHigh;
      if (inLow == 1) nearCount++;
    }
  }

  final mask = _erode(_dilate(lowMask, w, h, y0, y1), w, h, y0, y1);

  final visited = Uint8List(w * h);
  final roiPixels = (y1 - y0 + 1) * w;
  final blobs = <BlobResult>[];
  int nextId = 1;

  void tryPush(List<int> qx, List<int> qy, int x, int y) {
    if (x < 0 || x >= w || y < y0 || y > y1) return;
    final idx = y * w + x;
    if (visited[idx] == 0 && mask[idx] == 1) {
      visited[idx] = 1;
      qx.add(x);
      qy.add(y);
    }
  }

  void growFromSeed(int seedX, int seedY) {
    final qx = <int>[seedX], qy = <int>[seedY];
    visited[seedY * w + seedX] = 1;
    int minX = seedX, maxX = seedX, minY = seedY, maxY = seedY;
    int area = 0;

    while (qx.isNotEmpty) {
      final x = qx.removeLast();
      final y = qy.removeLast();
      area++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      if (kUse4Conn) {
        tryPush(qx, qy, x - 1, y);
        tryPush(qx, qy, x + 1, y);
        tryPush(qx, qy, x, y - 1);
        tryPush(qx, qy, x, y + 1);
      } else {
        for (int dy = -1; dy <= 1; dy++)
          for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            tryPush(qx, qy, x + dx, y + dy);
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

    // จุดกึ่งกลางกรอบ → depth center (0..1)
    final cxGrid = ((minX + maxX) * 0.5).clamp(0.0, w - 1.0);
    final cyGrid = ((minY + maxY) * 0.5).clamp(0.0, h - 1.0);
    final ci =
        cyGrid.round().clamp(0, h - 1) * w + cxGrid.round().clamp(0, w - 1);
    final zCenter = f.norm[ci];

    // สร้าง bbox ในพื้นที่ preview
    final scaleXf = f.targetW / w;
    final scaleYf = f.targetH / h;
    final rect = Rect.fromLTWH(
      minX * scaleXf,
      minY * scaleYf,
      bw * scaleXf,
      bh * scaleYf,
    );

    blobs.add(
      BlobResult(
        id: nextId++,
        bboxInPreview: rect,
        centerZ: zCenter,
        areaFrac: areaFrac,
      ),
    );
  }

  for (int y = y0; y <= y1; y++) {
    final base = y * w;
    for (int x = 0; x < w; x++) {
      final idx = base + x;
      if (highMask[idx] == 1 && mask[idx] == 1 && visited[idx] == 0)
        growFromSeed(x, y);
    }
  }

  // sort ใกล้สุดก่อน (centerZ สูงก่อน)
  blobs.sort((a, b) => b.centerZ.compareTo(a.centerZ));
  return _autoSelectBlobs(
    blobs: blobs,
    frameW: w,
    frameH: h,
    roiPixels: roiPixels,
    nearCount: nearCount,
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
  final opts =
      tfl.InterpreterOptions()
        ..threads = init.threads
        ..useNnApiForAndroid = init.useNNAPI;
  if (init.preferGPU) {
    try {
      final gpu = tfl.GpuDelegateV2(
        options: tfl.GpuDelegateOptionsV2(isPrecisionLossAllowed: true),
      );
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
    out0 = _zerosByShape(
      interpreter.getOutputTensor(0).shape,
      asFloat: true,
    ); // boxes [1,N,4]
    out1 = _zerosByShape(
      interpreter.getOutputTensor(1).shape,
      asFloat: true,
    ); // classes [1,N]
    out2 = _zerosByShape(
      interpreter.getOutputTensor(2).shape,
      asFloat: true,
    ); // scores [1,N]
    if (interpreter.getOutputTensors().length >= 4) {
      out3 = _zerosByShape(
        interpreter.getOutputTensor(3).shape,
        asFloat: true,
      ); // count [1] or [1,1]
    }
  }

  final lowerLabels = init.labels.map((s) => s.toLowerCase()).toList();
  final int personIdx = lowerLabels.indexOf('person');

  int mapClassId(int rawCls) {
    if (init.labels.isEmpty) return rawCls;
    final first = init.labels.first.toLowerCase();
    final hasBg =
        first == 'background' || first == '???' || first.contains('background');
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

      final input4D =
          isFloatInput
              ? _build4DFloat(flat as Float32List, detInH, detInW)
              : _build4DUint8(flat as Uint8List, detInH, detInW);

      ensureOutBufs();
      final outMap = <int, Object>{
        0: out0!,
        1: out1!,
        2: out2!,
        if (out3 != null) 3: out3!,
      };
      interpreter.runForMultipleInputs([input4D], outMap);

      final boxes = out0!;
      final classes = out1!;
      final scores = out2!;
      if (boxes.isEmpty || classes.isEmpty || scores.isEmpty) {
        reply.send({
          'type': 'det',
          'durMs': sw.elapsedMilliseconds,
          'detections': const [],
        });
        return;
      }

      int nCand = (scores[0] as List).length;
      if (out3 != null && out3!.isNotEmpty) {
        final c =
            (out3![0] is List && out3![0].isNotEmpty)
                ? ((out3![0][0] as num?)?.toInt() ?? nCand)
                : ((out3![0] as num?)?.toInt() ?? nCand);
        if (c > 0 && c <= nCand) nCand = c;
      }

      // Filtering (person-boost)
      const int topKBase = 220;
      const double iouBase = 0.55;
      const double iouPerson = 0.70;
      final double scoreBase =
          (init.scoreThreshold < 0.10
              ? 0.10
              : (init.scoreThreshold > 0.60 ? 0.60 : init.scoreThreshold));
      const double minFracBase = 0.0010;
      const double minFracPerson = 0.00035;
      const double maxFracBase = 0.95;

      final totalArea = previewW * previewH;
      final cand = <List>[];

      for (int i = 0; i < nCand; i++) {
        final s = (scores[0][i] as num?)?.toDouble() ?? 0.0;

        final rawCls = (classes[0][i] as num?)?.toInt() ?? -1;
        final mappedCls = mapClassId(rawCls);
        final label =
            (mappedCls >= 0 && mappedCls < init.labels.length)
                ? init.labels[mappedCls]
                : 'id:$rawCls';

        final low = label.toLowerCase();
        final isBg = (label == '???') || low.contains('background');

        final bool isPerson =
            (!isBg && personIdx >= 0 && mappedCls == personIdx);
        final double sTh = isPerson ? (scoreBase * 0.55) : scoreBase;
        if (s < sTh) continue;

        final loc = (boxes[0][i] as List?);
        if (loc == null || loc.length < 4) continue;
        final ymin = (loc[0] as num?)?.toDouble().clamp(0.0, 1.0) ?? 0.0;
        final xmin = (loc[1] as num?)?.toDouble().clamp(0.0, 1.0) ?? 0.0;
        final ymax = (loc[2] as num?)?.toDouble().clamp(0.0, 1.0) ?? 0.0;
        final xmax = (loc[3] as num?)?.toDouble().clamp(0.0, 1.0) ?? 0.0;
        if (xmax <= xmin || ymax <= ymin) continue;

        final left = xmin * previewW;
        final top = ymin * previewH;
        final right = xmax * previewW;
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

      if (cand.length > 1)
        cand.sort((a, b) => (b[4] as double).compareTo(a[4] as double));
      final preNms =
          (cand.length > topKBase) ? cand.sublist(0, topKBase) : cand;

      final picked = _nmsPerClassWithCustomIoU(
        preNms,
        defaultIou: iouBase,
        specialClass: personIdx >= 0 ? personIdx : null,
        specialIou: iouPerson,
      );

      reply.send({
        'type': 'det',
        'durMs': sw.elapsedMilliseconds,
        'detections': picked,
      });
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
  final opts =
      tfl.InterpreterOptions()
        ..threads = init.threads
        ..useNnApiForAndroid = init.useNNAPI;
  if (init.preferGPU) {
    try {
      final gpu = tfl.GpuDelegateV2(
        options: tfl.GpuDelegateOptionsV2(isPrecisionLossAllowed: true),
      );
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
    depthOut = _zerosByShape(
      interpreter.getOutputTensor(0).shape,
      asFloat: true,
    ); // [1,h,w,1] or [1,h,w]
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

      final flat =
          _yuvToRgbFlatBilinearLuma(
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
              )
              as Float32List;

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
          final v =
              is4D
                  ? (row[x][0] as num?)?.toDouble() ?? 0.0
                  : (row[x] as num?)?.toDouble() ?? 0.0;
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
        double top = 0, bot = 0;
        int nt = 0, nb = 0;
        for (int y = 0; y < hh; y++) {
          final base = y * ww;
          for (int x = 0; x < ww; x++) {
            final v = tmp[base + x];
            if (y < split) {
              top += v;
              nt++;
            } else {
              bot += v;
              nb++;
            }
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
    final thisIou =
        (specialClass != null && specialIou != null && ci == specialClass)
            ? specialIou
            : defaultIou;
    for (final d in list) {
      bool keep = true;
      for (final p in kept) {
        if (_iouList(d, p) > thisIou) {
          keep = false;
          break;
        }
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

        mapSrc(x0, y0, temp);
        final sx00 = temp[0], sy00 = temp[1];
        mapSrc(x1, y0, temp);
        final sx10 = temp[0], sy10 = temp[1];
        mapSrc(x0, y1, temp);
        final sx01 = temp[0], sy01 = temp[1];
        mapSrc(x1, y1, temp);
        final sx11 = temp[0], sy11 = temp[1];

        final y00 =
            yBytes[(sy00 * yRowStride + sx00).clamp(0, yBytes.length - 1)] &
            0xFF;
        final y10 =
            yBytes[(sy10 * yRowStride + sx10).clamp(0, yBytes.length - 1)] &
            0xFF;
        final y01 =
            yBytes[(sy01 * yRowStride + sx01).clamp(0, yBytes.length - 1)] &
            0xFF;
        final y11 =
            yBytes[(sy11 * yRowStride + sx11).clamp(0, yBytes.length - 1)] &
            0xFF;

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

        if (r < 0)
          r = 0;
        else if (r > 255)
          r = 255;
        if (g < 0)
          g = 0;
        else if (g > 255)
          g = 255;
        if (b < 0)
          b = 0;
        else if (b > 255)
          b = 255;

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

        mapSrc(x0, y0, temp);
        final sx00 = temp[0], sy00 = temp[1];
        mapSrc(x1, y0, temp);
        final sx10 = temp[0], sy10 = temp[1];
        mapSrc(x0, y1, temp);
        final sx01 = temp[0], sy01 = temp[1];
        mapSrc(x1, y1, temp);
        final sx11 = temp[0], sy11 = temp[1];

        final y00 =
            yBytes[(sy00 * yRowStride + sx00).clamp(0, yBytes.length - 1)] &
            0xFF;
        final y10 =
            yBytes[(sy10 * yRowStride + sx10).clamp(0, yBytes.length - 1)] &
            0xFF;
        final y01 =
            yBytes[(sy01 * yRowStride + sx01).clamp(0, yBytes.length - 1)] &
            0xFF;
        final y11 =
            yBytes[(sy11 * yRowStride + sx11).clamp(0, yBytes.length - 1)] &
            0xFF;

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

        if (r < 0)
          r = 0;
        else if (r > 255)
          r = 255;
        if (g < 0)
          g = 0;
        else if (g > 255)
          g = 255;
        if (b < 0)
          b = 0;
        else if (b > 255)
          b = 255;

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
    List.generate(
      H,
      (_) => List.generate(W, (_) {
        final r = flat[idx++];
        final g = flat[idx++];
        final b = flat[idx++];
        return [r, g, b];
      }),
    ),
  ];
}

List _build4DUint8(Uint8List flat, int H, int W) {
  int idx = 0;
  return [
    List.generate(
      H,
      (_) => List.generate(W, (_) {
        final r = flat[idx++];
        final g = flat[idx++];
        final b = flat[idx++];
        return [r, g, b];
      }),
    ),
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