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

// ถ้าโมเดลให้ "inverse depth/disparity" (ค่ายิ่งมากยิ่งใกล้) ให้ true
// ถ้าโมเดลให้ "depth ตรง" (ค่ายิ่งมากยิ่งไกล) ให้ false (แล้วจะ 1/(v+eps))
const bool kModelIsInverseDepth = true;

// แปลงหน่วยเป็น “เมตร” แบบหยาบ: meters ≈ k / nearScore
double _metersPerUnit = 0.0; // ใส่ k > 0 หลังคาลิเบรตถ้าต้องการ

// ============ โซนสนใจ & multi-blob params ============
const double kRoiTopFrac       = 0.55;  // ใช้ช่วงล่าง ~45% ของภาพ
const double kHighPct          = 0.12;  // seed: “ใกล้มาก” (เปอร์เซ็นไทล์สูง)
const double kLowPct           = 0.25;  // mask: “ใกล้พอ”
const int    kMorphOpenIters   = 1;     // 0=ปิด, 1–2 เพื่อตัดสะพานบางๆ
const bool   kUse4Conn         = true;  // true=4-conn, false=8-conn
const double kMinBlobAreaFrac  = 0.010; // ≥1% ของภาพ
const double kMinAspect        = 0.35;  // กรอง blob เตี้ยยาว (h/w ต่ำเกินไป)
const double kMaxWidthFrac     = 0.90;  // กว้างกินเกินภาพ (มักเป็นพื้น)
const double kMinNearFrac      = 0.030; // พิกเซล "ใกล้พอ" ต้อง ≥3% ของ ROI
const double kMinDepthContrast = 0.08;  // p90-p10 ภายใน ROI ≥ 0.08

// ====== ตัวคัดเลือกจำนวน blob แบบไดนามิก ======
const int    kMaxBlobsCapBase   = 3;     // เพดานพื้นฐาน
const int    kMaxBlobsCapMax    = 8;     // เพดานสูงสุด
const double kDenseSceneBoostAt = 0.12;  // nearPixels/ROI > 12% ถือว่าฉากหนาแน่น
const double kIouSuppress       = 0.45;  // IoU เกินนี้ตัด
const double kMinMedianRel      = 0.02;  // ค่ากลางใกล้ขั้นต่ำ (กัน noise)
const double kMinAreaFracHard   = 0.006; // ≥0.6% ของภาพ (ดัก noise)

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
      title: 'Depth-only Obstacle Demo (Multi-Blob)',
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

  // ผลลัพธ์ depth ล่าสุด (near score – ค่ายิ่งมากยิ่งใกล้, ไม่ normalize ต่อเฟรม)
  _DepthField? _lastDepthField;

  // ผล blobs ล่าสุด
  List<_BlobResult> _lastBlobs = const [];

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

      final TransferableTypedData ttd = message['norm'] as TransferableTypedData; // nearScore (raw scale)
      final Float32List grid = ttd.materialize().asFloat32List();

      _lastDepthField = _DepthField(
        norm: grid, w: w, h: h, targetW: targetW, targetH: targetH,
      );

      if (_showHeatmap) {
        _depthOverlayPng = message['png'] as Uint8List?;
      }

      // วิเคราะห์หลาย blob ใน ROI ล่าง (เลือกรอบสุดท้ายแบบไดนามิก)
      _lastBlobs = (_lastDepthField != null)
          ? _extractNearestBlobs(_lastDepthField!)
          : const [];

      setState(() {
        final topZ = _lastBlobs.isNotEmpty ? _lastBlobs.first.medianRel : null;
        final zTxt = (topZ != null)
            ? (_metersPerUnit > 0
                ? "≈${(_metersPerUnit / (topZ + 1e-3)).toStringAsFixed(2)} m"
                : "z=${(topZ).toStringAsFixed(2)} (rel)")
            : "—";
        _status =
            "depth-only ✅ • depth:${_lastDepDurMs}ms/$kDepthIntervalMs • FPS:${_fpsNow.toStringAsFixed(1)} • blobs:${_lastBlobs.length} • $zTxt";
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

  // ====== วิเคราะห์หลาย blob (dual-threshold + morphology + CC + auto-select) ======
  List<_BlobResult> _extractNearestBlobs(_DepthField f) {
    final int w = f.w, h = f.h;
    if (w == 0 || h == 0 || f.norm.isEmpty) return const [];

    final int y0 = (kRoiTopFrac * h).clamp(0, h - 1).toInt();
    final int y1 = h - 1;

    // 1) รวบรวม nearScore ใน ROI (ค่ายิ่งมากยิ่งใกล้)
    final vals = <double>[];
    for (int y = y0; y <= y1; y++) {
      final base = y * w;
      for (int x = 0; x < w; x++) vals.add(f.norm[base + x]);
    }
    if (vals.isEmpty) return const [];

    vals.sort();
    final p10 = vals[(0.10 * (vals.length - 1)).toInt()];
    final p90 = vals[(0.90 * (vals.length - 1)).toInt()];
    if ((p90 - p10) < kMinDepthContrast) return const []; // ภาพแบน/ฉากโล่ง

    // 2) dual threshold
    final highThr = vals[(kHighPct * (vals.length - 1)).toInt()];
    final lowThr  = vals[(kLowPct  * (vals.length - 1)).toInt()];

    // 2.1) มาสก์แบบ low (ใกล้พอ) + seed แบบ high (ใกล้มาก)
    final lowMask  = Uint8List(w * h);
    final highMask = Uint8List(w * h);
    int nearCount = 0;
    for (int y = y0; y <= y1; y++) {
      final base = y * w;
      for (int x = 0; x < w; x++) {
        final v = f.norm[base + x];
        if (v >= lowThr)  { lowMask [base + x] = 1; nearCount++; }
        if (v >= highThr) { highMask[base + x] = 1; }
      }
    }
    final int roiPixels = (y1 - y0 + 1) * w;
    if (nearCount < (kMinNearFrac * roiPixels)) return const [];

    // 3) morphology opening เพื่อตัดสะพาน (กับ lowMask เท่านั้น)
    Uint8List mask = lowMask;
    for (int i = 0; i < kMorphOpenIters; i++) {
      mask = _erode(mask, w, h, y0, y1);
      mask = _dilate(mask, w, h, y0, y1);
    }

    // 4) region growing: เริ่มจาก seed (highMask) แล้วโตใน mask
    final visited = Uint8List(w * h);
    final blobs = <_BlobResult>[];
    int nextId = 1;

    void growFromSeed(int seedX, int seedY) {
      final qx = <int>[seedX];
      final qy = <int>[seedY];
      visited[seedY * w + seedX] = 1;

      int minX = seedX, maxX = seedX, minY = seedY, maxY = seedY;
      int area = 0;
      double cxSum = 0, cySum = 0;
      final idxs = <int>[];

      while (qx.isNotEmpty) {
        final x = qx.removeLast();
        final y = qy.removeLast();
        final i0 = y * w + x;
        area++;
        cxSum += x; cySum += y;
        idxs.add(i0);

        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;

        void tryPush(int nx, int ny) {
          if (nx < 0 || nx >= w || ny < y0 || ny > y1) return;
          final ii = ny * w + nx;
          if (visited[ii] == 0 && mask[ii] == 1) {
            visited[ii] = 1; qx.add(nx); qy.add(ny);
          }
        }
        if (kUse4Conn) {
          tryPush(x-1, y); tryPush(x+1, y); tryPush(x, y-1); tryPush(x, y+1);
        } else {
          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              if (dx==0 && dy==0) continue; tryPush(x+dx, y+dy);
            }
          }
        }
      }

      // กรอง shape/area
      final areaFrac = area / (w * h);
      final bw = (maxX - minX + 1).toDouble();
      final bh = (maxY - minY + 1).toDouble();
      final aspect = bh / (bw + 1e-6);
      final widthFrac = bw / w;

      if (areaFrac < kMinBlobAreaFrac) return;
      if (aspect < kMinAspect) return;
      if (widthFrac > kMaxWidthFrac) return;

      // median near ของ blob
      final vv = <double>[];
      for (final ii in idxs) { vv.add(f.norm[ii]); }
      vv.sort();
      final med = vv[vv.length ~/ 2];

      // scale preview
      final double scaleXf = f.targetW / w;
      final double scaleYf = f.targetH / h;

      final rect = Rect.fromLTWH(
        minX * scaleXf, minY * scaleYf,
        bw * scaleXf,   bh * scaleYf,
      );
      final cx = (cxSum / area) * scaleXf;
      final cy = (cySum / area) * scaleYf;

      blobs.add(_BlobResult(
        id: nextId++,
        bboxInPreview: rect,
        medianRel: med,
        areaFrac: areaFrac,
        centroidPreview: Offset(cx, cy),
      ));
    }

    // เริ่มจากทุก seed ใน highMask ที่อยู่ใน mask
    for (int y = y0; y <= y1; y++) {
      final base = y * w;
      for (int x = 0; x < w; x++) {
        final i = base + x;
        if (highMask[i] == 1 && mask[i] == 1 && visited[i] == 0) {
          growFromSeed(x, y);
        }
      }
    }

    if (blobs.isEmpty) return const [];

    // sort: ใกล้สุดก่อน (medianNear สูงก่อน)
    blobs.sort((a, b) => b.medianRel.compareTo(a.medianRel));

    // เลือกจำนวนก้อนแบบไดนามิก + NMS (IoU)
    final selected = _autoSelectBlobs(
      blobs: blobs,
      frameW: w,
      frameH: h,
      roiPixels: roiPixels,
      nearCount: nearCount,
    );
    return selected;
  }

  // morphology helpers (binary 0/1, 3x3)
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

  // ====== คัดเลือก blobs แบบไดนามิก + NMS ======
  List<_BlobResult> _autoSelectBlobs({
    required List<_BlobResult> blobs,
    required int frameW,
    required int frameH,
    required int roiPixels,
    required int nearCount,
  }) {
    if (blobs.isEmpty) return const [];

    // 1) ปรับเพดานตาม "ความหนาแน่นของฉากใกล้"
    final nearFrac = nearCount / (roiPixels.toDouble() + 1e-6);
    int cap = kMaxBlobsCapBase + (nearFrac > kDenseSceneBoostAt ? 2 : 0);
    if (nearFrac > (kDenseSceneBoostAt * 2.2)) cap += 2;
    if (cap > kMaxBlobsCapMax) cap = kMaxBlobsCapMax;

    // 2) NMS-like ด้วย IoU
    final picked = <_BlobResult>[];
    double iou(Rect a, Rect b) {
      final inter = a.intersect(b);
      if (inter.isEmpty) return 0.0;
      final interA = inter.width * inter.height;
      final unionA = a.width * a.height + b.width * b.height - interA;
      return unionA <= 0 ? 0.0 : interA / unionA;
    }

    for (final cand in blobs) {
      if (cand.medianRel < kMinMedianRel) continue;
      if (cand.areaFrac  < kMinAreaFracHard) continue;

      bool overlap = false;
      for (final p in picked) {
        if (iou(cand.bboxInPreview, p.bboxInPreview) > kIouSuppress) { overlap = true; break; }
      }
      if (!overlap) picked.add(cand);
      if (picked.length >= cap) break;
    }

    // 3) เผื่อกรณีว่าง/น้อยเกินไป
    if (picked.isEmpty) {
      for (final cand in blobs) {
        if (cand.areaFrac >= kMinAreaFracHard * 0.75) {
          picked.add(cand);
          if (picked.length >= math.max(1, cap ~/ 2)) break;
        }
      }
    }

    return picked;
  }

  @override
  Widget build(BuildContext context) {
    final camReady = _controller?.value.isInitialized == true;
    final previewSize = _controller?.value.previewSize;

    String medianText;
    if (_lastBlobs.isEmpty) {
      medianText = "Depth median in blobs = —";
    } else {
      final parts = _lastBlobs.take(5).map((b) {
        final z = b.medianRel;
        return (_metersPerUnit > 0)
            ? "#${b.id}:${(_metersPerUnit/(z+1e-3)).toStringAsFixed(2)}m"
            : "#${b.id}:z=${z.toStringAsFixed(3)}";
      }).join("  ");
      medianText = "Depth median in blobs →  $parts";
    }

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
                        // วาดหลายก้อน
                        IgnorePointer(
                          child: CustomPaint(
                            painter: _BlobsPainter(
                              blobs: _lastBlobs,
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
                          "Tip: ตั้ง _metersPerUnit > 0 เพื่อนำ nearScore → เมตร (≈ k/(z+eps))",
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
  final int id;
  final Rect bboxInPreview;     // พิกัดบนพื้นที่ preview (ยังไม่ scale ตาม canvas)
  final double medianRel;       // near score (ค่ายิ่งมากยิ่งใกล้) — ไม่ normalize ต่อเฟรม
  final double areaFrac;        // สัดส่วนพื้นที่ blob / ภาพทั้งหมด
  final Offset centroidPreview;
  _BlobResult({
    required this.id,
    required this.bboxInPreview,
    required this.medianRel,
    required this.areaFrac,
    required this.centroidPreview,
  });
}

class _BlobsPainter extends CustomPainter {
  final List<_BlobResult> blobs;
  final bool mirrorX; // วาดส่องกระจกถ้ากล้องหน้า
  final double srcW;  // ความกว้างของพรีวิวจากกล้อง
  final double srcH;  // ความสูงของพรีวิวจากกล้อง
  const _BlobsPainter({
    required this.blobs,
    this.mirrorX = false,
    required this.srcW,
    required this.srcH,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (blobs.isEmpty) return;

    final scaleX = size.width  / srcW;
    final scaleY = size.height / srcH;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.greenAccent;

    final dotPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.greenAccent.withOpacity(0.9);

    final bg = Paint()..color = Colors.black.withOpacity(0.55);

    for (final b in blobs) {
      Rect rect = Rect.fromLTWH(
        b.bboxInPreview.left   * scaleX,
        b.bboxInPreview.top    * scaleY,
        b.bboxInPreview.width  * scaleX,
        b.bboxInPreview.height * scaleY,
      );
      Offset c = Offset(b.centroidPreview.dx * scaleX, b.centroidPreview.dy * scaleY);

      if (mirrorX) {
        rect = Rect.fromLTRB(
          size.width - rect.right, rect.top,
          size.width - rect.left,  rect.bottom,
        );
        c = Offset(size.width - c.dx, c.dy);
      }

      canvas.drawRect(rect, boxPaint);
      canvas.drawCircle(c, 3.0, dotPaint);

      final label = _metersPerUnit > 0
          ? "Obs#${b.id} • ≈${(_metersPerUnit/(b.medianRel+1e-3)).toStringAsFixed(2)} m"
          : "Obs#${b.id} • z=${b.medianRel.toStringAsFixed(2)}";
      final tp = TextPainter(
        text: TextSpan(text: label, style: const TextStyle(color: Colors.white, fontSize: 12)),
        textDirection: TextDirection.ltr,
        maxLines: 1,
      )..layout();

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

  @override
  bool shouldRepaint(covariant _BlobsPainter old) =>
      old.blobs != blobs || old.mirrorX != mirrorX || old.srcW != srcW || old.srcH != srcH;
}

class _DepthField {
  final Float32List norm; // near score (ใหญ่=ใกล้) — ค่าดิบ ไม่ normalize ต่อเฟรม
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

      // อ่านผล: เก็บ nearScore "ไม่ normalize ต่อเฟรม"
      final outShape = interpreter.getOutputTensor(0).shape;
      final hOut = outShape[1];
      final wOut = outShape[2];
      final is4D = outShape.length == 4;

      final tmp = Float32List(wOut * hOut);
      double minV = double.infinity, maxV = -double.infinity; // ใช้ทำ heatmap เท่านั้น
      int k = 0;
      for (int y = 0; y < hOut; y++) {
        final row = depthOut![0][y];
        for (int x = 0; x < wOut; x++) {
          final vRaw = is4D ? (row[x][0] as num?)?.toDouble() ?? 0.0
                            : (row[x]     as num?)?.toDouble() ?? 0.0;
          final near = kModelIsInverseDepth ? vRaw : 1.0 / (vRaw + 1e-6);
          tmp[k++] = near;
          if (near < minV) minV = near;
          if (near > maxV) maxV = near;
        }
      }

      // heatmap PNG (normalize เฉพาะเพื่อแสดงผล)
      Uint8List? png;
      if (makeHeatmap) {
        final range = (maxV - minV).abs() < 1e-6 ? 1e-6 : (maxV - minV);
        final heat = img.Image(width: wOut, height: hOut);
        for (int y = 0; y < hOut; y++) {
          final base = y * wOut;
          for (int x = 0; x < wOut; x++) {
            final vv = ((tmp[base + x] - minV) / range).clamp(0.0, 1.0);
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
        'norm': TransferableTypedData.fromList([Uint8List.view(tmp.buffer)]), // nearScore (raw scale)
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