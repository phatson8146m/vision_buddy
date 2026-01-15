import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img; // ใช้ทำ heatmap PNG เท่านั้น

/// ====== Toggle delegate (เริ่มแบบปลอดภัย) ======
const bool kEnableNNAPI = false;        // เปิด true ถ้าทดสอบแล้วเสถียร
const bool kEnableGPUForDepth = false;  // เปิด true เฉพาะ depth (float32)

/// ====== ปรับแต่ง ======
const double _scoreThreshold = 0.30;   // detector จะ clamp >= 0.50 อีกชั้น
const int _desiredFps = 15;            // FPS เป้าหมายของ pipeline (ไม่ใช่กล้อง)
const int _minProcessIntervalMs = 90;  // กันรอบถี่เกิน; max กับ 1000/_desiredFps
final int _frameIntervalMs =
    math.max(_minProcessIntervalMs, (1000 / _desiredFps).round());

double _metersPerUnit = 0.0;           // 0 = โหมดสัมพัทธ์
const int _tfliteThreads = 4;          // เธรด CPU

// เก็บเฟรมล่าสุดกัน backlog (ทับของเก่า)
CameraImage? _latestFrame;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  FlutterError.onError = (FlutterErrorDetails details) {
    FlutterError.dumpErrorToConsole(details);
  };

  await runZonedGuarded(() async {
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
    final cameras = await availableCameras();
    runApp(MyApp(cameras: cameras));
  }, (error, stack) {
    debugPrint('Zoned error: $error');
    debugPrint(stack.toString());
  });
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Depth/Object App (4D input)',
      theme: ThemeData(primarySwatch: Colors.blue),
      debugShowCheckedModeBanner: false,
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
  CameraController? _controller;
  tfl.Interpreter? _detector;
  tfl.Interpreter? _depthModel;

  String _status = "Initializing...";
  List<String> _labels = [];

  // ====== Queue-less pipeline control ======
  bool _isProcessing = false;
  int _lastProcMs = 0;
  int _lastProcDurMs = 0;
  int _iter = 0;
  int _depthEvery = 2;   // เริ่มรัน depth ทุก ๆ 2 รอบ
  int _phase = 0;        // 0=det, 1=depth

  // ====== FPS measure ======
  final Stopwatch _fpsWindow = Stopwatch();
  int _framesInWindow = 0;
  double _fpsNow = 0.0;
  final Stopwatch _procSw = Stopwatch();

  // แสดงผล
  bool _showHeatmap = false; // ปิดไว้ก่อนเพื่อความลื่น
  List<_Detection> _lastDetections = [];
  _DepthField? _depthCache;
  List<_Detection> _detections = [];
  Uint8List? _depthOverlayPng;

  // บัฟเฟอร์ TFLite re-use (ลด alloc)
  List? _detOut0, _detOut1, _detOut2, _detOut3;
  List? _depthOut;

  // ขนาดอินพุตแต่ละโมเดล (cache)
  int? _detInH, _detInW;
  int? _depthInH, _depthInW;
  bool _detInputIsFloat = true; // จะเช็คจริงตอน init

  @override
  void initState() {
    super.initState();
    _fpsWindow.start();
    _initAll();
  }

  Future<void> _initAll() async {
    try {
      final camera = widget.cameras.isNotEmpty ? widget.cameras.first : null;
      if (camera == null) {
        setState(() => _status = "No camera found");
        return;
      }

      _controller = CameraController(
        camera,
        ResolutionPreset.low,           // เน้นลื่นก่อน (เครื่องแรงค่อยขยับ)
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _controller!.initialize();

      // labels: index 0-based
      _labels = await _loadLabels('assets/models/labels.txt');

      // detector
      _detector = await _createInterpreter(
        'assets/models/1.tflite',
        preferGpu: false,
        threads: _tfliteThreads,
      );
      final detIn = _detector!.getInputTensor(0);
      _detInH = detIn.shape[1];
      _detInW = detIn.shape[2];
      _detInputIsFloat =
          detIn.type.toString().toLowerCase().contains('float32');

      // depth
      _depthModel = await _createInterpreter(
        'assets/models/model_opt.tflite',
        preferGpu: kEnableGPUForDepth,
        threads: _tfliteThreads,
      );
      final dIn = _depthModel!.getInputTensor(0);
      _depthInH = dIn.shape[1];
      _depthInW = dIn.shape[2];

      // ===== Debug: พิมพ์รูปร่าง/ชนิดอินพุตเอาต์พุต =====
      debugPrint('=== DETECTOR ===');
      debugPrint('input[0].shape=${detIn.shape}, type=${detIn.type}');
      for (int i = 0; i < _detector!.getOutputTensors().length; i++) {
        final t = _detector!.getOutputTensor(i);
        debugPrint('output[$i].shape=${t.shape}, type=${t.type}');
      }
      debugPrint('=== DEPTH ===');
      debugPrint('input[0].shape=${dIn.shape}, type=${dIn.type}');
      for (int i = 0; i < _depthModel!.getOutputTensors().length; i++) {
        final t = _depthModel!.getOutputTensor(i);
        debugPrint('output[$i].shape=${t.shape}, type=${t.type}');
      }

      await _controller!.startImageStream((frame) {
        _latestFrame = frame; // ทับเฟรมเก่า ไม่ทำคิว
      });

      _startProcessLoop();

      setState(() => _status = "Real-time running ✅");
    } catch (e) {
      setState(() => _status = "Init error: $e");
    }
  }

  Future<tfl.Interpreter> _createInterpreter(
    String asset, {
    required bool preferGpu,
    required int threads,
  }) async {
    final opts = tfl.InterpreterOptions()
      ..threads = threads
      ..useNnApiForAndroid = kEnableNNAPI;

    if (preferGpu) {
      try {
        final gpuDelegate = tfl.GpuDelegateV2(
          options: tfl.GpuDelegateOptionsV2(isPrecisionLossAllowed: true),
        );
        opts.addDelegate(gpuDelegate);
      } catch (e) {
        debugPrint("GPU delegate init failed: $e");
      }
    }

    return await tfl.Interpreter.fromAsset(asset, options: opts);
  }

  @override
  void dispose() {
    _controller?.dispose();
    _detector?.close();
    _depthModel?.close();
    super.dispose();
  }

  Future<List<String>> _loadLabels(String assetPath) async {
    final raw = await rootBundle.loadString(assetPath);
    return raw
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty && e != '???')
        .toList();
  }

  // ====== MAIN PROCESS LOOP ======
  Future<void> _startProcessLoop() async {
    while (mounted) {
      final now = DateTime.now().millisecondsSinceEpoch;

      if (!_isProcessing &&
          (now - _lastProcMs) >= _frameIntervalMs &&
          _latestFrame != null) {
        _isProcessing = true;
        _lastProcMs = now;

        final frame = _latestFrame!;
        _procSw..reset()..start();

        try {
          await _processFrame(frame);
        } catch (e) {
          if (!mounted) break;
          setState(() => _status = "Runtime error: $e");
        } finally {
          _procSw.stop();
          _lastProcDurMs = _procSw.elapsedMilliseconds;

          // อัปเดต FPS ทุก ~0.5s
          _framesInWindow++;
          if (_fpsWindow.elapsedMilliseconds >= 500) {
            _fpsNow =
                (_framesInWindow * 1000) / _fpsWindow.elapsedMilliseconds;
            _framesInWindow = 0;
            _fpsWindow..reset()..start();
            if (mounted) setState(() {}); // รีเฟรช overlay
          }

          // ปรับความถี่ depth อัตโนมัติ
          if (_lastProcDurMs > (_frameIntervalMs * 1.2)) {
            _depthEvery = math.min(4, _depthEvery + 1); // ช้า → ผ่อน depth
          } else if (_lastProcDurMs < (_frameIntervalMs * 0.8)) {
            _depthEvery = math.max(2, _depthEvery - 1); // เร็ว → ถี่ขึ้น
          }

          _isProcessing = false;
        }
      }

      await Future.delayed(const Duration(milliseconds: 2));
    }
  }

  Future<void> _processFrame(CameraImage camImg) async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_detector == null || _depthModel == null) return;

    final previewW =
        _controller!.value.previewSize?.width ?? camImg.width.toDouble();
    final previewH =
        _controller!.value.previewSize?.height ?? camImg.height.toDouble();

    final runDepthThisRound = (_iter % _depthEvery == 0);

    if (_phase == 0 || !runDepthThisRound) {
      // ---------- Detection ----------
      final dets = await _runObjectDetection(
        camImg,
        previewW: previewW,
        previewH: previewH,
      );
      _lastDetections = dets;

      final enriched = (_depthCache != null)
          ? _attachDepthToDetections(
              dets,
              _DepthResult(heatmapPng: _depthOverlayPng, field: _depthCache!),
            )
          : dets;

      if (!mounted) return;
      setState(() {
        _detections = enriched;
        _status =
            "Real-time ✅  objs:${enriched.length}  • proc:${_lastProcDurMs}ms";
      });
      _phase = 1;
    } else {
      // ---------- Depth ----------
      final depthRes = await _runDepth(
        camImg,
        targetW: previewW,
        targetH: previewH,
        makeHeatmap: _showHeatmap,
      );
      _depthCache = depthRes.field;
      if (_showHeatmap) _depthOverlayPng = depthRes.heatmapPng;

      final enriched = _attachDepthToDetections(_lastDetections, depthRes);

      if (!mounted) return;
      setState(() {
        _detections = enriched;
        _status =
            "Real-time ✅  objs:${enriched.length}  • proc:${_lastProcDurMs}ms";
      });
      _phase = 0;
    }

    _iter++;
  }

  // ========================== Detection ==========================
  Future<List<_Detection>> _runObjectDetection(
    CameraImage cam, {
    required double previewW,
    required double previewH,
  }) async {
    final interpreter = _detector!;
    final inT = interpreter.getInputTensor(0);
    final inShape = inT.shape; // [1,H,W,3]
    if (inShape.length != 4 || inShape[0] != 1 || inShape[3] != 3) {
      throw Exception("Detector expects [1,H,W,3], got $inShape");
    }
    final H = _detInH!;
    final W = _detInW!;
    final isFloat = _detInputIsFloat;

    // 1) YUV420 → RGB (flat) ขนาด HxW
    final converted = yuvToModelInputBuffer(
      cam: cam,
      dstH: H,
      dstW: W,
      isFloat: isFloat,
    );

    // 2) สร้างอินพุตแบบ [1,H,W,3]
    final input4D = isFloat
        ? build4DFloatFromFlat(converted.buffer as Float32List, H, W)
        : build4DUint8FromFlat(converted.buffer as Uint8List, H, W);

    // 3) เตรียมเอาต์พุตและ run
    _ensureDetOutBuffers(interpreter);
    final outMap = <int, Object>{
      0: _detOut0!, 1: _detOut1!, 2: _detOut2!,
      if (_detOut3 != null) 3: _detOut3!,
    };
    interpreter.runForMultipleInputs([input4D], outMap);

    final boxes = _detOut0!;
    final classes = _detOut1!;
    final scores = _detOut2!;
    if (boxes.isEmpty || classes.isEmpty || scores.isEmpty) return const <_Detection>[];
    if (scores[0] is! List) return const <_Detection>[];

    int nCand = (scores[0] as List).length;
    if (_detOut3 != null && _detOut3!.isNotEmpty) {
      final c = (_detOut3![0] is List && _detOut3![0].isNotEmpty)
          ? ((_detOut3![0][0] as num?)?.toInt() ?? nCand)
          : ((_detOut3![0] as num?)?.toInt() ?? nCand);
      if (c > 0 && c <= nCand) nCand = c;
    }
    if (nCand == 0) return const <_Detection>[];

    const double iouTh = 0.50;
    const int topK = 60;
    const double minBoxFrac = 0.010;
    const double maxBoxFrac = 0.80;
    final double scoreTh = math.max(_scoreThreshold, 0.50);

    final totalArea = previewW * previewH;
    final cand = <_Detection>[];

    for (int i = 0; i < nCand; i++) {
      final s = (scores[0][i] as num?)?.toDouble() ?? 0.0;
      if (s < scoreTh) continue;

      final rawCls = (classes[0][i] as num?)?.toInt() ?? -1;
      final clsIdx = rawCls;
      final label =
          (clsIdx >= 0 && clsIdx < _labels.length) ? _labels[clsIdx] : "id:$rawCls";

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
      if (frac < minBoxFrac || frac > maxBoxFrac) continue;

      cand.add(_Detection(
        rect: Rect.fromLTRB(left, top, right, bottom),
        label: label,
        score: s,
      ));
    }

    if (cand.length > 1) cand.sort((a, b) => b.score.compareTo(a.score));
    final preNms = (cand.length > topK) ? cand.sublist(0, topK) : cand;
    return _nms(preNms, iouThreshold: iouTh);
  }

  void _ensureDetOutBuffers(tfl.Interpreter interpreter) {
    if (_detOut0 != null) return;
    final o0s = interpreter.getOutputTensor(0).shape; // [1,N,4]
    final o1s = interpreter.getOutputTensor(1).shape; // [1,N]
    final o2s = interpreter.getOutputTensor(2).shape; // [1,N]
    _detOut0 = _zerosByShape(o0s, asFloat: true);
    _detOut1 = _zerosByShape(o1s, asFloat: true);
    _detOut2 = _zerosByShape(o2s, asFloat: true);
    if (interpreter.getOutputTensors().length >= 4) {
      final o3s = interpreter.getOutputTensor(3).shape; // [1] หรือ [1,1]
      _detOut3 = _zerosByShape(o3s, asFloat: true);
    }
  }

  // ============================ Depth ============================
  Future<_DepthResult> _runDepth(
    CameraImage cam, {
    required double targetW,
    required double targetH,
    required bool makeHeatmap,
  }) async {
    final interpreter = _depthModel!;
    final inT = interpreter.getInputTensor(0);
    final inShape = inT.shape; // [1,H,W,3]
    if (inShape.length != 4 || inShape[0] != 1 || inShape[3] != 3) {
      throw Exception("Depth expects [1,H,W,3], got $inShape");
    }
    final H = _depthInH!;
    final W = _depthInW!;

    // 1) YUV420 → RGB (float) ขนาด HxW
    final converted = yuvToModelInputBuffer(
      cam: cam,
      dstH: H,
      dstW: W,
      isFloat: true, // depth โมเดลส่วนใหญ่ float32
    );

    // 2) อินพุตแบบ [1,H,W,3]
    final input4D = build4DFloatFromFlat(converted.buffer as Float32List, H, W);

    // 3) เตรียม output และ run
    final outT = interpreter.getOutputTensor(0);
    final outShape = outT.shape; // [1,h,w,1] หรือ [1,h,w]
    _depthOut ??= _zerosByShape(outShape, asFloat: true);
    interpreter.run(input4D, _depthOut!);

    if (_depthOut == null || _depthOut!.isEmpty) {
      return _DepthResult(
        heatmapPng: null,
        field: _DepthField(norm: const [], w: 0, h: 0, targetW: targetW, targetH: targetH),
      );
    }

    final hOut = outShape[1];
    final wOut = outShape[2];
    final is4D = outShape.length == 4;

    final field = List.generate(hOut, (_) => List<double>.filled(wOut, 0.0));
    double minV = double.infinity, maxV = -double.infinity;

    for (int y = 0; y < hOut; y++) {
      final rowOut = _depthOut![0][y];
      for (int x = 0; x < wOut; x++) {
        final v = is4D
            ? (rowOut[x][0] as num?)?.toDouble() ?? 0.0
            : (rowOut[x]     as num?)?.toDouble() ?? 0.0;
        field[y][x] = v;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
    final range = (maxV - minV).abs() < 1e-6 ? 1e-6 : (maxV - minV);
    for (int y = 0; y < hOut; y++) {
      for (int x = 0; x < wOut; x++) {
        field[y][x] = (field[y][x] - minV) / range; // 0..1 (มาก=ใกล้)
      }
    }

    Uint8List? png;
    if (makeHeatmap) {
      final heat = img.Image(width: wOut, height: hOut);
      for (int y = 0; y < hOut; y++) {
        for (int x = 0; x < wOut; x++) {
          final v = field[y][x];
          final c = _colormapJet(v);
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

    return _DepthResult(
      heatmapPng: png,
      field: _DepthField(norm: field, w: wOut, h: hOut, targetW: targetW, targetH: targetH),
    );
  }

  // แนบ median depth (0..1) เข้าไปในกรอบ
  List<_Detection> _attachDepthToDetections(List<_Detection> dets, _DepthResult depth) {
    final f = depth.field;
    if (f.w == 0 || f.h == 0 || f.norm.isEmpty) return dets;
    final w = f.w.toDouble(), h = f.h.toDouble();
    final sx = w / f.targetW,  sy = h / f.targetH;

    final enriched = <_Detection>[];
    for (final d in dets) {
      final l = (d.rect.left   * sx).clamp(0, w - 1).toDouble();
      final t = (d.rect.top    * sy).clamp(0, h - 1).toDouble();
      final r = (d.rect.right  * sx).clamp(0, w - 1).toDouble();
      final b = (d.rect.bottom * sy).clamp(0, f.h - 1).toDouble();
      if (r <= l || b <= t) { enriched.add(d); continue; }

      const int S = 6; // sample 6x6
      final stepX = math.max(1.0, (r - l) / (S - 1));
      final stepY = math.max(1.0, (b - t) / (S - 1));
      final samples = <double>[];

      for (double yy = t; yy <= b; yy += stepY) {
        final yi = yy.round().clamp(0, f.h - 1);
        final row = f.norm[yi];
        for (double xx = l; xx <= r; xx += stepX) {
          final xi = xx.round().clamp(0, f.w - 1);
          samples.add(row[xi]);
        }
      }
      samples.sort();
      final median = samples[samples.length ~/ 2];

      String depthText = "z=${median.toStringAsFixed(2)} (rel)";
      if (_metersPerUnit > 0) {
        final eps = 1e-3;
        final meters = _metersPerUnit / (median + eps);
        depthText = "≈${meters.toStringAsFixed(2)} m";
      }

      enriched.add(d.copyWith(extra: depthText));
    }
    return enriched;
  }

  // =================== YUV420 -> input buffer (FAST) ===================
  /// คืนค่าเป็น buffer (Float32List หรือ Uint8List) ความยาว H*W*3 (เรียง R,G,B)
  ({Object buffer, int H, int W}) yuvToModelInputBuffer({
    required CameraImage cam,
    required int dstH,
    required int dstW,
    required bool isFloat,
  }) {
    final yPlane = cam.planes[0];
    final uPlane = cam.planes[1];
    final vPlane = cam.planes[2];
    final yBytes = yPlane.bytes;
    final uBytes = uPlane.bytes;
    final vBytes = vPlane.bytes;

    final yRowStride = yPlane.bytesPerRow;
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    final srcW = cam.width;
    final srcH = cam.height;

    final len = dstH * dstW * 3;

    if (isFloat) {
      final out = Float32List(len);
      int o = 0;
      for (int y = 0; y < dstH; y++) {
        final sy = (y * (srcH - 1)) ~/ (dstH - 1);
        final yOff = sy * yRowStride;
        final uvRow = (sy >> 1) * uvRowStride;
        for (int x = 0; x < dstW; x++) {
          final sx = (x * (srcW - 1)) ~/ (dstW - 1);
          final uvCol = (sx >> 1) * uvPixelStride;

          int yIdx = yOff + sx;
          if (yIdx >= yBytes.length) yIdx = yBytes.length - 1;
          int uIdx = uvRow + uvCol;
          int vIdx = uvRow + uvCol;
          if (uIdx >= uBytes.length) uIdx = uBytes.length - 1;
          if (vIdx >= vBytes.length) vIdx = vBytes.length - 1;

          final Y = yBytes[yIdx] & 0xFF;
          final U = (uBytes[uIdx] & 0xFF) - 128;
          final V = (vBytes[vIdx] & 0xFF) - 128;

          int r = Y + ((1436 * V) >> 10);                 // 1.402 * 1024 ≈ 1436
          int g = Y - ((352 * U + 731 * V) >> 10);        // 0.344, 0.714
          int b = Y + ((1815 * U) >> 10);                 // 1.772 * 1024 ≈ 1815

          if (r < 0) r = 0; else if (r > 255) r = 255;
          if (g < 0) g = 0; else if (g > 255) g = 255;
          if (b < 0) b = 0; else if (b > 255) b = 255;

          out[o++] = r / 255.0;
          out[o++] = g / 255.0;
          out[o++] = b / 255.0;
        }
      }
      return (buffer: out, H: dstH, W: dstW);
    } else {
      final out = Uint8List(len);
      int o = 0;
      for (int y = 0; y < dstH; y++) {
        final sy = (y * (srcH - 1)) ~/ (dstH - 1);
        final yOff = sy * yRowStride;
        final uvRow = (sy >> 1) * uvRowStride;
        for (int x = 0; x < dstW; x++) {
          final sx = (x * (srcW - 1)) ~/ (dstW - 1);
          final uvCol = (sx >> 1) * uvPixelStride;

          int yIdx = yOff + sx;
          if (yIdx >= yBytes.length) yIdx = yBytes.length - 1;
          int uIdx = uvRow + uvCol;
          int vIdx = uvRow + uvCol;
          if (uIdx >= uBytes.length) uIdx = uBytes.length - 1;
          if (vIdx >= vBytes.length) vIdx = vBytes.length - 1;

          final Y = yBytes[yIdx] & 0xFF;
          final U = (uBytes[uIdx] & 0xFF) - 128;
          final V = (vBytes[vIdx] & 0xFF) - 128;

          int r = Y + ((1436 * V) >> 10);
          int g = Y - ((352 * U + 731 * V) >> 10);
          int b = Y + ((1815 * U) >> 10);

          if (r < 0) r = 0; else if (r > 255) r = 255;
          if (g < 0) g = 0; else if (g > 255) g = 255;
          if (b < 0) b = 0; else if (b > 255) b = 255;

          out[o++] = r;
          out[o++] = g;
          out[o++] = b;
        }
      }
      return (buffer: out, H: dstH, W: dstW);
    }
  }

  /// แปลง Float32List (H*W*3) → [1,H,W,3] สำหรับ float32
  List build4DFloatFromFlat(Float32List flat, int H, int W) {
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

  /// แปลง Uint8List (H*W*3) → [1,H,W,3] สำหรับ uint8
  List build4DUint8FromFlat(Uint8List flat, int H, int W) {
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

  // ============================ Utils ============================
  List _zerosByShape(List<int> shape, {required bool asFloat}) {
    dynamic build(List<int> s) {
      if (s.isEmpty) return asFloat ? 0.0 : 0;
      return List.generate(s.first, (_) => build(s.sublist(1)));
    }
    return build(shape);
  }

  List<int> _colormapJet(double v) {
    v = v.clamp(0.0, 1.0);
    final fourV = 4 * v;
    final r = (255 * (fourV - 1.5).clamp(0.0, 1.0)).toInt();
    final g = (255 * (1.5 - (fourV - 2.0).abs()).clamp(0.0, 1.0)).toInt();
    final b = (255 * (1.5 - (fourV - 0.5).abs()).clamp(0.0, 1.0)).toInt();
    return [r, g, b];
  }

  List<_Detection> _nms(List<_Detection> dets, {double iouThreshold = 0.5}) {
    if (dets.length <= 1) return dets;
    final sorted = [...dets]..sort((a, b) => b.score.compareTo(a.score));
    final picked = <_Detection>[];
    for (final d in sorted) {
      bool keep = true;
      for (final p in picked) {
        if (_iou(d.rect, p.rect) > iouThreshold) { keep = false; break; }
      }
      if (keep) picked.add(d);
    }
    return picked;
  }

  double _iou(Rect a, Rect b) {
    final inter = Rect.fromLTRB(
      math.max(a.left, b.left),
      math.max(a.top, b.top),
      math.min(a.right, b.right),
      math.min(a.bottom, b.bottom),
    );
    final interArea =
        math.max(0.0, inter.width) * math.max(0.0, inter.height);
    final unionArea = a.width * a.height + b.width * b.height - interArea;
    if (unionArea <= 0) return 0;
    return interArea / unionArea;
  }

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
                          IgnorePointer(
                            child: Image.memory(_depthOverlayPng!, fit: BoxFit.cover),
                          ),
                        IgnorePointer(
                          child: CustomPaint(painter: _DetectionsPainter(_detections)),
                        ),
                      ],
                    ),
                  ),
                ),

                // ====== สเตตัส (กลางบน) ======
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

                // ====== FPS Overlay (ซ้ายบน) ======
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
                        "FPS: ${_fpsNow.toStringAsFixed(1)}  • budget:${_frameIntervalMs}ms  • last:${_lastProcDurMs}ms  • depth/ ${_depthEvery}r",
                        style: const TextStyle(color: Colors.white, fontSize: 12),
                      ),
                    ),
                  ),
                ),

                // คำอธิบาย depth (ล่างกลาง)
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

                // ปุ่ม toggle heatmap (ขวาบน)
                SafeArea(
                  child: Align(
                    alignment: Alignment.topRight,
                    child: IconButton(
                      onPressed: () => setState(() => _showHeatmap = !_showHeatmap),
                      icon: Icon(Icons.heat_pump_outlined,
                          color: _showHeatmap ? Colors.orange : Colors.white),
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

// ============================ Drawing ============================
class _Detection {
  final Rect rect; // พิกเซลในพรีวิว
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
  _DetectionsPainter(this.dets);

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.greenAccent;

    final bg = Paint()..color = Colors.black.withOpacity(0.55);

    for (final d in dets) {
      final rect = d.rect;
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
  bool shouldRepaint(covariant _DetectionsPainter oldDelegate) => oldDelegate.dets != dets;
}

// ============================ Depth structs ============================
class _DepthField {
  final List<List<double>> norm; // [H][W] 0..1 (มาก=ใกล้)
  final int w;
  final int h;
  final double targetW; // ขนาดพรีวิว (ใช้ map กรอบ → depth grid)
  final double targetH;
  _DepthField({required this.norm, required this.w, required this.h, required this.targetW, required this.targetH});
}

class _DepthResult {
  final Uint8List? heatmapPng; // null เมื่อไม่สร้าง
  final _DepthField field;
  _DepthResult({required this.heatmapPng, required this.field});
}