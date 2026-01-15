import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

/// ตั้งค่าเริ่มต้น
const ResolutionPreset kResolution = ResolutionPreset.medium; // low/medium/high/ultraHigh/max
const bool kUseYuv420 = true; // ใช้รูปแบบ YUV420 สำหรับสตรีมภาพ
const int kUiThrottleMs = 120; // หน่วงการอัปเดต UI ไม่ให้ถี่เกินไป

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(App(cameras: cameras));
}

class App extends StatelessWidget {
  final List<CameraDescription> cameras;
  const App({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Camera FPS Monitor',
      debugShowCheckedModeBanner: false,
      home: FpsCameraPage(cameras: cameras),
    );
  }
}

/// โครงสร้างเก็บตัวอย่าง FPS ต่อวินาที
class _FpsSample {
  final int elapsedSec;
  final double inst;
  final double avg;
  _FpsSample(this.elapsedSec, this.inst, this.avg);
}

/// หน้าแสดงกล้อง + FPS
class FpsCameraPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const FpsCameraPage({super.key, required this.cameras});

  @override
  State<FpsCameraPage> createState() => _FpsCameraPageState();
}

class _FpsCameraPageState extends State<FpsCameraPage> {
  CameraController? _controller;
  String _status = 'Initializing...';

  // ตัวชี้วัด FPS
  double _fpsInstant = 0.0;   // คำนวณจากระยะห่างของเฟรมล่าสุด
  double _fpsAvg = 0.0;       // Exponential moving average ให้ค่าดูนิ่งขึ้น
  int _lastFrameMs = 0;
  int _winStartMs = 0;
  int _framesInWin = 0;
  int _lastUiUpdateMs = 0;

  // เลือกกล้องตัวแรกเป็นค่าเริ่มต้น
  int _cameraIndex = 0;

  // ====== ตัวแปรสำหรับการล็อก FPS รายวินาที ======
  final int _logSecondsTarget = 60; // 1 นาที
  Timer? _logTimer;
  bool _logging = false;
  final List<_FpsSample> _samples = [];

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      if (widget.cameras.isEmpty) {
        setState(() => _status = 'No cameras found.');
        return;
      }

      final cam = widget.cameras[_cameraIndex];
      final controller = CameraController(
        cam,
        kResolution,
        enableAudio: false,
        imageFormatGroup: kUseYuv420 ? ImageFormatGroup.yuv420 : ImageFormatGroup.bgra8888,
      );

      await controller.initialize();
      await controller.startImageStream(_onFrame);

      setState(() {
        _controller?.dispose(); // ทิ้งตัวเก่าหลังตั้งค่าสำเร็จ
        _controller = controller;
        _status = 'Running ✅ (${cam.lensDirection.name})';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = 'Init error: $e');
    }
  }

  @override
  void dispose() {
    _logTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  // เปลี่ยนกล้องหน้า/หลัง (ถ้ามี)
  Future<void> _switchCamera() async {
    if (widget.cameras.length < 2) return;
    _cameraIndex = (_cameraIndex + 1) % widget.cameras.length;
    _resetFps();
    await _controller?.stopImageStream().catchError((_) {});
    await _controller?.dispose();
    _controller = null;
    setState(() => _status = 'Switching camera...');
    await _initialize();
  }

  void _resetFps() {
    _fpsInstant = 0.0;
    _fpsAvg = 0.0;
    _lastFrameMs = 0;
    _winStartMs = 0;
    _framesInWin = 0;
    _lastUiUpdateMs = 0;
  }

  // Callback เมื่อได้เฟรมใหม่
  void _onFrame(CameraImage image) {
    final now = DateTime.now().millisecondsSinceEpoch;

    // Instant FPS
    if (_lastFrameMs != 0) {
      final dt = now - _lastFrameMs;
      final inst = 1000.0 / (dt <= 0 ? 1 : dt);
      _fpsInstant = inst;
      // EMA: 85% เก่า + 15% ใหม่
      _fpsAvg = _fpsAvg == 0.0 ? inst : (_fpsAvg * 0.85 + inst * 0.15);
    }
    _lastFrameMs = now;

    // Windowed FPS (สำรอง)
    _framesInWin++;
    if (_winStartMs == 0) _winStartMs = now;
    final winDt = now - _winStartMs;
    if (winDt >= 1000) {
      final fpsWin = _framesInWin * 1000.0 / (winDt == 0 ? 1 : winDt);
      _fpsAvg = _fpsAvg == 0.0 ? fpsWin : (_fpsAvg * 0.7 + fpsWin * 0.3);
      _winStartMs = now;
      _framesInWin = 0;
    }

    // อัปเดต UI ไม่ถี่เกินไป
    if (now - _lastUiUpdateMs >= kUiThrottleMs && mounted) {
      _lastUiUpdateMs = now;
      setState(() {});
    }
  }

  // ====== ฟังก์ชันเริ่ม/หยุด "ล็อก FPS ทุกวินาที 1 นาที" ======
  void _start1MinLogging() {
    if (_logging) return;
    _samples.clear();
    _logging = true;
    debugPrint('=== FPS LOG START (${DateTime.now()}) ===');

    int tick = 0;
    _logTimer = Timer.periodic(const Duration(seconds: 1), (t) {
      tick++;
      final inst = _fpsInstant;
      final avg = _fpsAvg;
      _samples.add(_FpsSample(tick, inst, avg));

      debugPrint('[t=${tick.toString().padLeft(2, '0')}s] '
          'inst=${inst.toStringAsFixed(1)}  avg=${avg.toStringAsFixed(1)}');

      if (tick >= _logSecondsTarget) {
        t.cancel();
        _logging = false;
        _finalizeLog();
      }
      if (mounted) setState(() {});
    });
    setState(() {});
  }

  void _stopLoggingEarly() {
    if (!_logging) return;
    _logTimer?.cancel();
    _logging = false;
    _finalizeLog();
    setState(() {});
  }

  void _finalizeLog() {
    final n = _samples.length;
    if (n == 0) {
      debugPrint('=== FPS LOG END (no samples) ===');
      return;
    }
    final avgVals = _samples.map((s) => s.avg).toList();
    double sum = 0.0, minV = avgVals.first, maxV = avgVals.first;
    for (final v in avgVals) {
      sum += v;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    final mean = sum / n;

    debugPrint('=== FPS LOG END (${DateTime.now()}) ===');
    debugPrint('Samples=$n  mean(avg)=${mean.toStringAsFixed(2)}  '
        'min=${minV.toStringAsFixed(1)}  max=${maxV.toStringAsFixed(1)}');

    // ส่งออกเป็น CSV-friendly text ให้ copy ได้ง่าย
    final buf = StringBuffer('elapsed_s,inst_fps,avg_fps\n');
    for (final s in _samples) {
      buf.writeln('${s.elapsedSec},'
          '${s.inst.toStringAsFixed(2)},'
          '${s.avg.toStringAsFixed(2)}');
    }
    debugPrint(buf.toString());
  }

  @override
  Widget build(BuildContext context) {
    final ready = _controller?.value.isInitialized == true;
    final previewSize = ready ? _controller!.value.previewSize : null;

    return Scaffold(
      backgroundColor: Colors.black,
      body: ready
          ? Stack(
              fit: StackFit.expand,
              children: [
                Center(
                  child: AspectRatio(
                    aspectRatio: _controller!.value.aspectRatio,
                    child: CameraPreview(_controller!),
                  ),
                ),
                SafeArea(
                  child: Align(
                    alignment: Alignment.topCenter,
                    child: Container(
                      margin: const EdgeInsets.only(top: 8),
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '${_status}'
                        '${previewSize != null ? ' • ${previewSize.width.toStringAsFixed(0)}×${previewSize.height.toStringAsFixed(0)}' : ''}'
                        ' • FPS: ${_fpsInstant.toStringAsFixed(1)} (avg ${_fpsAvg.toStringAsFixed(1)})'
                        ' • ${_presetName(kResolution)}'
                        '${kUseYuv420 ? ' • YUV420' : ' • BGRA'}'
                        '${_logging ? ' • LOG ${_samples.length}/$_logSecondsTarget' : ''}',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            )
          : const Center(child: CircularProgressIndicator()),
      // ปุ่มลอย: สลับกล้อง + เริ่ม/หยุดล็อก 1 นาที
      floatingActionButton: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (widget.cameras.length >= 2)
            FloatingActionButton.small(
              onPressed: _switchCamera,
              heroTag: 'switch_cam',
              child: const Icon(Icons.cameraswitch),
            ),
          const SizedBox(height: 12),
          FloatingActionButton(
            onPressed: _logging ? _stopLoggingEarly : _start1MinLogging,
            heroTag: 'start_log',
            child: Icon(_logging ? Icons.stop : Icons.speed),
          ),
        ],
      ),
    );
  }
}

String _presetName(ResolutionPreset p) {
  // รองรับ Flutter/Dart รุ่นที่ enum.name อาจยังไม่มี
  final s = p.toString(); // e.g., ResolutionPreset.medium
  final i = s.indexOf('.');
  return i >= 0 ? s.substring(i + 1) : s;
}