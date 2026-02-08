import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';

void main() {
  runApp(const MovieMakerApp());
}

class MovieMakerApp extends StatelessWidget {
  const MovieMakerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'News Short Generator Studio',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const StudioShell(),
    );
  }
}

class StudioShell extends StatefulWidget {
  const StudioShell({super.key});

  @override
  State<StudioShell> createState() => _StudioShellState();
}

class _StudioShellState extends State<StudioShell> {
  final List<String> _pages = const [
    '動画生成',
    '台本生成',
    '動画タイトル・説明',
    '資料作成',
    '動画編集',
    '詳細動画編集',
    '設定',
  ];
  int _selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          NavigationRail(
            selectedIndex: _selectedIndex,
            onDestinationSelected: (index) {
              setState(() {
                _selectedIndex = index;
              });
            },
            labelType: NavigationRailLabelType.all,
            destinations: _pages
                .map(
                  (page) => NavigationRailDestination(
                    icon: const Icon(Icons.circle_outlined),
                    selectedIcon: const Icon(Icons.circle),
                    label: Text(page),
                  ),
                )
                .toList(),
          ),
          const VerticalDivider(width: 1),
          Expanded(
            flex: 3,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: _buildCenterPanel(),
            ),
          ),
          const VerticalDivider(width: 1),
          Expanded(
            flex: 2,
            child: LogPanel(pageName: _pages[_selectedIndex]),
          ),
        ],
      ),
    );
  }

  Widget _buildCenterPanel() {
    switch (_selectedIndex) {
      case 0:
        return const VideoGenerateForm();
      default:
        return PlaceholderPanel(title: _pages[_selectedIndex]);
    }
  }
}

class PlaceholderPanel extends StatelessWidget {
  const PlaceholderPanel({super.key, required this.title});

  final String title;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text(
        '$title ページは準備中です。',
        style: Theme.of(context).textTheme.titleMedium,
      ),
    );
  }
}

class VideoGenerateForm extends StatefulWidget {
  const VideoGenerateForm({super.key});

  @override
  State<VideoGenerateForm> createState() => _VideoGenerateFormState();
}

class _VideoGenerateFormState extends State<VideoGenerateForm> {
  final _formKey = GlobalKey<FormState>();
  final _scriptController = TextEditingController();
  final _imageListController = TextEditingController();
  final _outputController = TextEditingController();
  final _apiKeyController = TextEditingController();
  final _widthController = TextEditingController(text: '1080');
  final _heightController = TextEditingController(text: '1920');
  final _fpsController = TextEditingController(text: '30');
  bool _useBgm = false;
  final _bgmController = TextEditingController();
  bool _isSubmitting = false;
  String? _jobId;
  String _statusMessage = 'Ready';

  @override
  void dispose() {
    _scriptController.dispose();
    _imageListController.dispose();
    _outputController.dispose();
    _apiKeyController.dispose();
    _widthController.dispose();
    _heightController.dispose();
    _fpsController.dispose();
    _bgmController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: ListView(
        children: [
          Text(
            '動画生成',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _scriptController,
            decoration: const InputDecoration(
              labelText: '原稿ファイルパス',
              hintText: 'dialogue_input.txt',
            ),
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _imageListController,
            decoration: const InputDecoration(
              labelText: '画像パス（カンマ区切り）',
              hintText: 'image1.png, image2.png',
            ),
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _outputController,
            decoration: const InputDecoration(
              labelText: '出力フォルダ',
              hintText: 'C:/videos/output',
            ),
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _apiKeyController,
            decoration: const InputDecoration(
              labelText: 'Gemini API キー',
            ),
            obscureText: true,
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              SizedBox(
                width: 160,
                child: TextFormField(
                  controller: _widthController,
                  decoration: const InputDecoration(labelText: '幅'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 160,
                child: TextFormField(
                  controller: _heightController,
                  decoration: const InputDecoration(labelText: '高さ'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 120,
                child: TextFormField(
                  controller: _fpsController,
                  decoration: const InputDecoration(labelText: 'FPS'),
                  keyboardType: TextInputType.number,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          SwitchListTile(
            value: _useBgm,
            title: const Text('BGM を使用する'),
            onChanged: (value) {
              setState(() {
                _useBgm = value;
              });
            },
          ),
          if (_useBgm)
            TextFormField(
              controller: _bgmController,
              decoration: const InputDecoration(labelText: 'BGM ファイルパス'),
            ),
          const SizedBox(height: 16),
          Row(
            children: [
              ElevatedButton.icon(
                onPressed: _isSubmitting ? null : _submitJob,
                icon: const Icon(Icons.play_arrow),
                label: Text(_isSubmitting ? '送信中...' : '動画を生成する'),
              ),
              const SizedBox(width: 16),
              Expanded(child: Text('状態: $_statusMessage')),
            ],
          ),
          if (_jobId != null) ...[
            const SizedBox(height: 12),
            SelectableText('Job ID: $_jobId'),
          ],
        ],
      ),
    );
  }

  Future<void> _submitJob() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isSubmitting = true;
      _statusMessage = 'Submitting...';
    });

    final imagePaths = _imageListController.text
        .split(',')
        .map((path) => path.trim())
        .where((path) => path.isNotEmpty)
        .toList();

    final payload = {
      'api_key': _apiKeyController.text,
      'script_path': _scriptController.text,
      'image_paths': imagePaths,
      'use_bgm': _useBgm,
      'bgm_path': _bgmController.text,
      'bgm_gain_db': 0.0,
      'output_dir': _outputController.text,
      'width': int.tryParse(_widthController.text) ?? 1080,
      'height': int.tryParse(_heightController.text) ?? 1920,
      'fps': int.tryParse(_fpsController.text) ?? 30,
    };

    try {
      final response = await http.post(
        Uri.parse('http://localhost:8000/video/generate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _jobId = data['job_id'] as String?;
          _statusMessage = 'Submitted';
        });
      } else {
        setState(() {
          _statusMessage = 'Error: ${response.statusCode}';
        });
      }
    } catch (error) {
      setState(() {
        _statusMessage = 'Error: $error';
      });
    } finally {
      setState(() {
        _isSubmitting = false;
      });
    }
  }
}

class LogPanel extends StatefulWidget {
  const LogPanel({super.key, required this.pageName});

  final String pageName;

  @override
  State<LogPanel> createState() => _LogPanelState();
}

class _LogPanelState extends State<LogPanel> {
  final List<String> _logs = [];
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;

  @override
  void dispose() {
    _subscription?.cancel();
    _channel?.sink.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: Text(
            '${widget.pageName} ログ',
            style: Theme.of(context).textTheme.titleMedium,
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: ElevatedButton.icon(
            onPressed: _connectWebSocket,
            icon: const Icon(Icons.wifi),
            label: const Text('WebSocket 接続（Job ID が必要）'),
          ),
        ),
        const SizedBox(height: 8),
        Expanded(
          child: Container(
            margin: const EdgeInsets.all(16),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              border: Border.all(color: Colors.grey.shade300),
              borderRadius: BorderRadius.circular(8),
            ),
            child: ListView.builder(
              itemCount: _logs.length,
              itemBuilder: (context, index) => Text(_logs[index]),
            ),
          ),
        ),
      ],
    );
  }

  void _connectWebSocket() {
    final jobId = _promptForJobId();
    if (jobId == null || jobId.isEmpty) {
      return;
    }

    _channel?.sink.close();
    _subscription?.cancel();

    final channel = WebSocketChannel.connect(
      Uri.parse('ws://localhost:8000/ws/jobs/$jobId'),
    );

    setState(() {
      _logs.add('Connecting to $jobId ...');
      _channel = channel;
    });

    _subscription = channel.stream.listen(
      (event) {
        final message = event is String ? event : jsonEncode(event);
        setState(() {
          _logs.add(message);
        });
      },
      onError: (error) {
        setState(() {
          _logs.add('WebSocket error: $error');
        });
      },
      onDone: () {
        setState(() {
          _logs.add('WebSocket closed');
        });
      },
    );
  }

  String? _promptForJobId() {
    final controller = TextEditingController();
    String? result;

    showDialog<void>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Job ID を入力'),
          content: TextField(
            controller: controller,
            decoration: const InputDecoration(hintText: 'UUID'),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: const Text('キャンセル'),
            ),
            ElevatedButton(
              onPressed: () {
                result = controller.text;
                Navigator.of(context).pop();
              },
              child: const Text('接続'),
            ),
          ],
        );
      },
    );

    return result;
  }
}
