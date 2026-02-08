import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';

void main() {
  runApp(const MovieMakerApp());
}

String _defaultApiBaseUrl() {
  if (Platform.isAndroid) {
    return 'http://10.0.2.2:8000';
  }
  return 'http://localhost:8000';
}

String _joinPaths(String basePath, String relativePath) {
  if (basePath.isEmpty || basePath == '/') {
    return relativePath;
  }
  final normalizedBase = basePath.endsWith('/')
      ? basePath.substring(0, basePath.length - 1)
      : basePath;
  final normalizedRelative =
      relativePath.startsWith('/') ? relativePath : '/$relativePath';
  return '$normalizedBase$normalizedRelative';
}

class ApiConfig {
  static final ValueNotifier<String> baseUrl =
      ValueNotifier<String>(_defaultApiBaseUrl());

  static Uri httpUri(String path) {
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    final baseUri = Uri.parse(baseUrl.value.trim());
    final combinedPath = _joinPaths(baseUri.path, normalizedPath);
    return baseUri.replace(path: combinedPath);
  }

  static Uri wsUri(String path) {
    final base = httpUri(path);
    final scheme = base.scheme == 'https' ? 'wss' : 'ws';
    return base.replace(scheme: scheme);
  }
}

class MovieMakerApp extends StatelessWidget {
  const MovieMakerApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF5B7CFA),
      brightness: Brightness.light,
    );
    return MaterialApp(
      title: 'News Short Generator Studio',
      theme: ThemeData(
        colorScheme: colorScheme,
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF5F6FA),
        cardTheme: CardThemeData(
          elevation: 2,
          shadowColor: Colors.black.withOpacity(0.08),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: BorderSide(color: colorScheme.primary, width: 1.4),
          ),
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          ),
        ),
        navigationRailTheme: NavigationRailThemeData(
          backgroundColor: Colors.white,
          indicatorColor: colorScheme.primary.withOpacity(0.12),
          selectedIconTheme: IconThemeData(color: colorScheme.primary),
          selectedLabelTextStyle: TextStyle(
            color: colorScheme.primary,
            fontWeight: FontWeight.w600,
          ),
          unselectedLabelTextStyle: const TextStyle(color: Color(0xFF768098)),
        ),
        textTheme: const TextTheme(
          headlineSmall: TextStyle(fontWeight: FontWeight.w700),
          titleMedium: TextStyle(fontWeight: FontWeight.w600),
        ),
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
  Process? _apiServerProcess;

  @override
  void initState() {
    super.initState();
    _startApiServer();
  }

  @override
  void dispose() {
    _apiServerProcess?.kill();
    super.dispose();
  }

  Future<void> _startApiServer() async {
    if (_apiServerProcess != null) {
      return;
    }
    if (!(Platform.isLinux || Platform.isMacOS || Platform.isWindows)) {
      return;
    }
    final pythonExecutable = Platform.isWindows ? 'python' : 'python3';
    try {
      _apiServerProcess = await Process.start(
        pythonExecutable,
        [
          '-m',
          'uvicorn',
          'backend.api_server:app',
          '--host',
          '0.0.0.0',
          '--port',
          '8000',
        ],
        workingDirectory: '..',
        runInShell: true,
      );
    } catch (_) {
      _apiServerProcess = null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          Container(
            width: 220,
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 18,
                  offset: const Offset(2, 0),
                ),
              ],
            ),
            child: NavigationRail(
              selectedIndex: _selectedIndex,
              onDestinationSelected: (index) {
                setState(() {
                  _selectedIndex = index;
                });
              },
              labelType: NavigationRailLabelType.all,
              leading: Padding(
                padding: const EdgeInsets.only(top: 24, bottom: 16),
                child: Column(
                  children: [
                    CircleAvatar(
                      radius: 24,
                      backgroundColor: Theme.of(context).colorScheme.primary.withOpacity(0.15),
                      child: Icon(
                        Icons.movie_creation,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'Studio',
                      style: TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
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
          ),
          const VerticalDivider(width: 1),
          Expanded(
            flex: 3,
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: _buildCenterPanel(),
                ),
              ),
            ),
          ),
          const VerticalDivider(width: 1),
          Expanded(
            flex: 2,
            child: Card(
              margin: const EdgeInsets.all(20),
              child: LogPanel(pageName: _pages[_selectedIndex]),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCenterPanel() {
    switch (_selectedIndex) {
      case 0:
        return const VideoGenerateForm();
      case 1:
        return const ScriptGenerateForm();
      case 2:
        return const TitleGenerateForm();
      case 3:
        return const MaterialsGenerateForm();
      case 4:
        return const VideoEditForm();
      case 5:
        return const DetailedEditForm();
      case 6:
        return const SettingsForm();
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

class ScriptGenerateForm extends StatefulWidget {
  const ScriptGenerateForm({super.key});

  @override
  State<ScriptGenerateForm> createState() => _ScriptGenerateFormState();
}

class _ScriptGenerateFormState extends State<ScriptGenerateForm> {
  final _formKey = GlobalKey<FormState>();
  final _promptController = TextEditingController();
  final _outputController = TextEditingController(text: 'dialogue_input.txt');
  final _maxTokensController = TextEditingController(text: '20000');
  String _provider = 'Gemini';
  String _model = 'gemini-2.0-flash';

  @override
  void dispose() {
    _promptController.dispose();
    _outputController.dispose();
    _maxTokensController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: ListView(
        children: [
          Text(
            '台本生成',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<String>(
            value: _provider,
            decoration: const InputDecoration(labelText: '生成AI'),
            items: const [
              DropdownMenuItem(value: 'Gemini', child: Text('Gemini')),
              DropdownMenuItem(value: 'ChatGPT', child: Text('ChatGPT')),
              DropdownMenuItem(value: 'ClaudeCode', child: Text('ClaudeCode')),
            ],
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _provider = value;
              });
            },
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _model,
            decoration: const InputDecoration(labelText: 'モデル'),
            items: const [
              DropdownMenuItem(value: 'gemini-2.0-flash', child: Text('gemini-2.0-flash')),
              DropdownMenuItem(value: 'gpt-4.1-mini', child: Text('gpt-4.1-mini')),
              DropdownMenuItem(value: 'claude-opus-4-5-20251101', child: Text('claude-opus-4-5-20251101')),
            ],
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _model = value;
              });
            },
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _maxTokensController,
            decoration: const InputDecoration(labelText: 'Claude max_tokens'),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _promptController,
            maxLines: 8,
            decoration: const InputDecoration(labelText: 'プロンプト'),
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _outputController,
            decoration: InputDecoration(
              labelText: '保存ファイル',
              suffixIcon: IconButton(
                icon: const Icon(Icons.file_present),
                onPressed: () => _selectSavePath(
                  _outputController,
                  const XTypeGroup(label: 'Text', extensions: ['txt']),
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              ElevatedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.play_arrow),
                label: const Text('台本生成'),
              ),
              OutlinedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.copy),
                label: const Text('コピー'),
              ),
              OutlinedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.save),
                label: const Text('保存'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class TitleGenerateForm extends StatefulWidget {
  const TitleGenerateForm({super.key});

  @override
  State<TitleGenerateForm> createState() => _TitleGenerateFormState();
}

class _TitleGenerateFormState extends State<TitleGenerateForm> {
  final _formKey = GlobalKey<FormState>();
  final _scriptPathController = TextEditingController();
  final _countController = TextEditingController(text: '5');
  final _instructionsController = TextEditingController();
  String _provider = 'Gemini';
  String _model = 'gemini-2.0-flash';

  @override
  void dispose() {
    _scriptPathController.dispose();
    _countController.dispose();
    _instructionsController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: ListView(
        children: [
          Text(
            '動画タイトル・説明作成',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _scriptPathController,
            decoration: InputDecoration(
              labelText: '台本ファイル（SRT/TXT）',
              suffixIcon: IconButton(
                icon: const Icon(Icons.folder_open),
                onPressed: () => _selectFile(
                  _scriptPathController,
                  const XTypeGroup(label: 'Script', extensions: ['srt', 'txt']),
                ),
              ),
            ),
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _provider,
            decoration: const InputDecoration(labelText: '生成AI'),
            items: const [
              DropdownMenuItem(value: 'Gemini', child: Text('Gemini')),
              DropdownMenuItem(value: 'ChatGPT', child: Text('ChatGPT')),
              DropdownMenuItem(value: 'ClaudeCode', child: Text('ClaudeCode')),
            ],
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _provider = value;
              });
            },
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _model,
            decoration: const InputDecoration(labelText: 'モデル'),
            items: const [
              DropdownMenuItem(value: 'gemini-2.0-flash', child: Text('gemini-2.0-flash')),
              DropdownMenuItem(value: 'gpt-4.1-mini', child: Text('gpt-4.1-mini')),
              DropdownMenuItem(value: 'claude-opus-4-5-20251101', child: Text('claude-opus-4-5-20251101')),
            ],
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _model = value;
              });
            },
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _countController,
            decoration: const InputDecoration(labelText: 'タイトル案の数'),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _instructionsController,
            maxLines: 4,
            decoration: const InputDecoration(labelText: '追加指示（任意）'),
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 12,
            children: [
              ElevatedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.play_arrow),
                label: const Text('タイトル生成'),
              ),
              OutlinedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.copy),
                label: const Text('コピー'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class MaterialsGenerateForm extends StatefulWidget {
  const MaterialsGenerateForm({super.key});

  @override
  State<MaterialsGenerateForm> createState() => _MaterialsGenerateFormState();
}

class _MaterialsGenerateFormState extends State<MaterialsGenerateForm> {
  final _formKey = GlobalKey<FormState>();
  final _modelController = TextEditingController(text: 'gemini-2.0-flash');
  final _promptController = TextEditingController();
  final _outputController = TextEditingController();

  @override
  void dispose() {
    _modelController.dispose();
    _promptController.dispose();
    _outputController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: ListView(
        children: [
          Text(
            '資料作成',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _modelController,
            decoration: const InputDecoration(labelText: 'モデル名'),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _promptController,
            maxLines: 6,
            decoration: const InputDecoration(labelText: '画像生成プロンプト'),
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _outputController,
            decoration: InputDecoration(
              labelText: '保存フォルダ',
              suffixIcon: IconButton(
                icon: const Icon(Icons.folder),
                onPressed: () => _selectDirectory(_outputController),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 12,
            children: [
              ElevatedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.image),
                label: const Text('画像生成'),
              ),
              OutlinedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.save),
                label: const Text('保存'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class VideoEditForm extends StatefulWidget {
  const VideoEditForm({super.key});

  @override
  State<VideoEditForm> createState() => _VideoEditFormState();
}

class _VideoEditFormState extends State<VideoEditForm> {
  final _formKey = GlobalKey<FormState>();
  final _inputVideoController = TextEditingController();
  final _outputVideoController = TextEditingController();
  final _overlayImageController = TextEditingController();
  final _startController = TextEditingController(text: '00:00');
  final _endController = TextEditingController(text: '00:10');
  final _xController = TextEditingController(text: '100');
  final _yController = TextEditingController(text: '200');
  final _widthController = TextEditingController(text: '0');
  final _heightController = TextEditingController(text: '0');
  final _opacityController = TextEditingController(text: '1.0');
  final List<String> _overlays = [];

  @override
  void dispose() {
    _inputVideoController.dispose();
    _outputVideoController.dispose();
    _overlayImageController.dispose();
    _startController.dispose();
    _endController.dispose();
    _xController.dispose();
    _yController.dispose();
    _widthController.dispose();
    _heightController.dispose();
    _opacityController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: ListView(
        children: [
          Text(
            '動画編集（簡易オーバーレイ）',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _inputVideoController,
            decoration: InputDecoration(
              labelText: '入力動画（MP4）',
              suffixIcon: IconButton(
                icon: const Icon(Icons.video_file),
                onPressed: () => _selectFile(
                  _inputVideoController,
                  const XTypeGroup(label: 'Video', extensions: ['mp4', 'mov', 'mkv']),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _outputVideoController,
            decoration: InputDecoration(
              labelText: '出力動画',
              suffixIcon: IconButton(
                icon: const Icon(Icons.save_alt),
                onPressed: () => _selectSavePath(
                  _outputVideoController,
                  const XTypeGroup(label: 'Video', extensions: ['mp4', 'mov', 'mkv']),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _overlayImageController,
            decoration: InputDecoration(
              labelText: 'オーバーレイ画像',
              suffixIcon: IconButton(
                icon: const Icon(Icons.image_outlined),
                onPressed: () => _selectFile(
                  _overlayImageController,
                  const XTypeGroup(label: 'Image', extensions: ['png', 'jpg', 'jpeg', 'webp']),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              SizedBox(
                width: 120,
                child: TextFormField(
                  controller: _startController,
                  decoration: const InputDecoration(labelText: '開始'),
                ),
              ),
              SizedBox(
                width: 120,
                child: TextFormField(
                  controller: _endController,
                  decoration: const InputDecoration(labelText: '終了'),
                ),
              ),
              SizedBox(
                width: 100,
                child: TextFormField(
                  controller: _xController,
                  decoration: const InputDecoration(labelText: 'X'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 100,
                child: TextFormField(
                  controller: _yController,
                  decoration: const InputDecoration(labelText: 'Y'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 100,
                child: TextFormField(
                  controller: _widthController,
                  decoration: const InputDecoration(labelText: 'W'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 100,
                child: TextFormField(
                  controller: _heightController,
                  decoration: const InputDecoration(labelText: 'H'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 120,
                child: TextFormField(
                  controller: _opacityController,
                  decoration: const InputDecoration(labelText: '不透明度'),
                  keyboardType: TextInputType.number,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              ElevatedButton.icon(
                onPressed: () {
                  setState(() {
                    _overlays.add(
                      '${_overlayImageController.text} '
                      '${_startController.text}〜${_endController.text}',
                    );
                  });
                },
                icon: const Icon(Icons.add),
                label: const Text('オーバーレイ追加'),
              ),
              OutlinedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.upload_file),
                label: const Text('JSONインポート'),
              ),
              OutlinedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.search),
                label: const Text('SRT検索'),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text('オーバーレイ一覧', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          ..._overlays.map((overlay) => ListTile(title: Text(overlay))),
          const SizedBox(height: 12),
          ElevatedButton.icon(
            onPressed: () {},
            icon: const Icon(Icons.movie),
            label: const Text('書き出し'),
          ),
        ],
      ),
    );
  }
}

class DetailedEditForm extends StatefulWidget {
  const DetailedEditForm({super.key});

  @override
  State<DetailedEditForm> createState() => _DetailedEditFormState();
}

class _DetailedEditFormState extends State<DetailedEditForm> {
  final _projectNameController = TextEditingController(text: 'project.mmproj');
  final _resolutionController = TextEditingController(text: '1080x1920');
  final _fpsController = TextEditingController(text: '30');
  bool _audioEnabled = true;

  @override
  void dispose() {
    _projectNameController.dispose();
    _resolutionController.dispose();
    _fpsController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      children: [
        Text(
          '詳細動画編集',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 16),
        TextFormField(
          controller: _projectNameController,
          decoration: InputDecoration(
            labelText: 'プロジェクトファイル',
            suffixIcon: IconButton(
              icon: const Icon(Icons.folder_open),
              onPressed: () => _selectFile(
                _projectNameController,
                const XTypeGroup(label: 'Project', extensions: ['mmproj']),
              ),
            ),
          ),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            ElevatedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.create_new_folder),
              label: const Text('新規'),
            ),
            OutlinedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.folder_open),
              label: const Text('読み込み'),
            ),
            OutlinedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.save),
              label: const Text('保存'),
            ),
          ],
        ),
        const SizedBox(height: 16),
        Text('素材管理', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        Wrap(
          spacing: 12,
          children: [
            OutlinedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.add_photo_alternate),
              label: const Text('素材追加'),
            ),
            OutlinedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.delete),
              label: const Text('削除'),
            ),
            OutlinedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.link),
              label: const Text('再リンク'),
            ),
          ],
        ),
        const SizedBox(height: 16),
        Text('タイムライン', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        Wrap(
          spacing: 12,
          children: [
            OutlinedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.video_library),
              label: const Text('メイン動画読込'),
            ),
            OutlinedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.cut),
              label: const Text('クリップ分割'),
            ),
            OutlinedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.swap_horiz),
              label: const Text('順序変更'),
            ),
          ],
        ),
        const SizedBox(height: 16),
        Text('音声設定', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        SwitchListTile(
          value: _audioEnabled,
          onChanged: (value) {
            setState(() {
              _audioEnabled = value;
            });
          },
          title: const Text('動画音声 ON/OFF'),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            SizedBox(
              width: 160,
              child: TextFormField(
                controller: _resolutionController,
                decoration: const InputDecoration(labelText: '解像度'),
              ),
            ),
            SizedBox(
              width: 120,
              child: TextFormField(
                controller: _fpsController,
                decoration: const InputDecoration(labelText: 'FPS'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        ElevatedButton.icon(
          onPressed: () {},
          icon: const Icon(Icons.movie_creation),
          label: const Text('書き出し'),
        ),
      ],
    );
  }
}

class SettingsForm extends StatefulWidget {
  const SettingsForm({super.key});

  @override
  State<SettingsForm> createState() => _SettingsFormState();
}

class _SettingsFormState extends State<SettingsForm> {
  final _backendController = TextEditingController();
  final _geminiController = TextEditingController();
  final _openAiController = TextEditingController();
  final _claudeController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _backendController.text = ApiConfig.baseUrl.value;
    _backendController.addListener(() {
      ApiConfig.baseUrl.value = _backendController.text.trim();
    });
  }

  @override
  void dispose() {
    _backendController.dispose();
    _geminiController.dispose();
    _openAiController.dispose();
    _claudeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      children: [
        Text(
          '設定',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 16),
        TextFormField(
          controller: _backendController,
          decoration: const InputDecoration(
            labelText: 'バックエンドURL',
            helperText: 'Android エミュレータの場合は http://10.0.2.2:8000',
          ),
        ),
        const SizedBox(height: 12),
        TextFormField(
          controller: _geminiController,
          decoration: const InputDecoration(labelText: 'Gemini API キー'),
          obscureText: true,
        ),
        const SizedBox(height: 12),
        TextFormField(
          controller: _openAiController,
          decoration: const InputDecoration(labelText: 'ChatGPT API キー'),
          obscureText: true,
        ),
        const SizedBox(height: 12),
        TextFormField(
          controller: _claudeController,
          decoration: const InputDecoration(labelText: 'ClaudeCode API キー'),
          obscureText: true,
        ),
        const SizedBox(height: 16),
        ElevatedButton.icon(
          onPressed: () {},
          icon: const Icon(Icons.save),
          label: const Text('保存'),
        ),
      ],
    );
  }
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
            decoration: InputDecoration(
              labelText: '原稿ファイルパス',
              hintText: 'dialogue_input.txt',
              suffixIcon: IconButton(
                icon: const Icon(Icons.folder_open),
                onPressed: () => _selectFile(
                  _scriptController,
                  const XTypeGroup(label: 'Script', extensions: ['txt', 'srt']),
                ),
              ),
            ),
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _imageListController,
            decoration: InputDecoration(
              labelText: '画像パス（カンマ区切り）',
              hintText: 'image1.png, image2.png',
              suffixIcon: IconButton(
                icon: const Icon(Icons.collections),
                onPressed: () => _selectFiles(
                  _imageListController,
                  const XTypeGroup(label: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp']),
                ),
              ),
            ),
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _outputController,
            decoration: InputDecoration(
              labelText: '出力フォルダ',
              hintText: 'C:/videos/output',
              suffixIcon: IconButton(
                icon: const Icon(Icons.folder),
                onPressed: () => _selectDirectory(_outputController),
              ),
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
              decoration: InputDecoration(
                labelText: 'BGM ファイルパス',
                suffixIcon: IconButton(
                  icon: const Icon(Icons.library_music),
                  onPressed: () => _selectFile(
                    _bgmController,
                    const XTypeGroup(label: 'Audio', extensions: ['mp3', 'wav', 'aac']),
                  ),
                ),
              ),
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
        ApiConfig.httpUri('/video/generate'),
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
      ApiConfig.wsUri('/ws/jobs/$jobId'),
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

Future<void> _selectFile(TextEditingController controller, XTypeGroup typeGroup) async {
  final file = await openFile(acceptedTypeGroups: [typeGroup]);
  if (file == null) return;
  controller.text = file.path;
}

Future<void> _selectFiles(TextEditingController controller, XTypeGroup typeGroup) async {
  final files = await openFiles(acceptedTypeGroups: [typeGroup]);
  if (files.isEmpty) return;
  controller.text = files.map((file) => file.path).join(', ');
}

Future<void> _selectDirectory(TextEditingController controller) async {
  final path = await getDirectoryPath();
  if (path == null) return;
  controller.text = path;
}

Future<void> _selectSavePath(TextEditingController controller, XTypeGroup typeGroup) async {
  final location = await getSaveLocation(acceptedTypeGroups: [typeGroup]);
  if (location == null) return;
  controller.text = location.path;
}
