import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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

class ApiKeys {
  static final ValueNotifier<String> gemini = ValueNotifier<String>('');
  static final ValueNotifier<String> openAi = ValueNotifier<String>('');
  static final ValueNotifier<String> claude = ValueNotifier<String>('');
}

class MovieMakerApp extends StatelessWidget {
  const MovieMakerApp({super.key});

  @override
  Widget build(BuildContext context) {
    final baseScheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF6C5CE7),
      brightness: Brightness.light,
    );
    final colorScheme = baseScheme.copyWith(
      primary: const Color(0xFF6C5CE7),
      secondary: const Color(0xFFFF6B6B),
      tertiary: const Color(0xFF00D2D3),
      surface: const Color(0xFFFFFFFF),
      background: const Color(0xFFF6F4FF),
    );
    return MaterialApp(
      title: 'News Short Generator Studio',
      theme: ThemeData(
        colorScheme: colorScheme,
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF6F4FF),
        cardTheme: CardThemeData(
          elevation: 3,
          shadowColor: colorScheme.primary.withOpacity(0.2),
          surfaceTintColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide(color: colorScheme.primary, width: 1.6),
          ),
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
            backgroundColor: colorScheme.primary,
            foregroundColor: Colors.white,
            elevation: 2,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            side: BorderSide(color: colorScheme.primary.withOpacity(0.6)),
          ),
        ),
        navigationRailTheme: NavigationRailThemeData(
          backgroundColor: Colors.white.withOpacity(0.92),
          indicatorColor: colorScheme.primary.withOpacity(0.16),
          selectedIconTheme: IconThemeData(color: colorScheme.primary),
          selectedLabelTextStyle: TextStyle(
            color: colorScheme.primary,
            fontWeight: FontWeight.w600,
          ),
          unselectedLabelTextStyle: const TextStyle(color: Color(0xFF768098)),
          unselectedIconTheme: const IconThemeData(color: Color(0xFF7C8BA1)),
        ),
        textTheme: const TextTheme(
          headlineSmall: TextStyle(fontWeight: FontWeight.w700),
          titleMedium: TextStyle(fontWeight: FontWeight.w600),
        ),
        snackBarTheme: SnackBarThemeData(
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
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
    '台本生成',
    '動画生成',
    '動画タイトル・説明',
    'サムネイル作成',
    'ポンチ絵作成',
    '動画編集',
    '詳細動画編集',
    '設定',
    'About',
  ];
  int _selectedIndex = 0;
  Process? _apiServerProcess;
  bool _apiServerStarting = false;
  bool _apiServerReady = false;
  String? _apiServerStatus;
  String? _apiServerErrorDetails;
  String? _apiServerLaunchCommand;
  Directory? _apiServerRoot;
  final ValueNotifier<String?> _latestJobId = ValueNotifier<String?>(null);

  @override
  void initState() {
    super.initState();
    _ensureApiServerRunning();
  }

  @override
  void dispose() {
    _apiServerProcess?.kill();
    _latestJobId.dispose();
    super.dispose();
  }

  Future<void> _ensureApiServerRunning() async {
    if (_apiServerStarting) {
      return;
    }
    _apiServerStarting = true;

    if (await _isApiHealthy()) {
      _setApiServerStatus(ready: true, message: 'API サーバー稼働中');
      _clearApiServerError();
      _apiServerStarting = false;
      return;
    }

    if (!(Platform.isLinux || Platform.isMacOS || Platform.isWindows)) {
      _setApiServerStatus(
        ready: false,
        message: 'API サーバーはデスクトップ環境でのみ自動起動します。',
      );
      _clearApiServerError();
      _apiServerStarting = false;
      return;
    }

    await _startApiServerProcess();

    final started = await _waitForApiHealth();
    if (started) {
      _setApiServerStatus(ready: true, message: 'API サーバー起動完了');
      _clearApiServerError();
    } else {
      _setApiServerStatus(
        ready: false,
        message: 'API サーバーの起動に失敗しました。手動で起動してください。',
      );
      _showApiServerSnackBar(
        'API サーバーの起動に失敗しました。ターミナルで '
        '${_apiServerLaunchCommand ?? 'python -m uvicorn backend.api_server:app --host 0.0.0.0 --port 8000'} '
        'を実行してください。',
      );
    }
    _apiServerStarting = false;
  }

  Directory? _findApiServerRoot() {
    final visited = <String>{};
    final candidates = <Directory>[];
    var current = Directory.current;
    candidates.add(current);
    final flutterAppSuffix =
        '${Platform.pathSeparator}flutter_app';
    if (current.path.endsWith(flutterAppSuffix)) {
      candidates.add(current.parent);
    }
    while (true) {
      final parent = current.parent;
      if (parent.path == current.path) {
        break;
      }
      current = parent;
      candidates.add(current);
    }

    for (final candidate in candidates) {
      if (!visited.add(candidate.path)) {
        continue;
      }
      final path = _joinFilePath(
        [candidate.path, 'backend', 'api_server.py'],
      );
      if (File(path).existsSync()) {
        return candidate;
      }
    }
    return null;
  }

  String _joinFilePath(List<String> segments) {
    if (segments.isEmpty) {
      return '';
    }
    var current = segments.first;
    for (var index = 1; index < segments.length; index += 1) {
      final part = segments[index];
      if (current.endsWith(Platform.pathSeparator)) {
        current = '$current$part';
      } else {
        current = '$current${Platform.pathSeparator}$part';
      }
    }
    return current;
  }

  Future<void> _startApiServerProcess() async {
    if (_apiServerProcess != null) {
      return;
    }
    _apiServerRoot ??= _findApiServerRoot();
    if (_apiServerRoot == null) {
      _setApiServerStatus(
        ready: false,
        message: 'API サーバーのパスが見つかりません。Flutter プロジェクトをルートで起動してください。',
      );
      _setApiServerError('backend/api_server.py が見つかりませんでした。');
      _showApiServerSnackBar('API サーバーの配置場所を確認してください。');
      return;
    }

    final pythonExecutables = Platform.isWindows
        ? ['python']
        : ['python3', 'python'];

    for (final pythonExecutable in pythonExecutables) {
      try {
        _apiServerLaunchCommand =
            '$pythonExecutable -m uvicorn backend.api_server:app --host 0.0.0.0 --port 8000';
        final process = await Process.start(
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
          workingDirectory: _apiServerRoot!.path,
          environment: {
            ...Platform.environment,
            'PYTHONPATH': _apiServerRoot!.path,
          },
          runInShell: true,
        );
        _apiServerProcess = process;
        _clearApiServerError();
        process.stderr
            .transform(const Utf8Decoder(allowMalformed: true))
            .transform(const LineSplitter())
            .listen(_appendApiServerError);
        process.stdout
            .transform(const Utf8Decoder(allowMalformed: true))
            .transform(const LineSplitter())
            .listen((line) {
          if (line.contains('ERROR') || line.contains('Error')) {
            _appendApiServerError(line);
          }
        });
        process.exitCode.then((code) {
          if (!mounted) {
            return;
          }
          if (code != 0) {
            _setApiServerStatus(
              ready: false,
              message: 'API サーバーが終了しました (exit code: $code)。',
            );
            if (_apiServerErrorDetails == null) {
              _setApiServerError('プロセスが終了しました (exit code: $code)。');
            }
            _showApiServerSnackBar('API サーバーが停止しました。');
          }
          _apiServerProcess = null;
        });
        return;
      } catch (error) {
        _apiServerProcess = null;
        _setApiServerError('起動コマンドの実行に失敗しました: $error');
      }
    }
  }

  Future<bool> _isApiHealthy() async {
    try {
      final response = await http
          .get(ApiConfig.httpUri('/health'))
          .timeout(const Duration(seconds: 3));
      return response.statusCode >= 200 && response.statusCode < 300;
    } catch (_) {
      return false;
    }
  }

  Future<bool> _waitForApiHealth() async {
    for (var attempt = 0; attempt < 20; attempt += 1) {
      if (await _isApiHealthy()) {
        return true;
      }
      await Future<void>.delayed(const Duration(milliseconds: 500));
    }
    return false;
  }

  void _setApiServerStatus({required bool ready, required String message}) {
    if (!mounted) {
      return;
    }
    setState(() {
      _apiServerReady = ready;
      _apiServerStatus = message;
    });
  }

  void _showApiServerSnackBar(String message) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  void _appendApiServerError(String line) {
    if (!mounted) {
      return;
    }
    if (line.trim().isEmpty) {
      return;
    }
    setState(() {
      final existing = _apiServerErrorDetails;
      if (existing == null || existing.isEmpty) {
        _apiServerErrorDetails = line;
      } else {
        final combined = '$existing\n$line';
        _apiServerErrorDetails = combined.length > 1200
            ? combined.substring(combined.length - 1200)
            : combined;
      }
    });
  }

  void _setApiServerError(String message) {
    if (!mounted) {
      return;
    }
    setState(() {
      _apiServerErrorDetails = message;
    });
  }

  void _clearApiServerError() {
    if (!mounted) {
      return;
    }
    setState(() {
      _apiServerErrorDetails = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFF7F2FF), Color(0xFFEEF4FF)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Column(
          children: [
            if (_apiServerStatus != null)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: _apiServerReady
                        ? [const Color(0xFFD8F5E5), const Color(0xFFE9FFF5)]
                        : [const Color(0xFFFFE0E0), const Color(0xFFFFF1F1)],
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      _apiServerReady ? Icons.check_circle : Icons.error_outline,
                      color:
                          _apiServerReady ? Colors.green.shade700 : Colors.red.shade700,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _apiServerStatus ?? '',
                            style: TextStyle(
                              color: _apiServerReady
                                  ? Colors.green.shade700
                                  : Colors.red.shade700,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          if (!_apiServerReady && _apiServerErrorDetails != null)
                            Padding(
                              padding: const EdgeInsets.only(top: 4),
                              child: Text(
                                _apiServerErrorDetails ?? '',
                                style: TextStyle(
                                  color: Colors.red.shade700,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                          if (!_apiServerReady &&
                              _apiServerLaunchCommand != null)
                            Padding(
                              padding: const EdgeInsets.only(top: 2),
                              child: Text(
                                '起動コマンド: ${_apiServerLaunchCommand!}',
                                style: TextStyle(
                                  color: Colors.red.shade700,
                                  fontSize: 11,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                    if (!_apiServerReady && !_apiServerStarting)
                      TextButton(
                        onPressed: _ensureApiServerRunning,
                        child: const Text('再試行'),
                      ),
                  ],
                ),
              ),
            Expanded(
              child: Row(
                children: [
                  Container(
                    width: 220,
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFFFFFFFF), Color(0xFFF3F6FF)],
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Theme.of(context).colorScheme.primary.withOpacity(0.12),
                          blurRadius: 20,
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
                              backgroundColor: Theme.of(context)
                                  .colorScheme
                                  .primary
                                  .withOpacity(0.15),
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
                            const SizedBox(height: 12),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 10,
                                vertical: 6,
                              ),
                              decoration: BoxDecoration(
                                color: _apiServerReady
                                    ? const Color(0xFFE7F7EF)
                                    : _apiServerStarting
                                        ? const Color(0xFFFFF4E1)
                                        : const Color(0xFFFFECEC),
                                borderRadius: BorderRadius.circular(12),
                                border: Border.all(
                                  color: _apiServerReady
                                      ? const Color(0xFFBEE8D0)
                                      : _apiServerStarting
                                          ? const Color(0xFFFFD29A)
                                          : const Color(0xFFF7B8B8),
                                ),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(
                                    _apiServerReady
                                        ? Icons.cloud_done
                                        : _apiServerStarting
                                            ? Icons.cloud_sync
                                            : Icons.cloud_off,
                                    size: 16,
                                    color: _apiServerReady
                                        ? Colors.green.shade700
                                        : _apiServerStarting
                                            ? Colors.orange.shade700
                                            : Colors.red.shade700,
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    _apiServerReady
                                        ? 'API 稼働中'
                                        : _apiServerStarting
                                            ? 'API 起動中'
                                            : 'API 停止中',
                                    style: TextStyle(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                      color: _apiServerReady
                                          ? Colors.green.shade800
                                          : _apiServerStarting
                                              ? Colors.orange.shade700
                                              : Colors.red.shade700,
                                    ),
                                  ),
                                ],
                              ),
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
                      child: LogPanel(
                        pageName: _pages[_selectedIndex],
                        latestJobId: _latestJobId,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCenterPanel() {
    switch (_selectedIndex) {
      case 0:
        return const ScriptGenerateForm();
      case 1:
        return VideoGenerateForm(
          onJobSubmitted: (jobId) {
            _latestJobId.value = jobId;
          },
        );
      case 2:
        return const TitleGenerateForm();
      case 3:
        return const MaterialsGenerateForm();
      case 4:
        return const PonchiGenerateForm();
      case 5:
        return const VideoEditForm();
      case 6:
        return const DetailedEditForm();
      case 7:
        return const SettingsForm();
      case 8:
        return const AboutPanel();
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
  const VideoGenerateForm({super.key, this.onJobSubmitted});

  final ValueChanged<String?>? onJobSubmitted;

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
  final _outputTextController = TextEditingController();
  String _provider = 'Gemini';
  String _geminiModel = 'gemini-2.0-flash';
  String _chatGptModel = 'gpt-4.1-mini';
  String _claudeModel = 'claude-opus-4-5-20251101';
  String _template = '（テンプレなし）';
  bool _isSubmitting = false;
  final List<String> _templates = [
    '（テンプレなし）',
    'ニュース原稿',
    '要約',
    'YouTube Shorts',
  ];

  @override
  void dispose() {
    _promptController.dispose();
    _outputController.dispose();
    _maxTokensController.dispose();
    _outputTextController.dispose();
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
          if (_provider == 'Gemini')
            DropdownButtonFormField<String>(
              value: _geminiModel,
              decoration: const InputDecoration(labelText: 'Gemini モデル'),
              items: const [
                DropdownMenuItem(value: 'gemini-2.0-flash', child: Text('gemini-2.0-flash')),
                DropdownMenuItem(value: 'gemini-1.5-flash', child: Text('gemini-1.5-flash')),
                DropdownMenuItem(value: 'gemini-1.5-pro', child: Text('gemini-1.5-pro')),
              ],
              onChanged: (value) {
                if (value == null) return;
                setState(() {
                  _geminiModel = value;
                });
              },
            ),
          if (_provider == 'ChatGPT')
            DropdownButtonFormField<String>(
              value: _chatGptModel,
              decoration: const InputDecoration(labelText: 'ChatGPT モデル'),
              items: const [
                DropdownMenuItem(value: 'gpt-4.1-mini', child: Text('gpt-4.1-mini')),
                DropdownMenuItem(value: 'gpt-4.1', child: Text('gpt-4.1')),
                DropdownMenuItem(value: 'gpt-4o-mini', child: Text('gpt-4o-mini')),
                DropdownMenuItem(value: 'gpt-4o', child: Text('gpt-4o')),
              ],
              onChanged: (value) {
                if (value == null) return;
                setState(() {
                  _chatGptModel = value;
                });
              },
            ),
          if (_provider == 'ClaudeCode')
            DropdownButtonFormField<String>(
              value: _claudeModel,
              decoration: const InputDecoration(labelText: 'ClaudeCode モデル'),
              items: const [
                DropdownMenuItem(
                  value: 'claude-opus-4-5-20251101',
                  child: Text('claude-opus-4-5-20251101'),
                ),
              ],
              onChanged: (value) {
                if (value == null) return;
                setState(() {
                  _claudeModel = value;
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
          DropdownButtonFormField<String>(
            value: _template,
            decoration: const InputDecoration(labelText: 'プロンプトテンプレ'),
            items: _templates
                .map((value) => DropdownMenuItem(value: value, child: Text(value)))
                .toList(),
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _template = value;
              });
            },
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
                onPressed: _isSubmitting ? null : _submitScriptGenerate,
                icon: const Icon(Icons.play_arrow),
                label: Text(_isSubmitting ? '送信中...' : '$_provider で台本生成'),
              ),
              OutlinedButton.icon(
                onPressed: _copyOutputText,
                icon: const Icon(Icons.copy),
                label: const Text('コピー'),
              ),
              OutlinedButton.icon(
                onPressed: _saveOutputToFile,
                icon: const Icon(Icons.save),
                label: const Text('保存'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text('生成結果', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          TextFormField(
            controller: _outputTextController,
            maxLines: 10,
            decoration: const InputDecoration(
              hintText: '生成結果がここに表示されます。',
            ),
          ),
        ],
      ),
    );
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  String _resolveApiKey() {
    switch (_provider) {
      case 'ChatGPT':
        return ApiKeys.openAi.value;
      case 'ClaudeCode':
        return ApiKeys.claude.value;
      default:
        return ApiKeys.gemini.value;
    }
  }

  String _resolveModel() {
    switch (_provider) {
      case 'ChatGPT':
        return _chatGptModel;
      case 'ClaudeCode':
        return _claudeModel;
      default:
        return _geminiModel;
    }
  }

  Future<void> _submitScriptGenerate() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    final apiKey = _resolveApiKey();
    if (apiKey.isEmpty) {
      _showSnackBar('APIキーが未設定です。設定タブで入力してください。');
      return;
    }
    final maxTokens = int.tryParse(_maxTokensController.text);
    setState(() {
      _isSubmitting = true;
    });
    try {
      final payload = {
        'api_key': apiKey,
        'provider': _provider,
        'prompt': _promptController.text,
        'model': _resolveModel(),
        'max_tokens': maxTokens,
      };
      final response = await http
          .post(
            ApiConfig.httpUri('/script/generate'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 60));
      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final text = data['text'] as String? ?? '';
        setState(() {
          _outputTextController.text = text;
        });
        _showSnackBar('台本生成が完了しました。');
      } else {
        _showSnackBar('生成に失敗しました: ${response.statusCode} ${response.body}');
      }
    } on TimeoutException {
      _showSnackBar('リクエストがタイムアウトしました。');
    } catch (error) {
      _showSnackBar('生成エラー: $error');
    } finally {
      if (mounted) {
        setState(() {
          _isSubmitting = false;
        });
      }
    }
  }

  void _copyOutputText() {
    final text = _outputTextController.text;
    if (text.isEmpty) {
      _showSnackBar('コピーする内容がありません。');
      return;
    }
    Clipboard.setData(ClipboardData(text: text));
    _showSnackBar('生成結果をコピーしました。');
  }

  Future<void> _saveOutputToFile() async {
    final outputPath = _outputController.text.trim();
    if (outputPath.isEmpty) {
      _showSnackBar('保存先ファイルを指定してください。');
      return;
    }
    final text = _outputTextController.text;
    if (text.isEmpty) {
      _showSnackBar('保存する内容がありません。');
      return;
    }
    try {
      final file = File(outputPath);
      await file.writeAsString(text);
      _showSnackBar('保存しました: $outputPath');
    } catch (error) {
      _showSnackBar('保存に失敗しました: $error');
    }
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
  final _outputController = TextEditingController();
  String _provider = 'Gemini';
  String _geminiModel = 'gemini-2.0-flash';
  String _chatGptModel = 'gpt-4.1-mini';
  String _claudeModel = 'claude-opus-4-5-20251101';
  bool _isSubmitting = false;

  @override
  void dispose() {
    _scriptPathController.dispose();
    _countController.dispose();
    _instructionsController.dispose();
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
          if (_provider == 'Gemini')
            DropdownButtonFormField<String>(
              value: _geminiModel,
              decoration: const InputDecoration(labelText: 'Gemini モデル'),
              items: const [
                DropdownMenuItem(value: 'gemini-2.0-flash', child: Text('gemini-2.0-flash')),
                DropdownMenuItem(value: 'gemini-1.5-flash', child: Text('gemini-1.5-flash')),
                DropdownMenuItem(value: 'gemini-1.5-pro', child: Text('gemini-1.5-pro')),
              ],
              onChanged: (value) {
                if (value == null) return;
                setState(() {
                  _geminiModel = value;
                });
              },
            ),
          if (_provider == 'ChatGPT')
            DropdownButtonFormField<String>(
              value: _chatGptModel,
              decoration: const InputDecoration(labelText: 'ChatGPT モデル'),
              items: const [
                DropdownMenuItem(value: 'gpt-4.1-mini', child: Text('gpt-4.1-mini')),
                DropdownMenuItem(value: 'gpt-4.1', child: Text('gpt-4.1')),
                DropdownMenuItem(value: 'gpt-4o-mini', child: Text('gpt-4o-mini')),
                DropdownMenuItem(value: 'gpt-4o', child: Text('gpt-4o')),
              ],
              onChanged: (value) {
                if (value == null) return;
                setState(() {
                  _chatGptModel = value;
                });
              },
            ),
          if (_provider == 'ClaudeCode')
            DropdownButtonFormField<String>(
              value: _claudeModel,
              decoration: const InputDecoration(labelText: 'ClaudeCode モデル'),
              items: const [
                DropdownMenuItem(
                  value: 'claude-opus-4-5-20251101',
                  child: Text('claude-opus-4-5-20251101'),
                ),
              ],
              onChanged: (value) {
                if (value == null) return;
                setState(() {
                  _claudeModel = value;
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
                onPressed: _isSubmitting ? null : _submitTitleGenerate,
                icon: const Icon(Icons.play_arrow),
                label: Text(_isSubmitting ? '送信中...' : 'タイトル生成'),
              ),
              OutlinedButton.icon(
                onPressed: _copyOutput,
                icon: const Icon(Icons.copy),
                label: const Text('コピー'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text('生成結果', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          TextFormField(
            controller: _outputController,
            maxLines: 10,
            decoration: const InputDecoration(hintText: '生成結果がここに表示されます。'),
          ),
        ],
      ),
    );
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  String _resolveApiKey() {
    switch (_provider) {
      case 'ChatGPT':
        return ApiKeys.openAi.value;
      case 'ClaudeCode':
        return ApiKeys.claude.value;
      default:
        return ApiKeys.gemini.value;
    }
  }

  String _resolveModel() {
    switch (_provider) {
      case 'ChatGPT':
        return _chatGptModel;
      case 'ClaudeCode':
        return _claudeModel;
      default:
        return _geminiModel;
    }
  }

  Future<void> _submitTitleGenerate() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    final apiKey = _resolveApiKey();
    if (apiKey.isEmpty) {
      _showSnackBar('APIキーが未設定です。設定タブで入力してください。');
      return;
    }
    final count = int.tryParse(_countController.text) ?? 5;
    setState(() {
      _isSubmitting = true;
    });
    try {
      final payload = {
        'api_key': apiKey,
        'provider': _provider,
        'script_path': _scriptPathController.text,
        'count': count,
        'extra': _instructionsController.text,
        'model': _resolveModel(),
      };
      final response = await http
          .post(
            ApiConfig.httpUri('/title/generate'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 60));
      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _outputController.text = data['text'] as String? ?? '';
        });
        _showSnackBar('タイトル生成が完了しました。');
      } else {
        _showSnackBar('生成に失敗しました: ${response.statusCode} ${response.body}');
      }
    } on TimeoutException {
      _showSnackBar('リクエストがタイムアウトしました。');
    } catch (error) {
      _showSnackBar('生成エラー: $error');
    } finally {
      if (mounted) {
        setState(() {
          _isSubmitting = false;
        });
      }
    }
  }

  void _copyOutput() {
    final text = _outputController.text;
    if (text.isEmpty) {
      _showSnackBar('コピーする内容がありません。');
      return;
    }
    Clipboard.setData(ClipboardData(text: text));
    _showSnackBar('生成結果をコピーしました。');
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
  final _previewController = TextEditingController();
  bool _isSubmitting = false;

  @override
  void dispose() {
    _modelController.dispose();
    _promptController.dispose();
    _outputController.dispose();
    _previewController.dispose();
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
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _insertMaterialTemplate,
                  icon: const Icon(Icons.note_add),
                  label: const Text('雛形を挿入'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () {
                    _promptController.clear();
                  },
                  icon: const Icon(Icons.clear),
                  label: const Text('クリア'),
                ),
              ),
            ],
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
                onPressed: _isSubmitting ? null : _submitMaterialsGenerate,
                icon: const Icon(Icons.image),
                label: Text(_isSubmitting ? '生成中...' : '画像生成'),
              ),
              OutlinedButton.icon(
                onPressed: _copyPreviewPath,
                icon: const Icon(Icons.save),
                label: const Text('保存'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text('生成画像', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          TextFormField(
            controller: _previewController,
            maxLines: 6,
            readOnly: true,
            decoration: const InputDecoration(
              hintText: '生成された画像のパスがここに表示されます。',
            ),
          ),
        ],
      ),
    );
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  void _insertMaterialTemplate() {
    const template = [
      '主題/被写体:',
      '背景/場所:',
      '色味/雰囲気:',
      '構図/カメラアングル:',
      '必ず含めたい要素:',
      '避けたい要素:',
    ];
    _promptController.text = template.join('\n');
  }

  Future<void> _submitMaterialsGenerate() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    final apiKey = ApiKeys.gemini.value;
    if (apiKey.isEmpty) {
      _showSnackBar('Gemini APIキーが未設定です。設定タブで入力してください。');
      return;
    }
    setState(() {
      _isSubmitting = true;
    });
    try {
      final payload = {
        'api_key': apiKey,
        'prompt': _promptController.text,
        'model': _modelController.text,
        'output_dir': _outputController.text.trim().isEmpty
            ? null
            : _outputController.text.trim(),
      };
      final response = await http
          .post(
            ApiConfig.httpUri('/materials/generate'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 120));
      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final imagePath = data['image_path'] as String?;
        final imageBase64 = data['image_base64'] as String?;
        setState(() {
          _previewController.text = imagePath ?? imageBase64 ?? '';
        });
        final modelNote = data['model_note'] as String?;
        if (modelNote != null && modelNote.isNotEmpty) {
          _showSnackBar(modelNote);
        } else {
          _showSnackBar('画像生成が完了しました。');
        }
      } else {
        _showSnackBar('生成に失敗しました: ${response.statusCode} ${response.body}');
      }
    } on TimeoutException {
      _showSnackBar('リクエストがタイムアウトしました。');
    } catch (error) {
      _showSnackBar('生成エラー: $error');
    } finally {
      if (mounted) {
        setState(() {
          _isSubmitting = false;
        });
      }
    }
  }

  void _copyPreviewPath() {
    final text = _previewController.text.trim();
    if (text.isEmpty) {
      _showSnackBar('保存対象がありません。');
      return;
    }
    Clipboard.setData(ClipboardData(text: text));
    _showSnackBar('パスをコピーしました。');
  }
}

class PonchiGenerateForm extends StatefulWidget {
  const PonchiGenerateForm({super.key});

  @override
  State<PonchiGenerateForm> createState() => _PonchiGenerateFormState();
}

class _PonchiGenerateFormState extends State<PonchiGenerateForm> {
  final _formKey = GlobalKey<FormState>();
  final _srtController = TextEditingController();
  final _outputController = TextEditingController(text: 'ponchi_images');
  final _geminiModelController = TextEditingController(text: 'gemini-2.0-flash');
  final _chatGptModelController = TextEditingController(text: 'gpt-4.1-mini');
  final _outputTextController = TextEditingController();
  String _engine = 'Gemini';
  bool _isSubmittingIdeas = false;
  bool _isSubmittingImages = false;

  @override
  void dispose() {
    _srtController.dispose();
    _outputController.dispose();
    _geminiModelController.dispose();
    _chatGptModelController.dispose();
    _outputTextController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: ListView(
        children: [
          Text('ポンチ絵作成', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 16),
          TextFormField(
            controller: _srtController,
            decoration: InputDecoration(
              labelText: 'SRTファイル',
              suffixIcon: IconButton(
                icon: const Icon(Icons.subtitles),
                onPressed: () => _selectFile(
                  _srtController,
                  const XTypeGroup(label: 'SRT', extensions: ['srt']),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _outputController,
            decoration: InputDecoration(
              labelText: '出力フォルダ',
              suffixIcon: IconButton(
                icon: const Icon(Icons.folder),
                onPressed: () => _selectDirectory(_outputController),
              ),
            ),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _engine,
            decoration: const InputDecoration(labelText: '提案生成エンジン'),
            items: const [
              DropdownMenuItem(value: 'Gemini', child: Text('Gemini')),
              DropdownMenuItem(value: 'ChatGPT', child: Text('ChatGPT')),
            ],
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _engine = value;
              });
            },
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _geminiModelController,
            decoration: const InputDecoration(labelText: 'Gemini 提案モデル'),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _chatGptModelController,
            decoration: const InputDecoration(labelText: 'ChatGPT モデル'),
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              ElevatedButton.icon(
                onPressed: _isSubmittingIdeas ? null : _submitPonchiIdeas,
                icon: const Icon(Icons.lightbulb),
                label: Text(_isSubmittingIdeas ? '生成中...' : '案出し'),
              ),
              ElevatedButton.icon(
                onPressed: _isSubmittingImages ? null : _submitPonchiImages,
                icon: const Icon(Icons.brush),
                label: Text(_isSubmittingImages ? '生成中...' : 'ポンチ絵作成'),
              ),
              OutlinedButton.icon(
                onPressed: () {
                  _outputTextController.clear();
                },
                icon: const Icon(Icons.clear),
                label: const Text('クリア'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text('生成結果', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          TextFormField(
            controller: _outputTextController,
            maxLines: 10,
            decoration: const InputDecoration(
              hintText: '生成結果がここに表示されます。',
            ),
          ),
        ],
      ),
    );
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  String _resolvePonchiApiKey() {
    if (_engine == 'ChatGPT') {
      return ApiKeys.openAi.value;
    }
    return ApiKeys.gemini.value;
  }

  Future<void> _submitPonchiIdeas() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    final apiKey = _resolvePonchiApiKey();
    if (apiKey.isEmpty) {
      _showSnackBar('APIキーが未設定です。設定タブで入力してください。');
      return;
    }
    setState(() {
      _isSubmittingIdeas = true;
    });
    try {
      final payload = {
        'api_key': apiKey,
        'engine': _engine,
        'srt_path': _srtController.text,
        'output_dir': _outputController.text.trim().isEmpty
            ? null
            : _outputController.text.trim(),
        'gemini_model': _geminiModelController.text,
        'openai_model': _chatGptModelController.text,
      };
      final response = await http
          .post(
            ApiConfig.httpUri('/ponchi/ideas'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 120));
      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final items = data['items'] as List<dynamic>? ?? [];
        final jsonPath = data['json_path'] as String?;
        final buffer = StringBuffer();
        buffer.writeln('✅ ${items.length} 件の案を生成しました。');
        if (jsonPath != null) {
          buffer.writeln('JSON: $jsonPath');
        }
        for (final item in items) {
          final map = item as Map<String, dynamic>;
          buffer.writeln(
            '${map['start']}〜${map['end']} | ${map['visual_suggestion']} | ${map['image_prompt']}',
          );
        }
        setState(() {
          _outputTextController.text = buffer.toString();
        });
        _showSnackBar('ポンチ絵の案出しが完了しました。');
      } else {
        _showSnackBar('生成に失敗しました: ${response.statusCode} ${response.body}');
      }
    } on TimeoutException {
      _showSnackBar('リクエストがタイムアウトしました。');
    } catch (error) {
      _showSnackBar('生成エラー: $error');
    } finally {
      if (mounted) {
        setState(() {
          _isSubmittingIdeas = false;
        });
      }
    }
  }

  Future<void> _submitPonchiImages() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    final apiKey = ApiKeys.gemini.value;
    if (apiKey.isEmpty) {
      _showSnackBar('Gemini APIキーが未設定です。設定タブで入力してください。');
      return;
    }
    if (_outputController.text.trim().isEmpty) {
      _showSnackBar('出力フォルダを指定してください。');
      return;
    }
    setState(() {
      _isSubmittingImages = true;
    });
    try {
      final payload = {
        'api_key': apiKey,
        'srt_path': _srtController.text,
        'output_dir': _outputController.text.trim(),
      };
      final response = await http
          .post(
            ApiConfig.httpUri('/ponchi/images'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 300));
      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final items = data['items'] as List<dynamic>? ?? [];
        final outputDir = data['output_dir'] as String? ?? '';
        final jsonPath = data['json_path'] as String? ?? '';
        final buffer = StringBuffer();
        buffer.writeln('✅ ${items.length} 件のポンチ絵を生成しました。');
        if (outputDir.isNotEmpty) {
          buffer.writeln('出力フォルダ: $outputDir');
        }
        if (jsonPath.isNotEmpty) {
          buffer.writeln('JSON: $jsonPath');
        }
        for (final item in items) {
          final map = item as Map<String, dynamic>;
          buffer.writeln(
            '${map['start']}〜${map['end']} | ${map['visual_suggestion']} | ${map['image']}',
          );
        }
        setState(() {
          _outputTextController.text = buffer.toString();
        });
        _showSnackBar('ポンチ絵作成が完了しました。');
      } else {
        _showSnackBar('生成に失敗しました: ${response.statusCode} ${response.body}');
      }
    } on TimeoutException {
      _showSnackBar('リクエストがタイムアウトしました。');
    } catch (error) {
      _showSnackBar('生成エラー: $error');
    } finally {
      if (mounted) {
        setState(() {
          _isSubmittingImages = false;
        });
      }
    }
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
  final _srtController = TextEditingController();
  final _imageOutputController =
      TextEditingController(text: '${Directory.current.path}/srt_images');
  final _searchApiKeyController = TextEditingController();
  final _defaultXController = TextEditingController(text: '100');
  final _defaultYController = TextEditingController(text: '200');
  final _defaultWController = TextEditingController(text: '0');
  final _defaultHController = TextEditingController(text: '0');
  final _defaultOpacityController = TextEditingController(text: '1.0');
  String _searchProvider = 'Google';
  double _previewX = 0;
  double _previewY = 0;
  double _previewScale = 100;
  final List<Map<String, String>> _overlays = [];

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
    _srtController.dispose();
    _imageOutputController.dispose();
    _searchApiKeyController.dispose();
    _defaultXController.dispose();
    _defaultYController.dispose();
    _defaultWController.dispose();
    _defaultHController.dispose();
    _defaultOpacityController.dispose();
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
          Card(
            color: Theme.of(context).colorScheme.surface,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('プレビュー', style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 12),
                  Container(
                    height: 180,
                    width: double.infinity,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey.shade300),
                    ),
                    child: const Text('動画を選択してください'),
                  ),
                  const SizedBox(height: 12),
                  _buildSliderRow(
                    label: '画像X',
                    value: _previewX,
                    min: 0,
                    max: 1920,
                    onChanged: (value) {
                      setState(() {
                        _previewX = value;
                      });
                    },
                    displayValue: _previewX.toStringAsFixed(0),
                  ),
                  _buildSliderRow(
                    label: '画像Y',
                    value: _previewY,
                    min: 0,
                    max: 1080,
                    onChanged: (value) {
                      setState(() {
                        _previewY = value;
                      });
                    },
                    displayValue: _previewY.toStringAsFixed(0),
                  ),
                  _buildSliderRow(
                    label: '画像スケール(%)',
                    value: _previewScale,
                    min: 10,
                    max: 300,
                    onChanged: (value) {
                      setState(() {
                        _previewScale = value;
                      });
                    },
                    displayValue: '${_previewScale.toStringAsFixed(0)}%',
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
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
          Text(
            'オーバーレイ設定（1件ずつ追加）',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          Text(
            '時間は mm:ss / hh:mm:ss で指定。例: 00:12〜00:18',
            style: Theme.of(context).textTheme.bodySmall,
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
                    _overlays.add({
                      'image': _overlayImageController.text,
                      'start': _startController.text,
                      'end': _endController.text,
                      'x': _xController.text,
                      'y': _yController.text,
                      'w': _widthController.text,
                      'h': _heightController.text,
                      'opacity': _opacityController.text,
                    });
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
          Text('オーバーレイ一覧（表）', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: DataTable(
              columns: const [
                DataColumn(label: Text('image')),
                DataColumn(label: Text('start')),
                DataColumn(label: Text('end')),
                DataColumn(label: Text('x')),
                DataColumn(label: Text('y')),
                DataColumn(label: Text('w')),
                DataColumn(label: Text('h')),
                DataColumn(label: Text('opacity')),
              ],
              rows: _overlays
                  .map(
                    (overlay) => DataRow(
                      cells: [
                        DataCell(Text(overlay['image'] ?? '')),
                        DataCell(Text(overlay['start'] ?? '')),
                        DataCell(Text(overlay['end'] ?? '')),
                        DataCell(Text(overlay['x'] ?? '')),
                        DataCell(Text(overlay['y'] ?? '')),
                        DataCell(Text(overlay['w'] ?? '')),
                        DataCell(Text(overlay['h'] ?? '')),
                        DataCell(Text(overlay['opacity'] ?? '')),
                      ],
                    ),
                  )
                  .toList(),
            ),
          ),
          const SizedBox(height: 12),
          Text('SRTから画像収集', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          Text(
            '字幕ごとに検索キーワードを生成し、Google/Bing画像検索から取得します。',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _srtController,
            decoration: InputDecoration(
              labelText: 'SRTファイル',
              suffixIcon: IconButton(
                icon: const Icon(Icons.subtitles),
                onPressed: () => _selectFile(
                  _srtController,
                  const XTypeGroup(label: 'SRT', extensions: ['srt']),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _imageOutputController,
            decoration: InputDecoration(
              labelText: '保存先フォルダ',
              suffixIcon: IconButton(
                icon: const Icon(Icons.folder),
                onPressed: () => _selectDirectory(_imageOutputController),
              ),
            ),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _searchProvider,
            decoration: const InputDecoration(labelText: '画像検索プロバイダ'),
            items: const [
              DropdownMenuItem(value: 'Google', child: Text('Google')),
              DropdownMenuItem(value: 'Bing', child: Text('Bing')),
            ],
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _searchProvider = value;
              });
            },
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _searchApiKeyController,
            decoration: const InputDecoration(labelText: '画像検索 APIキー（SerpAPI）'),
            obscureText: true,
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              SizedBox(
                width: 120,
                child: TextFormField(
                  controller: _defaultXController,
                  decoration: const InputDecoration(labelText: '既定X'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 120,
                child: TextFormField(
                  controller: _defaultYController,
                  decoration: const InputDecoration(labelText: '既定Y'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 120,
                child: TextFormField(
                  controller: _defaultWController,
                  decoration: const InputDecoration(labelText: '既定W'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 120,
                child: TextFormField(
                  controller: _defaultHController,
                  decoration: const InputDecoration(labelText: '既定H'),
                  keyboardType: TextInputType.number,
                ),
              ),
              SizedBox(
                width: 140,
                child: TextFormField(
                  controller: _defaultOpacityController,
                  decoration: const InputDecoration(labelText: '既定Opacity'),
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
                onPressed: () {},
                icon: const Icon(Icons.auto_fix_high),
                label: const Text('SRTから画像収集'),
              ),
              OutlinedButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.upload_file),
                label: const Text('JSON読み込み'),
              ),
            ],
          ),
          const SizedBox(height: 16),
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

Widget _buildSliderRow({
  required String label,
  required double value,
  required double min,
  required double max,
  required ValueChanged<double> onChanged,
  required String displayValue,
}) {
  return Row(
    children: [
      SizedBox(width: 110, child: Text(label)),
      Expanded(
        child: Slider(
          value: value,
          min: min,
          max: max,
          onChanged: onChanged,
        ),
      ),
      SizedBox(width: 60, child: Text(displayValue, textAlign: TextAlign.end)),
    ],
  );
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
  final _mainVideoController = TextEditingController();
  final _clipInController = TextEditingController(text: '00:00');
  final _clipOutController = TextEditingController(text: '00:10');
  final _overlayImageController = TextEditingController();
  final _overlayStartController = TextEditingController(text: '00:00');
  final _overlayEndController = TextEditingController(text: '00:05');
  final _overlayXController = TextEditingController(text: '100');
  final _overlayYController = TextEditingController(text: '200');
  final _overlayOpacityController = TextEditingController(text: '1.0');
  final _exportPathController = TextEditingController();
  bool _audioEnabled = true;
  String _autoSaveStatus = '未設定';

  @override
  void dispose() {
    _projectNameController.dispose();
    _resolutionController.dispose();
    _fpsController.dispose();
    _mainVideoController.dispose();
    _clipInController.dispose();
    _clipOutController.dispose();
    _overlayImageController.dispose();
    _overlayStartController.dispose();
    _overlayEndController.dispose();
    _overlayXController.dispose();
    _overlayYController.dispose();
    _overlayOpacityController.dispose();
    _exportPathController.dispose();
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
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('プロジェクト管理', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 12),
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
                const SizedBox(height: 8),
                Text('自動保存: $_autoSaveStatus'),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 12,
                  runSpacing: 12,
                  children: [
                    ElevatedButton.icon(
                      onPressed: () {
                        setState(() {
                          _autoSaveStatus = '新規作成';
                        });
                      },
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
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('素材管理', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 12),
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
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('タイムライン', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _mainVideoController,
                  decoration: InputDecoration(
                    labelText: 'メイン動画',
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.video_library),
                      onPressed: () => _selectFile(
                        _mainVideoController,
                        const XTypeGroup(label: 'Video', extensions: ['mp4', 'mov', 'mkv']),
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
                      width: 140,
                      child: TextFormField(
                        controller: _clipInController,
                        decoration: const InputDecoration(labelText: 'In'),
                      ),
                    ),
                    SizedBox(
                      width: 140,
                      child: TextFormField(
                        controller: _clipOutController,
                        decoration: const InputDecoration(labelText: 'Out'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 12,
                  children: [
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
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('音声設定', style: Theme.of(context).textTheme.titleMedium),
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
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('オーバーレイ', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _overlayImageController,
                  decoration: InputDecoration(
                    labelText: '画像',
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.image),
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
                      width: 140,
                      child: TextFormField(
                        controller: _overlayStartController,
                        decoration: const InputDecoration(labelText: '開始'),
                      ),
                    ),
                    SizedBox(
                      width: 140,
                      child: TextFormField(
                        controller: _overlayEndController,
                        decoration: const InputDecoration(labelText: '終了'),
                      ),
                    ),
                    SizedBox(
                      width: 120,
                      child: TextFormField(
                        controller: _overlayXController,
                        decoration: const InputDecoration(labelText: 'X'),
                        keyboardType: TextInputType.number,
                      ),
                    ),
                    SizedBox(
                      width: 120,
                      child: TextFormField(
                        controller: _overlayYController,
                        decoration: const InputDecoration(labelText: 'Y'),
                        keyboardType: TextInputType.number,
                      ),
                    ),
                    SizedBox(
                      width: 140,
                      child: TextFormField(
                        controller: _overlayOpacityController,
                        decoration: const InputDecoration(labelText: '不透明度'),
                        keyboardType: TextInputType.number,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  onPressed: () {},
                  icon: const Icon(Icons.add),
                  label: const Text('オーバーレイ追加'),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('書き出し', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _exportPathController,
                  decoration: InputDecoration(
                    labelText: '出力先',
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.save_alt),
                      onPressed: () => _selectSavePath(
                        _exportPathController,
                        const XTypeGroup(label: 'Video', extensions: ['mp4', 'mov', 'mkv']),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                ElevatedButton.icon(
                  onPressed: () {},
                  icon: const Icon(Icons.movie_creation),
                  label: const Text('書き出し'),
                ),
              ],
            ),
          ),
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

class AboutPanel extends StatelessWidget {
  const AboutPanel({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      children: [
        Text('About', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 16),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text('News Short Generator Studio'),
                SizedBox(height: 8),
                Text('new_video_gui20.py の構成を再現した Flutter UI です。'),
                SizedBox(height: 8),
                Text('左: メニュー / 中央: フォーム / 右: ログ'),
              ],
            ),
          ),
        ),
      ],
    );
  }
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
    _geminiController.text = ApiKeys.gemini.value;
    _openAiController.text = ApiKeys.openAi.value;
    _claudeController.text = ApiKeys.claude.value;
    _geminiController.addListener(() {
      ApiKeys.gemini.value = _geminiController.text.trim();
    });
    _openAiController.addListener(() {
      ApiKeys.openAi.value = _openAiController.text.trim();
    });
    _claudeController.addListener(() {
      ApiKeys.claude.value = _claudeController.text.trim();
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
  final _widthController = TextEditingController(text: '1080');
  final _heightController = TextEditingController(text: '1920');
  final _fpsController = TextEditingController(text: '30');
  final _voiceController = TextEditingController(text: 'Kore');
  final _voicevoxUrlController =
      TextEditingController(text: 'http://127.0.0.1:50021');
  final _voicevoxRotationController = TextEditingController(text: '1,3');
  final _voicevoxCasterController = TextEditingController(text: '四国めたん');
  final _voicevoxAnalystController = TextEditingController(text: 'ずんだもん');
  final _captionFontSizeController = TextEditingController(text: '36');
  final _captionAlphaController = TextEditingController(text: '170');
  final _captionTextColorController = TextEditingController(text: '#FFFFFF');
  final _speakerFontSizeController = TextEditingController(text: '30');
  final _captionMaxCharsController = TextEditingController(text: '22');
  final _captionBoxHeightController = TextEditingController(text: '420');
  bool _useBgm = false;
  final _bgmController = TextEditingController();
  double _bgmGainDb = -18;
  String _ttsEngine = 'VOICEVOX';
  String _voicevoxMode = 'ローテーション';
  double _voicevoxSpeed = 1.0;
  bool _captionBoxEnabled = true;
  String _bgOffStyle = '影';
  bool _isSubmitting = false;
  String? _jobId;
  String _statusMessage = 'Ready';

  @override
  void dispose() {
    _scriptController.dispose();
    _imageListController.dispose();
    _outputController.dispose();
    _widthController.dispose();
    _heightController.dispose();
    _fpsController.dispose();
    _voiceController.dispose();
    _voicevoxUrlController.dispose();
    _voicevoxRotationController.dispose();
    _voicevoxCasterController.dispose();
    _voicevoxAnalystController.dispose();
    _captionFontSizeController.dispose();
    _captionAlphaController.dispose();
    _captionTextColorController.dispose();
    _speakerFontSizeController.dispose();
    _captionMaxCharsController.dispose();
    _captionBoxHeightController.dispose();
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
          Text(
            'Gemini API キーは設定タブで管理します。VOICEVOX 利用時は不要です。',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 12),
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
              labelText: '画像リスト（1行1枚）',
              hintText: 'image1.png\nimage2.png',
              suffixIcon: IconButton(
                icon: const Icon(Icons.collections),
                onPressed: () async {
                  final files = await openFiles(
                    acceptedTypeGroups: const [
                      XTypeGroup(
                        label: 'Images',
                        extensions: ['png', 'jpg', 'jpeg', 'webp'],
                      ),
                    ],
                  );
                  if (files.isEmpty) return;
                  final existing = _imageListController.text.trim();
                  final newLines = files.map((file) => file.path).join('\n');
                  _imageListController.text =
                      existing.isEmpty ? newLines : '$existing\n$newLines';
                },
              ),
            ),
            maxLines: 4,
            validator: (value) => value == null || value.isEmpty ? '必須です' : null,
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () async {
                    final files = await openFiles(
                      acceptedTypeGroups: const [
                        XTypeGroup(
                          label: 'Images',
                          extensions: ['png', 'jpg', 'jpeg', 'webp'],
                        ),
                      ],
                    );
                    if (files.isEmpty) return;
                    _imageListController.text =
                        files.map((file) => file.path).join('\n');
                  },
                  icon: const Icon(Icons.add_photo_alternate),
                  label: const Text('画像を追加'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () {
                    _imageListController.clear();
                  },
                  icon: const Icon(Icons.delete_sweep),
                  label: const Text('全削除'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
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
            Column(
              children: [
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
                const SizedBox(height: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'BGM 音量(dB, マイナスで小さく)',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    Slider(
                      value: _bgmGainDb,
                      min: -30,
                      max: 5,
                      divisions: 35,
                      label: _bgmGainDb.toStringAsFixed(1),
                      onChanged: (value) {
                        setState(() {
                          _bgmGainDb = value;
                        });
                      },
                    ),
                  ],
                ),
              ],
            ),
          const SizedBox(height: 12),
          Text(
            'TTS エンジン',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            value: _ttsEngine,
            decoration: const InputDecoration(labelText: '音声合成エンジン'),
            items: const [
              DropdownMenuItem(value: 'Gemini', child: Text('Gemini')),
              DropdownMenuItem(value: 'VOICEVOX', child: Text('VOICEVOX')),
            ],
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _ttsEngine = value;
              });
            },
          ),
          const SizedBox(height: 12),
          if (_ttsEngine == 'Gemini')
            TextFormField(
              controller: _voiceController,
              decoration: const InputDecoration(labelText: 'Gemini 音声'),
            ),
          if (_ttsEngine == 'VOICEVOX') ...[
            TextFormField(
              controller: _voicevoxUrlController,
              decoration: const InputDecoration(labelText: 'VOICEVOX エンジンURL'),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              value: _voicevoxMode,
              decoration: const InputDecoration(labelText: '話者モード'),
              items: const [
                DropdownMenuItem(value: 'ローテーション', child: Text('ローテーション')),
                DropdownMenuItem(value: '2人対談', child: Text('2人対談')),
              ],
              onChanged: (value) {
                if (value == null) return;
                setState(() {
                  _voicevoxMode = value;
                });
              },
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _voicevoxRotationController,
              decoration: const InputDecoration(labelText: 'ローテーション話者(カンマ)'),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _voicevoxCasterController,
              decoration: const InputDecoration(labelText: 'キャスター話者'),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _voicevoxAnalystController,
              decoration: const InputDecoration(labelText: 'アナリスト話者'),
            ),
            const SizedBox(height: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('話速(0.5〜2.0)'),
                Slider(
                  value: _voicevoxSpeed,
                  min: 0.5,
                  max: 2.0,
                  divisions: 30,
                  label: _voicevoxSpeed.toStringAsFixed(2),
                  onChanged: (value) {
                    setState(() {
                      _voicevoxSpeed = value;
                    });
                  },
                ),
              ],
            ),
          ],
          const SizedBox(height: 12),
          Text(
            '字幕設定',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          TextFormField(
            controller: _captionFontSizeController,
            decoration: const InputDecoration(labelText: '字幕フォントサイズ'),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _captionAlphaController,
            decoration: const InputDecoration(labelText: '字幕背景の透明度(alpha 0-255)'),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _bgOffStyle,
            decoration: const InputDecoration(labelText: '背景OFF時のデザイン'),
            items: const [
              DropdownMenuItem(value: '影', child: Text('影')),
              DropdownMenuItem(value: '角丸パネル', child: Text('角丸パネル')),
              DropdownMenuItem(value: 'なし', child: Text('なし')),
            ],
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _bgOffStyle = value;
              });
            },
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _captionTextColorController,
            decoration: const InputDecoration(labelText: '字幕文字色（#RRGGBB）'),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _speakerFontSizeController,
            decoration: const InputDecoration(labelText: '話者名フォントサイズ'),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _captionMaxCharsController,
            decoration: const InputDecoration(labelText: '1行あたり最大文字数'),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 12),
          SwitchListTile(
            value: _captionBoxEnabled,
            title: const Text('字幕背景（黒幕）を表示する（固定高さ）'),
            onChanged: (value) {
              setState(() {
                _captionBoxEnabled = value;
              });
            },
          ),
          TextFormField(
            controller: _captionBoxHeightController,
            decoration: const InputDecoration(labelText: '字幕背景の高さ(px, 固定)'),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 12),
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
      _statusMessage = 'Checking API...';
    });

    final isHealthy = await _isApiHealthy();
    if (!isHealthy) {
      setState(() {
        _statusMessage =
            'Error: API サーバーに接続できません。バックエンドURLを確認するか、サーバーを起動してください。';
        _isSubmitting = false;
      });
      return;
    }

    final imagePaths = _imageListController.text
        .split(RegExp(r'[\n,]'))
        .map((path) => path.trim())
        .where((path) => path.isNotEmpty)
        .toList();

    final rotationLabels = _voicevoxRotationController.text
        .split(',')
        .map((label) => label.trim())
        .where((label) => label.isNotEmpty)
        .toList();

    final voicevoxMode = _voicevoxMode == 'ローテーション' ? 'rotation' : 'two_person';

    final bgOffStyle = switch (_bgOffStyle) {
      '影' => 'shadow',
      '角丸パネル' => 'rounded_panel',
      _ => 'none',
    };

    final payload = {
      'api_key': ApiKeys.gemini.value,
      'script_path': _scriptController.text,
      'image_paths': imagePaths,
      'use_bgm': _useBgm,
      'bgm_path': _bgmController.text,
      'bgm_gain_db': _bgmGainDb,
      'output_dir': _outputController.text,
      'width': int.tryParse(_widthController.text) ?? 1080,
      'height': int.tryParse(_heightController.text) ?? 1920,
      'fps': int.tryParse(_fpsController.text) ?? 30,
      'voice_name': _voiceController.text,
      'tts_engine': _ttsEngine,
      'vv_mode': voicevoxMode,
      'vv_rotation_labels': rotationLabels.isEmpty ? null : rotationLabels,
      'vv_caster_label': _voicevoxCasterController.text,
      'vv_analyst_label': _voicevoxAnalystController.text,
      'vv_base_url': _voicevoxUrlController.text,
      'vv_speed_scale': _voicevoxSpeed,
      'caption_font_size': int.tryParse(_captionFontSizeController.text) ?? 36,
      'speaker_font_size': int.tryParse(_speakerFontSizeController.text) ?? 30,
      'caption_max_chars': int.tryParse(_captionMaxCharsController.text) ?? 22,
      'caption_box_alpha': int.tryParse(_captionAlphaController.text) ?? 170,
      'caption_box_enabled': _captionBoxEnabled,
      'caption_box_height': int.tryParse(_captionBoxHeightController.text) ?? 420,
      'bg_off_style': bgOffStyle,
      'caption_text_color': _captionTextColorController.text,
    };

    try {
      setState(() {
        _statusMessage = 'Submitting...';
      });
      final response = await http
          .post(
            ApiConfig.httpUri('/video/generate'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 30));

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final jobId = data['job_id'] as String?;
        setState(() {
          _jobId = jobId;
          _statusMessage = 'Submitted';
        });
        widget.onJobSubmitted?.call(jobId);
      } else {
        final responseBody = response.body.isEmpty ? '' : ' ${response.body}';
        setState(() {
          _statusMessage = 'Error: ${response.statusCode}$responseBody';
        });
      }
    } on TimeoutException {
      setState(() {
        _statusMessage = 'Error: request timed out (30s)';
      });
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

  Future<bool> _isApiHealthy() async {
    try {
      final response = await http
          .get(ApiConfig.httpUri('/health'))
          .timeout(const Duration(seconds: 3));
      return response.statusCode >= 200 && response.statusCode < 300;
    } catch (_) {
      return false;
    }
  }
}

class LogPanel extends StatefulWidget {
  const LogPanel({
    super.key,
    required this.pageName,
    this.latestJobId,
  });

  final String pageName;
  final ValueListenable<String?>? latestJobId;

  @override
  State<LogPanel> createState() => _LogPanelState();
}

enum _LogLevel { info, warning, error, success }

class _LogEntry {
  _LogEntry({
    required this.timestamp,
    required this.message,
    this.level = _LogLevel.info,
  });

  final DateTime timestamp;
  final String message;
  final _LogLevel level;
}

class _LogPanelState extends State<LogPanel> {
  final List<_LogEntry> _logs = [];
  final ScrollController _scrollController = ScrollController();
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  String? _currentJobId;
  double _progress = 0.0;
  String? _eta;

  @override
  void initState() {
    super.initState();
    widget.latestJobId?.addListener(_handleLatestJobIdChanged);
  }

  @override
  void didUpdateWidget(LogPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.latestJobId != widget.latestJobId) {
      oldWidget.latestJobId?.removeListener(_handleLatestJobIdChanged);
      widget.latestJobId?.addListener(_handleLatestJobIdChanged);
    }
  }

  @override
  void dispose() {
    widget.latestJobId?.removeListener(_handleLatestJobIdChanged);
    _subscription?.cancel();
    _channel?.sink.close();
    _scrollController.dispose();
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
            onPressed: _connectLatestJob,
            icon: const Icon(Icons.auto_graph),
            label: const Text('最新ジョブに自動接続'),
          ),
        ),
        if (_currentJobId != null) ...[
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
            child: Text('Job: $_currentJobId'),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: LinearProgressIndicator(value: _progress),
          ),
          if (_eta != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 0),
              child: Text('ETA: $_eta'),
            ),
        ],
        const SizedBox(height: 8),
        Expanded(
          child: Container(
            margin: const EdgeInsets.all(16),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [Colors.white, Colors.grey.shade50],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              border: Border.all(color: Colors.grey.shade200),
              borderRadius: BorderRadius.circular(16),
            ),
            child: ListView.builder(
              controller: _scrollController,
              itemCount: _logs.length,
              itemBuilder: (context, index) {
                final entry = _logs[index];
                return Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Text(
                    '[${_formatTimestamp(entry.timestamp)}] ${entry.message}',
                    style: TextStyle(color: _logColor(entry.level)),
                  ),
                );
              },
            ),
          ),
        ),
      ],
    );
  }

  void _handleLatestJobIdChanged() {
    final jobId = widget.latestJobId?.value;
    if (jobId == null || jobId.isEmpty || jobId == _currentJobId) {
      return;
    }
    _connectWebSocket(jobId);
  }

  void _connectLatestJob() {
    final jobId = widget.latestJobId?.value;
    if (jobId == null || jobId.isEmpty) {
      _addLog('まだジョブが送信されていません。動画生成を開始してください。');
      return;
    }
    _connectWebSocket(jobId);
  }

  void _connectWebSocket(String jobId) {
    _channel?.sink.close();
    _subscription?.cancel();

    final channel = WebSocketChannel.connect(
      ApiConfig.wsUri('/ws/jobs/$jobId'),
    );

    setState(() {
      _currentJobId = jobId;
      _logs.clear();
      _progress = 0.0;
      _eta = null;
      _channel = channel;
    });
    _addLog('Connecting to $jobId ...');

    _subscription = channel.stream.listen(
      (event) {
        _handleSocketEvent(event);
      },
      onError: (error) {
        _addLog('WebSocket error: $error', level: _LogLevel.error);
      },
      onDone: () {
        _addLog('WebSocket closed', level: _LogLevel.warning);
      },
    );
  }

  void _handleSocketEvent(dynamic event) {
    Map<String, dynamic>? payload;
    if (event is Map<String, dynamic>) {
      payload = event;
    } else if (event is String) {
      try {
        final decoded = jsonDecode(event);
        if (decoded is Map<String, dynamic>) {
          payload = decoded;
        }
      } catch (_) {
        payload = null;
      }
    }

    if (payload == null) {
      _addLog(event.toString());
      return;
    }

    final type = payload['type'] as String? ?? 'log';
    switch (type) {
      case 'progress':
        final progress = (payload['progress'] as num?)?.toDouble() ?? 0.0;
        final etaSeconds = payload['eta_seconds'];
        setState(() {
          _progress = progress;
          _eta = etaSeconds == null ? null : '${etaSeconds.toString()} sec';
        });
        _addLog(
          '進捗: ${(progress * 100).toStringAsFixed(1)}% '
          '${_eta == null ? '' : '(ETA: $_eta)'}',
        );
        return;
      case 'error':
        _addLog('エラー: ${payload['message']}', level: _LogLevel.error);
        return;
      case 'completed':
        _addLog('完了: ${jsonEncode(payload['result'])}', level: _LogLevel.success);
        return;
      case 'log':
      default:
        _addLog('${payload['message']}');
        return;
    }
  }

  void _addLog(String message, { _LogLevel level = _LogLevel.info }) {
    setState(() {
      _logs.add(_LogEntry(timestamp: DateTime.now(), message: message, level: level));
    });
    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    });
  }

  String _formatTimestamp(DateTime time) {
    final h = time.hour.toString().padLeft(2, '0');
    final m = time.minute.toString().padLeft(2, '0');
    final s = time.second.toString().padLeft(2, '0');
    return '$h:$m:$s';
  }

  Color _logColor(_LogLevel level) {
    switch (level) {
      case _LogLevel.error:
        return Colors.redAccent;
      case _LogLevel.warning:
        return Colors.orangeAccent;
      case _LogLevel.success:
        return Colors.green;
      case _LogLevel.info:
      default:
        return Colors.black87;
    }
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
