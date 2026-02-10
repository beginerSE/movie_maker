import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'config/input_persistence.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ApiSettingsBootstrap.load();
  runApp(const MovieMakerApp());
}

String _defaultApiBaseUrl() {
  if (Platform.isAndroid) {
    return 'http://10.0.2.2:8000';
  }
  return 'http://127.0.0.1:8000';
}

String _normalizeBackendUrlForCurrentPlatform(String rawUrl) {
  final trimmed = rawUrl.trim();
  if (trimmed.isEmpty) {
    return _defaultApiBaseUrl();
  }
  final uri = Uri.tryParse(trimmed);
  if (uri == null || !uri.hasScheme || uri.host.isEmpty) {
    return trimmed;
  }
  if (Platform.isAndroid) {
    final normalizedHost = uri.host.toLowerCase();
    if (normalizedHost == 'localhost' ||
        normalizedHost == '127.0.0.1' ||
        normalizedHost == '::1') {
      return uri.replace(host: '10.0.2.2').toString();
    }
  }
  return trimmed;
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

Uri _normalizeVoicevoxSpeakersUri(String baseUrl) {
  final rawUri = Uri.parse(baseUrl.trim());
  var path = rawUri.path;
  if (path.endsWith('/speakers')) {
    path = path.substring(0, path.length - '/speakers'.length);
  } else if (path.endsWith('/speaker')) {
    path = path.substring(0, path.length - '/speaker'.length);
  }
  final speakersPath = _joinPaths(path, '/speakers');
  return rawUri.replace(path: speakersPath);
}

Uri _withApiPrefix(Uri uri) {
  if (uri.path == '/api' || uri.path.startsWith('/api/')) {
    return uri;
  }
  final normalizedPath = uri.path.startsWith('/') ? uri.path : '/${uri.path}';
  return uri.replace(path: '/api$normalizedPath');
}

Future<http.Response> _getWithApiPrefixFallback(String path) async {
  final uri = ApiConfig.httpUri(path);
  final response = await http.get(uri);
  if (response.statusCode != 404) {
    return response;
  }
  final fallbackUri = _withApiPrefix(uri);
  if (fallbackUri.path == uri.path) {
    return response;
  }
  return http.get(fallbackUri);
}

Future<http.Response> _postWithApiPrefixFallback(
  String path, {
  Map<String, String>? headers,
  Object? body,
}) async {
  final uri = ApiConfig.httpUri(path);
  final response = await http.post(uri, headers: headers, body: body);
  if (response.statusCode != 404) {
    return response;
  }
  final fallbackUri = _withApiPrefix(uri);
  if (fallbackUri.path == uri.path) {
    return response;
  }
  return http.post(fallbackUri, headers: headers, body: body);
}

List<String> _extractVoicevoxSpeakerNames(dynamic data) {
  if (data is! List) {
    return [];
  }
  final names = <String>[];
  final seen = <String>{};
  for (final item in data) {
    if (item is! Map<String, dynamic>) {
      continue;
    }
    final name = item['name'];
    if (name is! String) {
      continue;
    }
    final trimmed = name.trim();
    if (trimmed.isEmpty || seen.contains(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    names.add(trimmed);
  }
  return names;
}

String _extractApiErrorDetail(String responseBody) {
  final body = responseBody.trim();
  if (body.isEmpty) {
    return '';
  }
  try {
    final decoded = jsonDecode(body);
    if (decoded is Map<String, dynamic>) {
      final detail = decoded['detail'];
      if (detail is String && detail.trim().isNotEmpty) {
        return detail.trim();
      }
      if (detail is List) {
        final messages = detail
            .map((item) {
              if (item is Map<String, dynamic>) {
                final loc = (item['loc'] as List<dynamic>? ?? [])
                    .map((part) => part.toString())
                    .join('/');
                final msg = (item['msg'] as String? ?? '').trim();
                if (loc.isNotEmpty && msg.isNotEmpty) {
                  return '$loc: $msg';
                }
                return msg;
              }
              return item.toString();
            })
            .where((message) => message.trim().isNotEmpty)
            .toList();
        if (messages.isNotEmpty) {
          return messages.join(', ');
        }
      }
      final message = decoded['message'];
      if (message is String && message.trim().isNotEmpty) {
        return message.trim();
      }
      final error = decoded['error'];
      if (error is String && error.trim().isNotEmpty) {
        return error.trim();
      }
    }
  } catch (_) {
    // ignore JSON parse errors and fallback to raw body.
  }
  return body;
}

String _formatApiFailureMessage({
  required String action,
  required Uri uri,
  http.Response? response,
  Object? error,
}) {
  final pathInfo = '${uri.path}${uri.hasQuery ? '?${uri.query}' : ''}';
  if (response != null) {
    final detail = _extractApiErrorDetail(response.body);
    final suffix = detail.isEmpty ? '' : ' / $detail';
    return '$action (PATH: $pathInfo, HTTP ${response.statusCode}$suffix)';
  }
  if (error != null) {
    return '$action (PATH: $pathInfo, ERROR: $error)';
  }
  return '$action (PATH: $pathInfo)';
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

class VoicevoxConfig {
  static final ValueNotifier<String> baseUrl =
      ValueNotifier<String>('http://127.0.0.1:50021');
}

typedef ApiHealthCheck = Future<bool> Function();

class ApiKeys {
  static final ValueNotifier<String> gemini = ValueNotifier<String>('');
  static final ValueNotifier<String> openAi = ValueNotifier<String>('');
  static final ValueNotifier<String> claude = ValueNotifier<String>('');
}

class ProjectState {
  static const String defaultProjectId = 'default';
  static final ValueNotifier<String> currentProjectId =
      ValueNotifier<String>(defaultProjectId);
  static const String undeletableProjectId = defaultProjectId;
}

class ProjectSummary {
  ProjectSummary({required this.id, required this.name, required this.projectType});

  final String id;
  final String name;
  final String projectType;

  bool get isFlowProject => projectType == 'flow';

  factory ProjectSummary.fromJson(Map<String, dynamic> json) {
    final id = (json['id'] as String? ?? '').trim();
    final projectType = (json['project_type'] as String? ?? 'standard').trim().toLowerCase();
    return ProjectSummary(
      id: id,
      name: (json['name'] as String? ?? id).trim(),
      projectType: projectType == 'flow' ? 'flow' : 'standard',
    );
  }
}

class ProjectFlowStep {
  const ProjectFlowStep({required this.key, required this.label});

  final String key;
  final String label;
}

const List<ProjectFlowStep> kFlowSteps = [
  ProjectFlowStep(key: 'script', label: '① 台本作成（AI + 人間修正）'),
  ProjectFlowStep(key: 'base_video', label: '② 動画作成（ベース動画生成）'),
  ProjectFlowStep(key: 'title_description', label: '③ 動画タイトル・説明文作成（AI）'),
  ProjectFlowStep(key: 'thumbnail', label: '④ サムネイル作成（AI）'),
  ProjectFlowStep(key: 'ponchi', label: '⑤ ポンチ絵（補足ビジュアル）案の作成'),
  ProjectFlowStep(key: 'final_edit', label: '⑥ 動画編集（最終編集）'),
];

const List<String> kFlowStatuses = ['未着手', '編集中', '完了'];

class ApiSettingsBootstrap {
  static Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    final savedBackendUrl =
        prefs.getString('settings.backend_url') ?? _defaultApiBaseUrl();
    ApiConfig.baseUrl.value =
        _normalizeBackendUrlForCurrentPlatform(savedBackendUrl);
    ApiKeys.gemini.value = (prefs.getString('settings.gemini_key') ?? '').trim();
    ApiKeys.openAi.value = (prefs.getString('settings.openai_key') ?? '').trim();
    ApiKeys.claude.value = (prefs.getString('settings.claude_key') ?? '').trim();
    VoicevoxConfig.baseUrl.value =
        (prefs.getString('video_generate.vv_url') ?? 'http://127.0.0.1:50021').trim();
    ProjectState.currentProjectId.value =
        (prefs.getString('project.current_id') ?? ProjectState.defaultProjectId)
            .trim();
  }
}

class MovieMakerApp extends StatelessWidget {
  const MovieMakerApp({super.key});
  static const String appTitle = 'News Short Generator Studio';

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
      title: appTitle,
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
    'プロジェクト一覧',
    '台本生成',
    '動画作成',
    '動画タイトル・説明',
    'サムネイル作成',
    'ポンチ絵作成',
    '動画編集',
    '詳細動画編集',
    '設定',
    'About',
    'AIフロー',
  ];
  int _selectedIndex = 0;
  int? _hoveredIndex;
  Process? _apiServerProcess;
  bool _apiServerStarting = false;
  bool _apiServerReady = false;
  String? _apiServerStatus;
  String? _apiServerErrorDetails;
  String? _apiServerLaunchCommand;
  Directory? _apiServerRoot;
  int _apiServerPort = 8000;
  final ValueNotifier<String?> _latestJobId = ValueNotifier<String?>(null);
  final ValueNotifier<bool> _videoJobInProgress = ValueNotifier<bool>(false);
  static const double _navDotSize = 10;
  List<ProjectSummary> _projects = const [];
  bool _projectsLoading = false;

  @override
  void initState() {
    super.initState();
    _ensureApiServerRunning();
    _loadProjects();
  }

  @override
  void dispose() {
    _apiServerProcess?.kill();
    _latestJobId.dispose();
    _videoJobInProgress.dispose();
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
      final portNote =
          _apiServerPort == 8000 ? '' : ' (port: $_apiServerPort)';
      _setApiServerStatus(ready: true, message: 'API サーバー起動完了$portNote');
      _clearApiServerError();
    } else {
      _setApiServerStatus(
        ready: false,
        message: 'API サーバーの起動に失敗しました。手動で起動してください。',
      );
        _showApiServerSnackBar(
          'API サーバーの起動に失敗しました。ターミナルで '
        '${_apiServerLaunchCommand ?? 'python -m uvicorn backend.api_server:app --host 127.0.0.1 --port $_apiServerPort'} '
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

  List<String> _candidatePythonExecutables() {
    final candidates = <String>[];
    final pythonNames = Platform.isWindows
        ? ['python.exe', 'python']
        : ['python3', 'python'];
    final envRoots = <String?>[
      Platform.environment['VIRTUAL_ENV'],
      Platform.environment['CONDA_PREFIX'],
    ];
    for (final root in envRoots) {
      if (root == null || root.trim().isEmpty) {
        continue;
      }
      final executable = Platform.isWindows
          ? _joinFilePath([root, 'Scripts', 'python.exe'])
          : _joinFilePath([root, 'bin', 'python']);
      if (File(executable).existsSync()) {
        candidates.add(executable);
      }
    }
    final localEnvRoots = [
      '.venv',
      'venv',
      'env',
    ];
    for (final localRoot in localEnvRoots) {
      final root = _joinFilePath([_apiServerRoot!.path, localRoot]);
      final executable = Platform.isWindows
          ? _joinFilePath([root, 'Scripts', 'python.exe'])
          : _joinFilePath([root, 'bin', 'python']);
      if (File(executable).existsSync()) {
        candidates.add(executable);
      }
    }
    candidates.addAll(pythonNames);
    return candidates;
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

    _apiServerPort = await _selectApiServerPort();
    _updateApiBaseUrlForPort(_apiServerPort);

    final pythonExecutables = _candidatePythonExecutables();

    for (final pythonExecutable in pythonExecutables) {
      try {
        _apiServerLaunchCommand =
            '$pythonExecutable -m uvicorn backend.api_server:app --host 127.0.0.1 --port $_apiServerPort';
        final process = await Process.start(
          pythonExecutable,
          [
            '-m',
            'uvicorn',
            'backend.api_server:app',
            '--host',
            '127.0.0.1',
            '--port',
            '$_apiServerPort',
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

  Future<int> _selectApiServerPort() async {
    final baseUri = Uri.tryParse(ApiConfig.baseUrl.value.trim());
    if (baseUri != null && _isLocalHost(baseUri.host) && baseUri.port != 0) {
      if (await _isPortAvailable(baseUri.port)) {
        return baseUri.port;
      }
      final released = await _attemptPortRelease(baseUri.port);
      if (released && await _isPortAvailable(baseUri.port)) {
        return baseUri.port;
      }
    }

    const fallbackPorts = [8000, 8001, 8002, 8003, 8004, 8005];
    for (final port in fallbackPorts) {
      if (await _isPortAvailable(port)) {
        return port;
      }
      final released = await _attemptPortRelease(port);
      if (released && await _isPortAvailable(port)) {
        return port;
      }
    }
    return 8000;
  }

  Future<bool> _isPortAvailable(int port) async {
    try {
      final socket =
          await ServerSocket.bind(InternetAddress.loopbackIPv4, port);
      await socket.close();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> _attemptPortRelease(int port) async {
    try {
      if (Platform.isWindows) {
        await Process.run(
          'powershell',
          [
            '-NoProfile',
            '-Command',
            'Get-NetTCPConnection -LocalPort ${port} '
                '| Select-Object -ExpandProperty OwningProcess '
                '| ForEach-Object { Stop-Process -Id \$_ -Force }',
          ],
          runInShell: true,
        );
      } else if (Platform.isLinux || Platform.isMacOS) {
        await Process.run(
          'sh',
          [
            '-c',
            'pids=\$(lsof -ti tcp:${port} 2>/dev/null); '
                'if [ -n "\$pids" ]; then kill -9 \$pids; fi',
          ],
          runInShell: true,
        );
      } else {
        return false;
      }
    } catch (_) {
      return false;
    }
    return true;
  }

  bool _isLocalHost(String host) {
    final normalized = host.toLowerCase();
    return normalized == 'localhost' ||
        normalized == '127.0.0.1' ||
        normalized == '::1';
  }

  void _updateApiBaseUrlForPort(int port) {
    final current = ApiConfig.baseUrl.value.trim();
    final baseUri = Uri.tryParse(current);
    if (baseUri == null) {
      return;
    }
    if (!_isLocalHost(baseUri.host)) {
      return;
    }
    final updated = baseUri.replace(port: port);
    ApiConfig.baseUrl.value = updated.toString();
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

  Future<bool> _checkApiHealthAndUpdate() async {
    final healthy = await _isApiHealthy();
    if (healthy) {
      _setApiServerStatus(ready: true, message: 'API サーバー稼働中');
      _clearApiServerError();
    } else {
      _setApiServerStatus(
        ready: false,
        message: 'API サーバーに接続できません。',
      );
    }
    return healthy;
  }

  Future<void> _loadProjects() async {
    setState(() {
      _projectsLoading = true;
    });
    try {
      final response = await _getWithApiPrefixFallback('/projects');
      if (response.statusCode < 200 || response.statusCode >= 300) {
        return;
      }
      final body = jsonDecode(response.body) as Map<String, dynamic>;
      final projectsRaw = (body['projects'] as List<dynamic>? ?? [])
          .whereType<Map<String, dynamic>>()
          .toList();
      final projects = projectsRaw.map(ProjectSummary.fromJson).toList();
      final current = ProjectState.currentProjectId.value;
      if (projects.where((p) => p.id == current).isEmpty) {
        final fallback =
            (body['default_project_id'] as String? ?? ProjectState.defaultProjectId)
                .trim();
        await _setCurrentProject(fallback);
      }
      if (!mounted) return;
      setState(() {
        _projects = projects;
      });
    } catch (_) {
      return;
    } finally {
      if (!mounted) return;
      setState(() {
        _projectsLoading = false;
      });
    }
  }

  Future<void> _setCurrentProject(String projectId) async {
    final normalized = projectId.trim().isEmpty
        ? ProjectState.defaultProjectId
        : projectId.trim();
    ProjectState.currentProjectId.value = normalized;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('project.current_id', normalized);
  }

  ProjectSummary? _currentProjectSummary() {
    final currentId = ProjectState.currentProjectId.value;
    for (final project in _projects) {
      if (project.id == currentId) {
        return project;
      }
    }
    return null;
  }

  Future<void> _openFlowPage() async {
    setState(() {
      _selectedIndex = 10;
    });
  }

  Future<void> _openProjectManager() async {
    await showDialog<void>(
      context: context,
      builder: (context) => _ProjectManagerDialog(
        projects: _projects,
        onChanged: _loadProjects,
      ),
    );
    await _loadProjects();
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
                    child: Column(
                      children: [
                        Padding(
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
                              const SizedBox(height: 12),
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 12),
                                child: Column(
                                  children: [
                                    ValueListenableBuilder<String>(
                                      valueListenable: ProjectState.currentProjectId,
                                      builder: (context, currentProjectId, _) {
                                        final hasCurrent = _projects.any((p) => p.id == currentProjectId);
                                        return DropdownButtonFormField<String>(
                                          value: hasCurrent ? currentProjectId : null,
                                          isExpanded: true,
                                          hint: Text(_projectsLoading ? 'プロジェクト読込中...' : 'プロジェクト選択'),
                                          items: _projects
                                              .map(
                                                (p) => DropdownMenuItem<String>(
                                                  value: p.id,
                                                  child: Text(
                                                    p.name,
                                                    overflow: TextOverflow.ellipsis,
                                                    style: const TextStyle(fontSize: 12),
                                                  ),
                                                ),
                                              )
                                              .toList(),
                                          onChanged: (value) {
                                            if (value != null) {
                                              _setCurrentProject(value);
                                            }
                                          },
                                        );
                                      },
                                    ),
                                    const SizedBox(height: 8),
                                    SizedBox(
                                      width: double.infinity,
                                      child: OutlinedButton(
                                        onPressed: _openProjectManager,
                                        child: const Text('管理', style: TextStyle(fontSize: 12)),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                        Expanded(
                          child: ListView.builder(
                            padding: const EdgeInsets.symmetric(horizontal: 12),
                            itemCount: _pages.length,
                            itemBuilder: (context, index) {
                              return _buildMenuItem(index: index, label: _pages[index]);
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
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
                  Expanded(
                    flex: 2,
                    child: Card(
                      margin: const EdgeInsets.all(20),
                      child: LogPanel(
                        pageName: _pages[_selectedIndex],
                        latestJobId: _latestJobId,
                        jobInProgress: _videoJobInProgress,
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

  Widget _buildMenuItem({required int index, required String label}) {
    final bool isSelected = _selectedIndex == index;
    final bool isHovered = _hoveredIndex == index;
    final Color backgroundColor = isSelected
        ? Colors.black.withOpacity(0.06)
        : isHovered
            ? Colors.black.withOpacity(0.04)
            : Colors.transparent;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(8),
          onTap: () {
            setState(() {
              _selectedIndex = index;
            });
          },
          onHover: (hovered) {
            setState(() {
              if (hovered) {
                _hoveredIndex = index;
              } else if (_hoveredIndex == index) {
                _hoveredIndex = null;
              }
            });
          },
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 120),
            curve: Curves.easeOut,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: backgroundColor,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              '○ $label',
              style: TextStyle(
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
                color: isSelected ? Colors.black87 : Colors.black54,
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCenterPanel() {
    switch (_selectedIndex) {
      case 0:
        return ProjectListPanel(
          projects: _projects,
          projectsLoading: _projectsLoading,
          selectedProject: _currentProjectSummary(),
          onRefresh: _loadProjects,
          onManage: _openProjectManager,
          onSelectProject: _setCurrentProject,
          onOpenFlow: _openFlowPage,
        );
      case 1:
        return ScriptGenerateForm(
          checkApiHealth: _checkApiHealthAndUpdate,
        );
      case 2:
        return VideoGenerateForm(
          checkApiHealth: _checkApiHealthAndUpdate,
          jobInProgress: _videoJobInProgress,
          onJobSubmitted: (jobId) {
            _latestJobId.value = jobId;
          },
        );
      case 3:
        return TitleGenerateForm(
          checkApiHealth: _checkApiHealthAndUpdate,
        );
      case 4:
        return MaterialsGenerateForm(
          checkApiHealth: _checkApiHealthAndUpdate,
        );
      case 5:
        return PonchiGenerateForm(
          checkApiHealth: _checkApiHealthAndUpdate,
        );
      case 6:
        return const VideoEditForm();
      case 7:
        return const DetailedEditForm();
      case 8:
        return const SettingsForm();
      case 9:
        return const AboutPanel();
      case 10:
        return FlowProjectPanel(
          selectedProject: _currentProjectSummary(),
          checkApiHealth: _checkApiHealthAndUpdate,
        );
      default:
        return PlaceholderPanel(title: _pages[_selectedIndex]);
    }
  }
}

class ProjectListPanel extends StatelessWidget {
  const ProjectListPanel({
    super.key,
    required this.projects,
    required this.projectsLoading,
    required this.selectedProject,
    required this.onRefresh,
    required this.onManage,
    required this.onSelectProject,
    required this.onOpenFlow,
  });

  final List<ProjectSummary> projects;
  final bool projectsLoading;
  final ProjectSummary? selectedProject;
  final Future<void> Function() onRefresh;
  final Future<void> Function() onManage;
  final Future<void> Function(String projectId) onSelectProject;
  final Future<void> Function() onOpenFlow;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Text('プロジェクト一覧', style: Theme.of(context).textTheme.headlineSmall),
            const Spacer(),
            OutlinedButton.icon(
              onPressed: projectsLoading ? null : onRefresh,
              icon: const Icon(Icons.refresh),
              label: const Text('更新'),
            ),
            const SizedBox(width: 8),
            ElevatedButton.icon(
              onPressed: onManage,
              icon: const Icon(Icons.settings),
              label: const Text('管理'),
            ),
          ],
        ),
        const SizedBox(height: 16),
        if (selectedProject?.isFlowProject ?? false)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: FilledButton.icon(
              onPressed: onOpenFlow,
              icon: const Icon(Icons.alt_route),
              label: const Text('選択中のAIフローを開く'),
            ),
          ),
        Expanded(
          child: Card(
            child: projectsLoading
                ? const Center(child: CircularProgressIndicator())
                : ValueListenableBuilder<String>(
                    valueListenable: ProjectState.currentProjectId,
                    builder: (context, currentProjectId, _) {
                      if (projects.isEmpty) {
                        return const Center(child: Text('プロジェクトがありません。'));
                      }
                      return ListView.separated(
                        padding: const EdgeInsets.all(12),
                        itemCount: projects.length,
                        separatorBuilder: (_, __) => const SizedBox(height: 8),
                        itemBuilder: (context, index) {
                          final project = projects[index];
                          final isCurrent = project.id == currentProjectId;
                          return Material(
                            color: Colors.transparent,
                            child: InkWell(
                              borderRadius: BorderRadius.circular(12),
                              onTap: () => onSelectProject(project.id),
                              child: Container(
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(12),
                                  color: isCurrent
                                      ? Theme.of(context)
                                          .colorScheme
                                          .primary
                                          .withOpacity(0.08)
                                      : Colors.grey.withOpacity(0.06),
                                  border: Border.all(
                                    color: isCurrent
                                        ? Theme.of(context).colorScheme.primary
                                        : Colors.grey.withOpacity(0.25),
                                  ),
                                ),
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 14,
                                  vertical: 12,
                                ),
                                child: Row(
                                  children: [
                                    Icon(
                                      isCurrent
                                          ? Icons.radio_button_checked
                                          : Icons.radio_button_unchecked,
                                      color: isCurrent
                                          ? Theme.of(context).colorScheme.primary
                                          : Colors.black45,
                                    ),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: Text(
                                        project.name,
                                        style: const TextStyle(
                                          fontWeight: FontWeight.w600,
                                          fontSize: 15,
                                        ),
                                      ),
                                    ),
                                    Text(
                                      '${project.id} / ${project.projectType == 'flow' ? 'AIフロー' : '通常'}',
                                      style: const TextStyle(color: Colors.black45),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      );
                    },
                  ),
          ),
        ),
      ],
    );
  }
}

class FlowProjectPanel extends StatefulWidget {
  const FlowProjectPanel({
    super.key,
    required this.selectedProject,
    required this.checkApiHealth,
  });

  final ProjectSummary? selectedProject;
  final ApiHealthCheck checkApiHealth;

  @override
  State<FlowProjectPanel> createState() => _FlowProjectPanelState();
}

class _FlowProjectPanelState extends State<FlowProjectPanel> {
  Map<String, String> _flowState = {
    for (final step in kFlowSteps) step.key: '未着手',
  };
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadFlowState();
  }

  @override
  void didUpdateWidget(covariant FlowProjectPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.selectedProject?.id != widget.selectedProject?.id) {
      _loadFlowState();
    }
  }

  Future<void> _loadFlowState() async {
    final selected = widget.selectedProject;
    if (selected == null || !selected.isFlowProject) {
      setState(() {
        _flowState = {for (final step in kFlowSteps) step.key: '未着手'};
        _error = null;
      });
      return;
    }
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final healthy = await widget.checkApiHealth();
      if (!healthy) {
        throw Exception('API サーバーに接続できません。');
      }
      final uri = ApiConfig.httpUri('/projects/${selected.id}/flow');
      final response = await http.get(uri).timeout(const Duration(seconds: 20));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(_formatApiFailureMessage(action: 'AIフロー状態の取得に失敗しました。', uri: uri, response: response));
      }
      final body = jsonDecode(response.body) as Map<String, dynamic>;
      final map = (body['flow_state'] as Map<String, dynamic>? ?? {});
      setState(() {
        _flowState = {
          for (final step in kFlowSteps)
            step.key: (map[step.key] as String? ?? '未着手'),
        };
      });
    } catch (error) {
      setState(() {
        _error = error.toString().replaceFirst('Exception: ', '');
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  Future<void> _updateStep(String step, String status) async {
    final selected = widget.selectedProject;
    if (selected == null || !selected.isFlowProject) {
      return;
    }
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final uri = ApiConfig.httpUri('/projects/${selected.id}/flow');
      final response = await http.put(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'step': step, 'status': status}),
      ).timeout(const Duration(seconds: 20));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(_formatApiFailureMessage(action: 'AIフロー状態の更新に失敗しました。', uri: uri, response: response));
      }
      setState(() {
        _flowState[step] = status;
      });
    } catch (error) {
      setState(() {
        _error = error.toString().replaceFirst('Exception: ', '');
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final selected = widget.selectedProject;
    if (selected == null) {
      return const Center(child: Text('プロジェクトを選択してください。'));
    }
    if (!selected.isFlowProject) {
      return Center(
        child: Text(
          '選択中のプロジェクト（${selected.name}）は通常作成です。\nAI作成（フロー型）プロジェクトを選択してください。',
          textAlign: TextAlign.center,
        ),
      );
    }
    return ListView(
      children: [
        Text('AI動画フロー: ${selected.name}', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 12),
        const Text('各工程は自動で進みません。内容を確認・編集し、手動で状態を更新してください。'),
        const SizedBox(height: 12),
        if (_error != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(_error!, style: const TextStyle(color: Colors.red)),
          ),
        if (_loading) const LinearProgressIndicator(),
        const SizedBox(height: 8),
        ...kFlowSteps.map((step) {
          final current = _flowState[step.key] ?? '未着手';
          return Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(step.label, style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),
                  DropdownButtonFormField<String>(
                    value: kFlowStatuses.contains(current) ? current : '未着手',
                    decoration: const InputDecoration(labelText: '進捗状態'),
                    items: kFlowStatuses
                        .map((status) => DropdownMenuItem(value: status, child: Text(status)))
                        .toList(),
                    onChanged: _loading
                        ? null
                        : (value) {
                            if (value == null || value == current) return;
                            _updateStep(step.key, value);
                          },
                  ),
                ],
              ),
            ),
          );
        }),
      ],
    );
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

class _ProjectManagerDialog extends StatefulWidget {
  const _ProjectManagerDialog({
    required this.projects,
    required this.onChanged,
  });

  final List<ProjectSummary> projects;
  final Future<void> Function() onChanged;

  @override
  State<_ProjectManagerDialog> createState() => _ProjectManagerDialogState();
}

class _ProjectManagerDialogState extends State<_ProjectManagerDialog> {
  bool _busy = false;

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> _createProject() async {
    final controller = TextEditingController();
    String selectedType = 'standard';
    final result = await showDialog<Map<String, String>>(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setLocalState) => AlertDialog(
          title: const Text('プロジェクト作成'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(controller: controller, decoration: const InputDecoration(labelText: '表示名')),
              const SizedBox(height: 12),
              DropdownButtonFormField<String>(
                value: selectedType,
                decoration: const InputDecoration(labelText: '作成方法'),
                items: const [
                  DropdownMenuItem(value: 'standard', child: Text('任意作成（従来通り）')),
                  DropdownMenuItem(value: 'flow', child: Text('AI作成（フロー型）')),
                ],
                onChanged: (value) {
                  if (value == null) return;
                  setLocalState(() {
                    selectedType = value;
                  });
                },
              ),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('キャンセル')),
            ElevatedButton(
              onPressed: () => Navigator.pop(context, {
                'name': controller.text.trim(),
                'project_type': selectedType,
              }),
              child: const Text('作成'),
            ),
          ],
        ),
      ),
    );
    final name = result?['name'] ?? '';
    final projectType = result?['project_type'] ?? 'standard';
    if (name.isEmpty) return;
    setState(() => _busy = true);
    try {
      final uri = ApiConfig.httpUri('/projects');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name, 'project_type': projectType}),
      );
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(
          _formatApiFailureMessage(
            action: 'プロジェクト作成に失敗しました。',
            uri: uri,
            response: response,
          ),
        );
      }
      await widget.onChanged();
    } catch (error) {
      _showErrorSnackBar(error.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _renameProject(ProjectSummary project) async {
    final controller = TextEditingController(text: project.name);
    final name = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('プロジェクト名変更'),
        content: TextField(controller: controller, decoration: const InputDecoration(labelText: '表示名')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('キャンセル')),
          ElevatedButton(onPressed: () => Navigator.pop(context, controller.text.trim()), child: const Text('保存')),
        ],
      ),
    );
    if (name == null || name.isEmpty) return;
    setState(() => _busy = true);
    try {
      final uri = ApiConfig.httpUri('/projects/${project.id}');
      final response = await http.put(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name}),
      );
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(
          _formatApiFailureMessage(
            action: 'プロジェクト名の変更に失敗しました。',
            uri: uri,
            response: response,
          ),
        );
      }
      await widget.onChanged();
    } catch (error) {
      _showErrorSnackBar(error.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _cloneProject(ProjectSummary project) async {
    setState(() => _busy = true);
    try {
      final uri = ApiConfig.httpUri('/projects/${project.id}/clone');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': '${project.name} copy'}),
      );
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(
          _formatApiFailureMessage(
            action: 'プロジェクト複製に失敗しました。',
            uri: uri,
            response: response,
          ),
        );
      }
      await widget.onChanged();
    } catch (error) {
      _showErrorSnackBar(error.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _deleteProject(ProjectSummary project) async {
    final ok = await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('削除確認'),
            content: Text('「${project.name}」を削除します。フォルダごと消去されます。よろしいですか？'),
            actions: [
              TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('キャンセル')),
              ElevatedButton(onPressed: () => Navigator.pop(context, true), child: const Text('削除')),
            ],
          ),
        ) ??
        false;
    if (!ok) return;
    setState(() => _busy = true);
    try {
      final uri = ApiConfig.httpUri('/projects/${project.id}');
      final response = await http.delete(uri);
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(
          _formatApiFailureMessage(
            action: 'プロジェクト削除に失敗しました。',
            uri: uri,
            response: response,
          ),
        );
      }
      if (ProjectState.currentProjectId.value == project.id) {
        ProjectState.currentProjectId.value = ProjectState.defaultProjectId;
      }
      await widget.onChanged();
    } catch (error) {
      _showErrorSnackBar(error.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('プロジェクト管理'),
      content: SizedBox(
        width: 520,
        height: 360,
        child: Column(
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: ElevatedButton.icon(
                onPressed: _busy ? null : _createProject,
                icon: const Icon(Icons.add),
                label: const Text('新規作成'),
              ),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ListView(
                children: widget.projects
                    .map(
                      (p) => ListTile(
                        title: Text(p.name),
                        subtitle: Text('${p.id} / ${p.projectType == 'flow' ? 'AIフロー' : '通常'}'),
                        trailing: Wrap(
                          spacing: 4,
                          children: [
                            IconButton(onPressed: _busy ? null : () => _renameProject(p), icon: const Icon(Icons.edit), tooltip: 'リネーム'),
                            IconButton(onPressed: _busy ? null : () => _cloneProject(p), icon: const Icon(Icons.copy), tooltip: '複製'),
                            IconButton(
                              onPressed: (_busy || p.id == ProjectState.undeletableProjectId) ? null : () => _deleteProject(p),
                              icon: const Icon(Icons.delete_outline),
                              tooltip: '削除',
                            ),
                          ],
                        ),
                      ),
                    )
                    .toList(),
              ),
            ),
          ],
        ),
      ),
      actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('閉じる'))],
    );
  }
}

class VideoGenerateForm extends StatefulWidget {
  const VideoGenerateForm({
    super.key,
    required this.checkApiHealth,
    required this.jobInProgress,
    this.onJobSubmitted,
  });

  final ApiHealthCheck checkApiHealth;
  final ValueNotifier<bool> jobInProgress;
  final ValueChanged<String?>? onJobSubmitted;

  @override
  State<VideoGenerateForm> createState() => _VideoGenerateFormState();
}

class ScriptGenerateForm extends StatefulWidget {
  const ScriptGenerateForm({
    super.key,
    required this.checkApiHealth,
  });

  final ApiHealthCheck checkApiHealth;

  @override
  State<ScriptGenerateForm> createState() => _ScriptGenerateFormState();
}

class _TemplateDialogResult {
  const _TemplateDialogResult(this.name, this.content);

  final String name;
  final String content;
}

class _ScriptGenerateFormState extends State<ScriptGenerateForm> {
  final _formKey = GlobalKey<FormState>();
  final _promptController = TextEditingController();
  final _outputController = TextEditingController(text: 'dialogue_input.txt');
  final _maxTokensController = TextEditingController(text: '20000');
  final _outputTextController = TextEditingController();
  late final InputPersistence _persistence;
  String _provider = 'Gemini';
  String _geminiModel = 'gemini-2.0-flash';
  String _chatGptModel = 'gpt-4.1-mini';
  String _claudeModel = 'claude-opus-4-5-20251101';
  String _template = '（テンプレなし）';
  bool _isSubmitting = false;
  static const Map<String, String> _defaultTemplates = {
    '（テンプレなし）': '',
    'ニュース原稿': '',
    '要約': '',
    'YouTube Shorts': '',
  };
  late Map<String, String> _templateContents =
      Map<String, String>.from(_defaultTemplates);
  late List<String> _templates = _defaultTemplates.keys.toList();

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('script_generate.', scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_promptController, 'prompt');
    _persistence.registerController(_outputController, 'output');
    _persistence.registerController(_maxTokensController, 'max_tokens');
    _persistence.registerController(_outputTextController, 'output_text');
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    final provider = await _persistence.readString('provider');
    final geminiModel = await _persistence.readString('gemini_model');
    final chatGptModel = await _persistence.readString('chatgpt_model');
    final claudeModel = await _persistence.readString('claude_model');
    final template = await _persistence.readString('template');
    final templateContents = await _readTemplates();
    final templateKeys = templateContents.keys.toList();
    if (!mounted) return;
    setState(() {
      _provider = provider ?? _provider;
      _geminiModel = geminiModel ?? _geminiModel;
      _chatGptModel = chatGptModel ?? _chatGptModel;
      _claudeModel = claudeModel ?? _claudeModel;
      _templateContents = templateContents;
      _templates = templateKeys;
      _template = template ?? _template;
      if (!_templates.contains(_template) && _templates.isNotEmpty) {
        _template = _templates.first;
      }
    });
  }

  @override
  void dispose() {
    _promptController.dispose();
    _outputController.dispose();
    _maxTokensController.dispose();
    _outputTextController.dispose();
    _persistence.dispose();
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
          Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<String>(
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
                    _persistence.setString('provider', value);
                  },
                ),
              ),
              const SizedBox(width: 12),
              Expanded(child: _buildModelDropdown()),
            ],
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
              _persistence.setString('template', value);
            },
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              OutlinedButton.icon(
                onPressed: _applySelectedTemplate,
                icon: const Icon(Icons.download),
                label: const Text('呼び出し'),
              ),
              OutlinedButton.icon(
                onPressed: _openSaveTemplateDialog,
                icon: const Icon(Icons.save),
                label: const Text('保存'),
              ),
              OutlinedButton.icon(
                onPressed: _openEditTemplateDialog,
                icon: const Icon(Icons.edit),
                label: const Text('編集'),
              ),
            ],
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

  Widget _buildModelDropdown() {
    switch (_provider) {
      case 'ChatGPT':
        return DropdownButtonFormField<String>(
          value: _chatGptModel,
          decoration: const InputDecoration(labelText: 'モデル'),
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
            _persistence.setString('chatgpt_model', value);
          },
        );
      case 'ClaudeCode':
        return DropdownButtonFormField<String>(
          value: _claudeModel,
          decoration: const InputDecoration(labelText: 'モデル'),
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
            _persistence.setString('claude_model', value);
          },
        );
      default:
        return DropdownButtonFormField<String>(
          value: _geminiModel,
          decoration: const InputDecoration(labelText: 'モデル'),
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
            _persistence.setString('gemini_model', value);
          },
        );
    }
  }

  Future<Map<String, String>> _readTemplates() async {
    final stored = await _persistence.readString('templates');
    if (stored == null || stored.isEmpty) {
      return Map<String, String>.from(_defaultTemplates);
    }
    try {
      final decoded = jsonDecode(stored);
      if (decoded is List) {
        final Map<String, String> result =
            Map<String, String>.from(_defaultTemplates);
        for (final entry in decoded) {
          if (entry is Map<String, dynamic>) {
            final name = entry['name'] as String?;
            final content = entry['content'] as String?;
            if (name != null) {
              result[name] = content ?? '';
            }
          }
        }
        return result;
      }
    } catch (_) {}
    return Map<String, String>.from(_defaultTemplates);
  }

  Future<void> _saveTemplates() async {
    final data = _templateContents.entries
        .map((entry) => {'name': entry.key, 'content': entry.value})
        .toList();
    await _persistence.setString('templates', jsonEncode(data));
  }

  void _applySelectedTemplate() {
    if (_template == '（テンプレなし）') {
      _showSnackBar('テンプレートが選択されていません。');
      return;
    }
    final content = _templateContents[_template];
    if (content == null || content.trim().isEmpty) {
      _showSnackBar('テンプレート内容がありません。');
      return;
    }
    _promptController.text = content;
    _showSnackBar('テンプレートを呼び出しました。');
  }

  Future<void> _openSaveTemplateDialog() async {
    final nameController = TextEditingController(text: _template);
    final contentController = TextEditingController(text: _promptController.text);
    final saved = await showDialog<_TemplateDialogResult>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('テンプレート保存'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: nameController,
                  decoration: const InputDecoration(labelText: 'テンプレート名'),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: contentController,
                  maxLines: 8,
                  decoration: const InputDecoration(labelText: 'テンプレート内容'),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('キャンセル'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop(
                  _TemplateDialogResult(
                    nameController.text.trim(),
                    contentController.text,
                  ),
                );
              },
              child: const Text('保存'),
            ),
          ],
        );
      },
    );
    nameController.dispose();
    contentController.dispose();
    if (saved == null) return;
    if (saved.name.isEmpty || saved.name == '（テンプレなし）') {
      _showSnackBar('テンプレート名を入力してください。');
      return;
    }
    setState(() {
      _templateContents[saved.name] = saved.content;
      _templates = _templateContents.keys.toList();
      _template = saved.name;
    });
    await _saveTemplates();
    _persistence.setString('template', saved.name);
    _showSnackBar('テンプレートを保存しました。');
  }

  Future<void> _openEditTemplateDialog() async {
    if (_template == '（テンプレなし）') {
      _showSnackBar('編集するテンプレートを選択してください。');
      return;
    }
    final currentContent = _templateContents[_template] ?? '';
    final nameController = TextEditingController(text: _template);
    final contentController = TextEditingController(text: currentContent);
    final updated = await showDialog<_TemplateDialogResult>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('テンプレート編集'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: nameController,
                  decoration: const InputDecoration(labelText: 'テンプレート名'),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: contentController,
                  maxLines: 8,
                  decoration: const InputDecoration(labelText: 'テンプレート内容'),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('キャンセル'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop(
                  _TemplateDialogResult(
                    nameController.text.trim(),
                    contentController.text,
                  ),
                );
              },
              child: const Text('更新'),
            ),
          ],
        );
      },
    );
    nameController.dispose();
    contentController.dispose();
    if (updated == null) return;
    if (updated.name.isEmpty || updated.name == '（テンプレなし）') {
      _showSnackBar('テンプレート名を入力してください。');
      return;
    }
    setState(() {
      if (updated.name != _template) {
        _templateContents.remove(_template);
      }
      _templateContents[updated.name] = updated.content;
      _templates = _templateContents.keys.toList();
      _template = updated.name;
    });
    await _saveTemplates();
    _persistence.setString('template', updated.name);
    _showSnackBar('テンプレートを更新しました。');
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
    final healthy = await widget.checkApiHealth();
    if (!healthy) {
      if (mounted) {
        setState(() {
          _isSubmitting = false;
        });
      }
      _showSnackBar('API サーバーに接続できません。');
      return;
    }
    try {
      final payload = {
        'api_key': apiKey,
        'provider': _provider,
        'prompt': _promptController.text,
        'model': _resolveModel(),
        'max_tokens': maxTokens,
        'project_id': ProjectState.currentProjectId.value,
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
      _showSnackBar('保存しました！');
    } catch (error) {
      _showSnackBar('保存に失敗しました: $error');
    }
  }
}

class TitleGenerateForm extends StatefulWidget {
  const TitleGenerateForm({
    super.key,
    required this.checkApiHealth,
  });

  final ApiHealthCheck checkApiHealth;

  @override
  State<TitleGenerateForm> createState() => _TitleGenerateFormState();
}

class _TitleGenerateFormState extends State<TitleGenerateForm> {
  final _formKey = GlobalKey<FormState>();
  final _scriptPathController = TextEditingController();
  final _countController = TextEditingController(text: '5');
  final _instructionsController = TextEditingController();
  final _outputController = TextEditingController();
  late final InputPersistence _persistence;
  String _provider = 'Gemini';
  String _geminiModel = 'gemini-2.0-flash';
  String _chatGptModel = 'gpt-4.1-mini';
  String _claudeModel = 'claude-opus-4-5-20251101';
  bool _isSubmitting = false;

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('title_generate.', scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_scriptPathController, 'script_path');
    _persistence.registerController(_countController, 'count');
    _persistence.registerController(_instructionsController, 'instructions');
    _persistence.registerController(_outputController, 'output_text');
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    final provider = await _persistence.readString('provider');
    final geminiModel = await _persistence.readString('gemini_model');
    final chatGptModel = await _persistence.readString('chatgpt_model');
    final claudeModel = await _persistence.readString('claude_model');
    if (!mounted) return;
    setState(() {
      _provider = provider ?? _provider;
      _geminiModel = geminiModel ?? _geminiModel;
      _chatGptModel = chatGptModel ?? _chatGptModel;
      _claudeModel = claudeModel ?? _claudeModel;
    });
  }

  @override
  void dispose() {
    _scriptPathController.dispose();
    _countController.dispose();
    _instructionsController.dispose();
    _outputController.dispose();
    _persistence.dispose();
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
          Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<String>(
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
                    _persistence.setString('provider', value);
                  },
                ),
              ),
              const SizedBox(width: 12),
              Expanded(child: _buildModelDropdown()),
            ],
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

  Widget _buildModelDropdown() {
    switch (_provider) {
      case 'ChatGPT':
        return DropdownButtonFormField<String>(
          value: _chatGptModel,
          decoration: const InputDecoration(labelText: 'モデル'),
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
            _persistence.setString('chatgpt_model', value);
          },
        );
      case 'ClaudeCode':
        return DropdownButtonFormField<String>(
          value: _claudeModel,
          decoration: const InputDecoration(labelText: 'モデル'),
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
            _persistence.setString('claude_model', value);
          },
        );
      default:
        return DropdownButtonFormField<String>(
          value: _geminiModel,
          decoration: const InputDecoration(labelText: 'モデル'),
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
            _persistence.setString('gemini_model', value);
          },
        );
    }
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
    final healthy = await widget.checkApiHealth();
    if (!healthy) {
      if (mounted) {
        setState(() {
          _isSubmitting = false;
        });
      }
      _showSnackBar('API サーバーに接続できません。');
      return;
    }
    try {
      final payload = {
        'api_key': apiKey,
        'provider': _provider,
        'script_path': _scriptPathController.text,
        'count': count,
        'extra': _instructionsController.text,
        'model': _resolveModel(),
        'project_id': ProjectState.currentProjectId.value,
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
  const MaterialsGenerateForm({
    super.key,
    required this.checkApiHealth,
  });

  final ApiHealthCheck checkApiHealth;

  @override
  State<MaterialsGenerateForm> createState() => _MaterialsGenerateFormState();
}

class _MaterialsGenerateFormState extends State<MaterialsGenerateForm> {
  final _formKey = GlobalKey<FormState>();
  final _modelController = TextEditingController(text: 'gemini-3-pro-image-preview');
  List<String> _availableModels = const ['gemini-3-pro-image-preview'];
  final _promptController = TextEditingController();
  final _outputController = TextEditingController();
  String? _generatedImagePath;
  Uint8List? _generatedImageBytes;
  String? _generatedImageMimeType;
  late final InputPersistence _persistence;
  bool _isSubmitting = false;

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('materials_generate.', scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_modelController, 'model');
    _persistence.registerController(_promptController, 'prompt');
    _persistence.registerController(_outputController, 'output_dir');
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    await _loadModelConfig();
  }

  Future<void> _loadModelConfig() async {
    try {
      final response = await http
          .get(ApiConfig.httpUri('/settings/ai-models'))
          .timeout(const Duration(seconds: 10));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        return;
      }
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final section = data['thumbnail'] as Map<String, dynamic>?;
      if (section == null) return;
      final models = (section['models'] as List<dynamic>? ?? [])
          .whereType<String>()
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();
      if (models.isEmpty || !mounted) {
        return;
      }
      final defaultModel = (section['default_model'] as String? ?? '').trim();
      setState(() {
        _availableModels = models;
        if (!_availableModels.contains(_modelController.text.trim())) {
          _modelController.text = _availableModels.contains(defaultModel)
              ? defaultModel
              : _availableModels.first;
        }
      });
    } catch (_) {
      // ignore
    }
  }

  @override
  void dispose() {
    _modelController.dispose();
    _promptController.dispose();
    _outputController.dispose();
    _persistence.dispose();
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
          DropdownButtonFormField<String>(
            value: _availableModels.contains(_modelController.text)
                ? _modelController.text
                : _availableModels.first,
            decoration: const InputDecoration(labelText: 'モデル（Gemini）'),
            items: _availableModels
                .map((model) => DropdownMenuItem(value: model, child: Text(model)))
                .toList(),
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _modelController.text = value;
              });
              _persistence.setString('model', value);
            },
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
          _buildGeneratedImagePreview(),
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
    final healthy = await widget.checkApiHealth();
    if (!healthy) {
      if (mounted) {
        setState(() {
          _isSubmitting = false;
        });
      }
      _showSnackBar('API サーバーに接続できません。');
      return;
    }
    try {
      final payload = {
        'api_key': apiKey,
        'prompt': _promptController.text,
        'model': _modelController.text,
        'output_dir': _outputController.text.trim().isEmpty
            ? null
            : _outputController.text.trim(),
        'project_id': ProjectState.currentProjectId.value,
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
        final mimeType = data['mime_type'] as String?;
        setState(() {
          _generatedImagePath = imagePath;
          _generatedImageMimeType = mimeType;
          _generatedImageBytes = (imageBase64 != null && imageBase64.isNotEmpty)
              ? base64Decode(imageBase64)
              : null;
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
    final text = _generatedImagePath?.trim() ?? '';
    if (text.isEmpty) {
      _showSnackBar('コピーする画像パスがありません。');
      return;
    }
    Clipboard.setData(ClipboardData(text: text));
    _showSnackBar('画像パスをコピーしました。');
  }

  Widget _buildGeneratedImagePreview() {
    final hasImageBytes = _generatedImageBytes != null && _generatedImageBytes!.isNotEmpty;
    final hasImagePath = (_generatedImagePath ?? '').isNotEmpty;
    Widget content;
    if (hasImageBytes) {
      content = Image.memory(_generatedImageBytes!, fit: BoxFit.contain);
    } else if (hasImagePath) {
      content = Image.file(
        File(_generatedImagePath!),
        fit: BoxFit.contain,
        errorBuilder: (_, __, ___) => const Text('画像の読み込みに失敗しました。'),
      );
    } else {
      content = const Text('生成された画像のプレビューがここに表示されます。');
    }

    final caption = hasImagePath
        ? _generatedImagePath!
        : (_generatedImageMimeType == null ? '' : 'MIME: ${_generatedImageMimeType!}');

    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(minHeight: 220, maxHeight: 360),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).dividerColor),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(child: Center(child: content)),
          if (caption.isNotEmpty) ...[
            const SizedBox(height: 8),
            SelectableText(
              caption,
              maxLines: 2,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ],
      ),
    );
  }
}

class PonchiGenerateForm extends StatefulWidget {
  const PonchiGenerateForm({
    super.key,
    required this.checkApiHealth,
  });

  final ApiHealthCheck checkApiHealth;

  @override
  State<PonchiGenerateForm> createState() => _PonchiGenerateFormState();
}

class _PonchiGenerateFormState extends State<PonchiGenerateForm> {
  final _formKey = GlobalKey<FormState>();
  final _srtController = TextEditingController();
  final _outputController = TextEditingController(text: 'ponchi_images');
  final _geminiModelController = TextEditingController(text: 'gemini-2.0-flash');
  final _outputTextController = TextEditingController();
  final List<_PonchiPreviewItem> _previewItems = [];
  late final InputPersistence _persistence;
  String _engine = 'Gemini';
  List<String> _ponchiModels = const ['gemini-2.0-flash'];
  bool _isSubmittingIdeas = false;
  bool _isSubmittingImages = false;

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('ponchi_generate.', scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_srtController, 'srt_path');
    _persistence.registerController(_outputController, 'output_dir');
    _persistence.registerController(_geminiModelController, 'gemini_model');
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    await _loadPonchiModelConfig();
  }

  Future<void> _loadPonchiModelConfig() async {
    try {
      final response = await http
          .get(ApiConfig.httpUri('/settings/ai-models'))
          .timeout(const Duration(seconds: 10));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        return;
      }
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final section = data['ponchi'] as Map<String, dynamic>?;
      if (section == null) return;
      final models = (section['models'] as List<dynamic>? ?? [])
          .whereType<String>()
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();
      if (models.isEmpty || !mounted) {
        return;
      }
      final defaultModel = (section['default_model'] as String? ?? '').trim();
      setState(() {
        _ponchiModels = models;
        if (!_ponchiModels.contains(_geminiModelController.text.trim())) {
          _geminiModelController.text = _ponchiModels.contains(defaultModel)
              ? defaultModel
              : _ponchiModels.first;
        }
      });
    } catch (_) {
      // ignore
    }
  }

  @override
  void dispose() {
    _srtController.dispose();
    _outputController.dispose();
    _geminiModelController.dispose();
    _outputTextController.dispose();
    _persistence.dispose();
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
          TextFormField(
            initialValue: 'Gemini',
            readOnly: true,
            decoration: const InputDecoration(labelText: '提案生成AI'),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _ponchiModels.contains(_geminiModelController.text)
                ? _geminiModelController.text
                : _ponchiModels.first,
            decoration: const InputDecoration(labelText: 'Gemini 提案モデル'),
            items: _ponchiModels
                .map((model) => DropdownMenuItem(value: model, child: Text(model)))
                .toList(),
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _geminiModelController.text = value;
              });
              _persistence.setString('gemini_model', value);
            },
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
                  setState(() {
                    _previewItems.clear();
                  });
                },
                icon: const Icon(Icons.clear),
                label: const Text('クリア'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text('生成結果', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          _buildPonchiPreviewGrid(),
          const SizedBox(height: 12),
          TextFormField(
            controller: _outputTextController,
            maxLines: 8,
            decoration: const InputDecoration(
              hintText: '生成結果のテキストがここに表示されます。',
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
    final healthy = await widget.checkApiHealth();
    if (!healthy) {
      if (mounted) {
        setState(() {
          _isSubmittingIdeas = false;
        });
      }
      _showSnackBar('API サーバーに接続できません。');
      return;
    }
    try {
      final payload = {
        'api_key': apiKey,
        'engine': _engine,
        'srt_path': _srtController.text,
        'output_dir': _outputController.text.trim().isEmpty
            ? null
            : _outputController.text.trim(),
        'gemini_model': _geminiModelController.text,
        'project_id': ProjectState.currentProjectId.value,
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


  Widget _buildPonchiPreviewGrid() {
    if (_previewItems.isEmpty) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          border: Border.all(color: Theme.of(context).dividerColor),
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Text('生成されたポンチ絵のプレビューがここに表示されます。'),
      );
    }
    return Column(
      children: _previewItems
          .map(
            (item) => Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                border: Border.all(color: Theme.of(context).dividerColor),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(item.title, style: Theme.of(context).textTheme.titleSmall),
                  if (item.subtitle.isNotEmpty) ...[
                    const SizedBox(height: 4),
                    Text(item.subtitle, style: Theme.of(context).textTheme.bodySmall),
                  ],
                  const SizedBox(height: 8),
                  SizedBox(
                    height: 180,
                    child: item.bytes != null
                        ? Image.memory(item.bytes!, fit: BoxFit.contain)
                        : Image.file(
                            File(item.path),
                            fit: BoxFit.contain,
                            errorBuilder: (_, __, ___) => const Text('画像を読み込めませんでした。'),
                          ),
                  ),
                ],
              ),
            ),
          )
          .toList(),
    );
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
    final healthy = await widget.checkApiHealth();
    if (!healthy) {
      if (mounted) {
        setState(() {
          _isSubmittingImages = false;
        });
      }
      _showSnackBar('API サーバーに接続できません。');
      return;
    }
    try {
      final payload = {
        'api_key': apiKey,
        'srt_path': _srtController.text,
        'output_dir': _outputController.text.trim(),
        'model': _geminiModelController.text.trim(),
        'project_id': ProjectState.currentProjectId.value,
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
        if (jsonPath.isNotEmpty) {
          buffer.writeln('JSON: $jsonPath');
        }
        final previews = <_PonchiPreviewItem>[];
        for (final item in items) {
          final map = item as Map<String, dynamic>;
          final imageName = (map['image'] as String? ?? '').trim();
          final imagePath = (outputDir.isNotEmpty && imageName.isNotEmpty)
              ? _joinPaths(outputDir, imageName)
              : imageName;
          final imageBase64 = (map['image_base64'] as String? ?? '').trim();
          Uint8List? bytes;
          if (imageBase64.isNotEmpty) {
            bytes = base64Decode(imageBase64);
          }
          previews.add(
            _PonchiPreviewItem(
              title: '${map['start']}〜${map['end']}',
              subtitle: (map['visual_suggestion'] as String? ?? '').trim(),
              path: imagePath,
              bytes: bytes,
            ),
          );
          buffer.writeln('${map['start']}〜${map['end']} | ${map['visual_suggestion']}');
        }
        setState(() {
          _outputTextController.text = buffer.toString();
          _previewItems
            ..clear()
            ..addAll(previews);
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

class _PonchiPreviewItem {
  const _PonchiPreviewItem({
    required this.title,
    required this.subtitle,
    required this.path,
    required this.bytes,
  });

  final String title;
  final String subtitle;
  final String path;
  final Uint8List? bytes;
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
  late final InputPersistence _persistence;
  String _searchProvider = 'Google';
  double _previewX = 0;
  double _previewY = 0;
  double _previewScale = 100;
  final List<Map<String, String>> _overlays = [];

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('video_edit.', scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_inputVideoController, 'input_video');
    _persistence.registerController(_outputVideoController, 'output_video');
    _persistence.registerController(_overlayImageController, 'overlay_image');
    _persistence.registerController(_startController, 'overlay_start');
    _persistence.registerController(_endController, 'overlay_end');
    _persistence.registerController(_xController, 'overlay_x');
    _persistence.registerController(_yController, 'overlay_y');
    _persistence.registerController(_widthController, 'overlay_w');
    _persistence.registerController(_heightController, 'overlay_h');
    _persistence.registerController(_opacityController, 'overlay_opacity');
    _persistence.registerController(_srtController, 'srt_path');
    _persistence.registerController(_imageOutputController, 'image_output');
    _persistence.registerController(_searchApiKeyController, 'search_api_key');
    _persistence.registerController(_defaultXController, 'default_x');
    _persistence.registerController(_defaultYController, 'default_y');
    _persistence.registerController(_defaultWController, 'default_w');
    _persistence.registerController(_defaultHController, 'default_h');
    _persistence.registerController(_defaultOpacityController, 'default_opacity');
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    final searchProvider = await _persistence.readString('search_provider');
    final previewX = await _persistence.readDouble('preview_x');
    final previewY = await _persistence.readDouble('preview_y');
    final previewScale = await _persistence.readDouble('preview_scale');
    if (!mounted) return;
    setState(() {
      _searchProvider = searchProvider ?? _searchProvider;
      _previewX = previewX ?? _previewX;
      _previewY = previewY ?? _previewY;
      _previewScale = previewScale ?? _previewScale;
    });
  }

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
    _persistence.dispose();
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
                      _persistence.setDouble('preview_x', value);
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
                      _persistence.setDouble('preview_y', value);
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
                      _persistence.setDouble('preview_scale', value);
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
              _persistence.setString('search_provider', value);
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
  late final InputPersistence _persistence;
  bool _audioEnabled = true;
  String _autoSaveStatus = '未設定';

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('detailed_edit.', scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_projectNameController, 'project_name');
    _persistence.registerController(_resolutionController, 'resolution');
    _persistence.registerController(_fpsController, 'fps');
    _persistence.registerController(_mainVideoController, 'main_video');
    _persistence.registerController(_clipInController, 'clip_in');
    _persistence.registerController(_clipOutController, 'clip_out');
    _persistence.registerController(_overlayImageController, 'overlay_image');
    _persistence.registerController(_overlayStartController, 'overlay_start');
    _persistence.registerController(_overlayEndController, 'overlay_end');
    _persistence.registerController(_overlayXController, 'overlay_x');
    _persistence.registerController(_overlayYController, 'overlay_y');
    _persistence.registerController(_overlayOpacityController, 'overlay_opacity');
    _persistence.registerController(_exportPathController, 'export_path');
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    final audioEnabled = await _persistence.readBool('audio_enabled');
    if (!mounted) return;
    setState(() {
      _audioEnabled = audioEnabled ?? _audioEnabled;
    });
  }

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
    _persistence.dispose();
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
                      onPressed: () {
                        _showSnackBar('保存しました！');
                      },
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
                    _persistence.setBool('audio_enabled', value);
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

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
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
  final _voicevoxUrlController = TextEditingController();
  late final InputPersistence _persistence;
  late final InputPersistence _voicevoxPersistence;

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('settings.');
    _voicevoxPersistence = InputPersistence('video_generate.');
    _backendController.text = ApiConfig.baseUrl.value;
    _backendController.addListener(() {
      ApiConfig.baseUrl.value = _backendController.text.trim();
      _persistence.setString('backend_url', _backendController.text.trim());
    });
    _geminiController.text = ApiKeys.gemini.value;
    _openAiController.text = ApiKeys.openAi.value;
    _claudeController.text = ApiKeys.claude.value;
    _voicevoxUrlController.text = VoicevoxConfig.baseUrl.value;
    _geminiController.addListener(() {
      ApiKeys.gemini.value = _geminiController.text.trim();
      _persistence.setString('gemini_key', _geminiController.text.trim());
    });
    _openAiController.addListener(() {
      ApiKeys.openAi.value = _openAiController.text.trim();
      _persistence.setString('openai_key', _openAiController.text.trim());
    });
    _claudeController.addListener(() {
      ApiKeys.claude.value = _claudeController.text.trim();
      _persistence.setString('claude_key', _claudeController.text.trim());
    });
    _voicevoxUrlController.addListener(() {
      VoicevoxConfig.baseUrl.value = _voicevoxUrlController.text.trim();
      _voicevoxPersistence.setString(
        'vv_url',
        _voicevoxUrlController.text.trim(),
      );
    });
    _persistence.registerController(_backendController, 'backend_url');
    _persistence.registerController(_geminiController, 'gemini_key');
    _persistence.registerController(_openAiController, 'openai_key');
    _persistence.registerController(_claudeController, 'claude_key');
    _voicevoxPersistence.registerController(_voicevoxUrlController, 'vv_url');
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    await _voicevoxPersistence.init();
    ApiConfig.baseUrl.value = _backendController.text.trim();
    ApiKeys.gemini.value = _geminiController.text.trim();
    ApiKeys.openAi.value = _openAiController.text.trim();
    ApiKeys.claude.value = _claudeController.text.trim();
    VoicevoxConfig.baseUrl.value = _voicevoxUrlController.text.trim();
  }

  @override
  void dispose() {
    _backendController.dispose();
    _geminiController.dispose();
    _openAiController.dispose();
    _claudeController.dispose();
    _voicevoxUrlController.dispose();
    _persistence.dispose();
    _voicevoxPersistence.dispose();
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
        const SizedBox(height: 12),
        TextFormField(
          controller: _voicevoxUrlController,
          decoration: const InputDecoration(
            labelText: 'VOICEVOX エンジンURL',
            helperText: '例: http://127.0.0.1:50021',
          ),
        ),
        const SizedBox(height: 16),
        ElevatedButton.icon(
          onPressed: () {
            _showSnackBar('保存しました！');
          },
          icon: const Icon(Icons.save),
          label: const Text('保存'),
        ),
      ],
    );
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }
}

class _VideoGenerateFormState extends State<VideoGenerateForm> {
  static const String _prefsPrefix = 'video_generate.';
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
  late final InputPersistence _persistence;
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
  List<String> _voicevoxSpeakers = [];
  bool _voicevoxSpeakersLoading = false;
  String? _voicevoxSpeakersError;
  late final VoidCallback _voicevoxUrlListener;
  late final VoidCallback _projectListener;

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence(_prefsPrefix, scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_scriptController, 'script');
    _persistence.registerController(_imageListController, 'images');
    _persistence.registerController(_outputController, 'output');
    _persistence.registerController(_widthController, 'width');
    _persistence.registerController(_heightController, 'height');
    _persistence.registerController(_fpsController, 'fps');
    _persistence.registerController(_voiceController, 'voice');
    _persistence.registerController(_voicevoxUrlController, 'vv_url');
    _persistence.registerController(_voicevoxRotationController, 'vv_rotation');
    _persistence.registerController(_voicevoxCasterController, 'vv_caster');
    _persistence.registerController(_voicevoxAnalystController, 'vv_analyst');
    _persistence.registerController(
      _captionFontSizeController,
      'caption_font_size',
    );
    _persistence.registerController(_captionAlphaController, 'caption_alpha');
    _persistence.registerController(
      _captionTextColorController,
      'caption_text_color',
    );
    _persistence.registerController(
      _speakerFontSizeController,
      'speaker_font_size',
    );
    _persistence.registerController(
      _captionMaxCharsController,
      'caption_max_chars',
    );
    _persistence.registerController(
      _captionBoxHeightController,
      'caption_box_height',
    );
    _persistence.registerController(_bgmController, 'bgm_path');
    _voicevoxUrlListener = () {
      final value = VoicevoxConfig.baseUrl.value.trim();
      if (_voicevoxUrlController.text.trim() == value) {
        return;
      }
      _voicevoxUrlController.text = value;
      if (!mounted) return;
      setState(() {
        _voicevoxSpeakers = [];
        _voicevoxSpeakersError = null;
      });
    };
    VoicevoxConfig.baseUrl.addListener(_voicevoxUrlListener);
    _projectListener = () {
      _loadProjectSettings();
    };
    ProjectState.currentProjectId.addListener(_projectListener);
    _loadSavedValues();
  }

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
    VoicevoxConfig.baseUrl.removeListener(_voicevoxUrlListener);
    ProjectState.currentProjectId.removeListener(_projectListener);
    _persistence.dispose();
    super.dispose();
  }

  Future<void> _loadSavedValues() async {
    await _persistence.init();
    final useBgm = await _persistence.readBool('use_bgm');
    final bgmGainDb = await _persistence.readDouble('bgm_gain_db');
    final ttsEngine = await _persistence.readString('tts_engine');
    final voicevoxMode = await _persistence.readString('vv_mode');
    final voicevoxSpeed = await _persistence.readDouble('vv_speed');
    final captionBoxEnabled =
        await _persistence.readBool('caption_box_enabled');
    final bgOffStyle = await _persistence.readString('bg_off_style');
    if (!mounted) return;
    setState(() {
      _useBgm = useBgm ?? _useBgm;
      _bgmGainDb = bgmGainDb ?? _bgmGainDb;
      _ttsEngine = ttsEngine ?? _ttsEngine;
      _voicevoxMode = voicevoxMode ?? _voicevoxMode;
      _voicevoxSpeed = voicevoxSpeed ?? _voicevoxSpeed;
      _captionBoxEnabled = captionBoxEnabled ?? _captionBoxEnabled;
      _bgOffStyle = bgOffStyle ?? _bgOffStyle;
    });
    VoicevoxConfig.baseUrl.value = _voicevoxUrlController.text.trim();
    await _loadProjectSettings();
    if (_ttsEngine == 'VOICEVOX') {
      await _fetchVoicevoxSpeakers();
    }
  }

  Future<void> _loadProjectSettings() async {
    try {
      final projectId = ProjectState.currentProjectId.value;
      final response = await http.get(ApiConfig.httpUri('/projects/$projectId/settings'));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        return;
      }
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final settings = data['settings'] as Map<String, dynamic>? ?? {};
      final video = settings['video_generate'] as Map<String, dynamic>? ?? {};
      if (!mounted) return;
      setState(() {
        _captionFontSizeController.text = '${video['caption_font_size'] ?? _captionFontSizeController.text}';
        _speakerFontSizeController.text = '${video['speaker_font_size'] ?? _speakerFontSizeController.text}';
        _captionMaxCharsController.text = '${video['caption_max_chars'] ?? _captionMaxCharsController.text}';
        _captionBoxHeightController.text = '${video['caption_box_height'] ?? _captionBoxHeightController.text}';
        _captionAlphaController.text = '${video['caption_box_alpha'] ?? _captionAlphaController.text}';
        _captionTextColorController.text = '${video['caption_text_color'] ?? _captionTextColorController.text}';
      });
    } catch (_) {
      return;
    }
  }

  Future<void> _saveProjectSettings() async {
    final projectId = ProjectState.currentProjectId.value;
    final settings = {
      'video_generate': {
        'caption_font_size': int.tryParse(_captionFontSizeController.text) ?? 36,
        'speaker_font_size': int.tryParse(_speakerFontSizeController.text) ?? 30,
        'caption_max_chars': int.tryParse(_captionMaxCharsController.text) ?? 22,
        'caption_box_height': int.tryParse(_captionBoxHeightController.text) ?? 420,
        'caption_box_alpha': int.tryParse(_captionAlphaController.text) ?? 170,
        'caption_text_color': _captionTextColorController.text,
      },
    };
    await http.put(
      ApiConfig.httpUri('/projects/$projectId/settings'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'settings': settings}),
    );
  }

  List<String> _voicevoxSpeakerOptions(String current) {
    final options = [..._voicevoxSpeakers];
    final trimmed = current.trim();
    if (trimmed.isNotEmpty && !options.contains(trimmed)) {
      options.insert(0, trimmed);
    }
    return options;
  }

  Future<void> _fetchVoicevoxSpeakers() async {
    final baseUrl = _voicevoxUrlController.text.trim();
    if (baseUrl.isEmpty) {
      setState(() {
        _voicevoxSpeakers = [];
        _voicevoxSpeakersError = 'VOICEVOX エンジンURLを入力してください。';
      });
      return;
    }

    setState(() {
      _voicevoxSpeakersLoading = true;
      _voicevoxSpeakersError = null;
    });

    try {
      final uri = ApiConfig.httpUri('/voicevox/speakers')
          .replace(queryParameters: {'base_url': baseUrl});
      final response = await http
          .get(uri)
          .timeout(const Duration(seconds: 10));
      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final speakers = (data['speakers'] as List<dynamic>? ?? [])
            .whereType<String>()
            .toList();
        setState(() {
          _voicevoxSpeakers = speakers;
          _voicevoxSpeakersError =
              speakers.isEmpty ? '話者一覧が空でした。' : null;
        });
        if (speakers.isNotEmpty) {
          if (_voicevoxCasterController.text.trim().isEmpty) {
            _voicevoxCasterController.text = speakers.first;
          }
          if (_voicevoxAnalystController.text.trim().isEmpty) {
            _voicevoxAnalystController.text = speakers.first;
          }
        }
      } else if (response.statusCode == 404) {
        final fallbackSpeakers =
            await _fetchVoicevoxSpeakersDirect(baseUrl);
        if (fallbackSpeakers.isNotEmpty) {
          setState(() {
            _voicevoxSpeakers = fallbackSpeakers;
            _voicevoxSpeakersError = null;
          });
          if (_voicevoxCasterController.text.trim().isEmpty) {
            _voicevoxCasterController.text = fallbackSpeakers.first;
          }
          if (_voicevoxAnalystController.text.trim().isEmpty) {
            _voicevoxAnalystController.text = fallbackSpeakers.first;
          }
          return;
        }
        setState(() {
          final apiBase = ApiConfig.baseUrl.value.trim();
          final engineUrl = baseUrl;
          _voicevoxSpeakersError =
              '取得に失敗しました (404)。APIサーバURLとVOICEVOX エンジンURLを確認してください。\n'
              'APIサーバURL例: http://127.0.0.1:8000\n'
              '現在のAPIサーバURL: $apiBase\n'
              'VOICEVOX エンジンURL: $engineUrl';
        });
      } else {
        setState(() {
          _voicevoxSpeakersError =
              '取得に失敗しました (${response.statusCode})';
        });
      }
    } on TimeoutException {
      setState(() {
        _voicevoxSpeakersError = '取得がタイムアウトしました。';
      });
    } catch (error) {
      setState(() {
        _voicevoxSpeakersError = '取得に失敗しました: $error';
      });
    } finally {
      if (!mounted) return;
      setState(() {
        _voicevoxSpeakersLoading = false;
      });
    }
  }

  Future<List<String>> _fetchVoicevoxSpeakersDirect(String baseUrl) async {
    final directUri = _normalizeVoicevoxSpeakersUri(baseUrl);
    final response = await http
        .get(directUri)
        .timeout(const Duration(seconds: 10));
    if (response.statusCode < 200 || response.statusCode >= 300) {
      return [];
    }
    final data = jsonDecode(response.body);
    return _extractVoicevoxSpeakerNames(data);
  }

  Widget _buildVoicevoxSpeakerDropdown({
    required String label,
    required TextEditingController controller,
    required String persistenceKey,
  }) {
    final options = _voicevoxSpeakerOptions(controller.text);
    final current =
        options.contains(controller.text) ? controller.text : null;
    return DropdownButtonFormField<String>(
      value: current,
      decoration: InputDecoration(labelText: label),
      items: options
          .map((speaker) => DropdownMenuItem(
                value: speaker,
                child: Text(speaker),
              ))
          .toList(),
      hint: const Text('話者一覧を取得してください'),
      onChanged: options.isEmpty
          ? null
          : (value) {
              if (value == null) return;
              setState(() {
                controller.text = value;
              });
              _persistence.setString(persistenceKey, value);
            },
    );
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
              _persistence.setBool('use_bgm', value);
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
                        _persistence.setDouble('bgm_gain_db', value);
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
              _persistence.setString('tts_engine', value);
              if (value == 'VOICEVOX') {
                _fetchVoicevoxSpeakers();
              }
            },
          ),
          const SizedBox(height: 12),
          if (_ttsEngine == 'Gemini')
            TextFormField(
              controller: _voiceController,
              decoration: const InputDecoration(labelText: 'Gemini 音声'),
            ),
          if (_ttsEngine == 'VOICEVOX') ...[
            Row(
              children: [
                ElevatedButton.icon(
                  onPressed:
                      _voicevoxSpeakersLoading ? null : _fetchVoicevoxSpeakers,
                  icon: const Icon(Icons.refresh),
                  label: const Text('話者一覧を取得'),
                ),
                if (_voicevoxSpeakersLoading) ...[
                  const SizedBox(width: 12),
                  const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                ],
              ],
            ),
            if (_voicevoxSpeakersError != null) ...[
              const SizedBox(height: 8),
              Text(
                _voicevoxSpeakersError!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _voicevoxMode,
                    decoration: const InputDecoration(labelText: '話者モード'),
                    items: const [
                      DropdownMenuItem(
                        value: 'ローテーション',
                        child: Text('ローテーション'),
                      ),
                      DropdownMenuItem(value: '2人対談', child: Text('2人対談')),
                    ],
                    onChanged: (value) {
                      if (value == null) return;
                      setState(() {
                        _voicevoxMode = value;
                      });
                      _persistence.setString('vv_mode', value);
                    },
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: TextFormField(
                    controller: _voicevoxRotationController,
                    decoration:
                        const InputDecoration(labelText: 'ローテーション話者(カンマ)'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _buildVoicevoxSpeakerDropdown(
                    label: 'キャスター話者',
                    controller: _voicevoxCasterController,
                    persistenceKey: 'vv_caster',
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildVoicevoxSpeakerDropdown(
                    label: 'アナリスト話者',
                    controller: _voicevoxAnalystController,
                    persistenceKey: 'vv_analyst',
                  ),
                ),
              ],
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
                    _persistence.setDouble('vv_speed', value);
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
          Row(
            children: [
              Expanded(
                child: TextFormField(
                  controller: _captionFontSizeController,
                  decoration: const InputDecoration(labelText: '字幕フォントサイズ'),
                  keyboardType: TextInputType.number,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: TextFormField(
                  controller: _captionAlphaController,
                  decoration:
                      const InputDecoration(labelText: '字幕背景の透明度(alpha 0-255)'),
                  keyboardType: TextInputType.number,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<String>(
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
                    _persistence.setString('bg_off_style', value);
                  },
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: TextFormField(
                  controller: _captionTextColorController,
                  decoration: const InputDecoration(labelText: '字幕文字色（#RRGGBB）'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: TextFormField(
                  controller: _speakerFontSizeController,
                  decoration: const InputDecoration(labelText: '話者名フォントサイズ'),
                  keyboardType: TextInputType.number,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: TextFormField(
                  controller: _captionMaxCharsController,
                  decoration: const InputDecoration(labelText: '1行あたり最大文字数'),
                  keyboardType: TextInputType.number,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          SwitchListTile(
            value: _captionBoxEnabled,
            title: const Text('字幕背景（黒幕）を表示する（固定高さ）'),
            onChanged: (value) {
              setState(() {
                _captionBoxEnabled = value;
              });
              _persistence.setBool('caption_box_enabled', value);
            },
          ),
          TextFormField(
            controller: _captionBoxHeightController,
            decoration: const InputDecoration(labelText: '字幕背景の高さ(px, 固定)'),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 12),
          ValueListenableBuilder<bool>(
            valueListenable: widget.jobInProgress,
            builder: (context, jobInProgress, _) {
              final isBusy = _isSubmitting || jobInProgress;
              return Row(
                children: [
                  ElevatedButton.icon(
                    onPressed: isBusy ? null : _submitJob,
                    icon: const Icon(Icons.play_arrow),
                    label: Text(isBusy ? '送信中...' : '動画を生成する'),
                  ),
                  const SizedBox(width: 16),
                  Expanded(child: Text('状態: $_statusMessage')),
                ],
              );
            },
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
    widget.jobInProgress.value = true;

    final isHealthy = await widget.checkApiHealth();
    if (!isHealthy) {
      setState(() {
        _statusMessage =
            'Error: API サーバーに接続できません。バックエンドURLを確認するか、サーバーを起動してください。';
        _isSubmitting = false;
      });
      widget.jobInProgress.value = false;
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
      'project_id': ProjectState.currentProjectId.value,
    };

    try {
      await _saveProjectSettings();
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
        if (jobId == null || jobId.isEmpty) {
          setState(() {
            _statusMessage = 'Error: job_id が取得できませんでした。';
          });
          widget.jobInProgress.value = false;
        } else {
          setState(() {
            _jobId = jobId;
            _statusMessage = 'Submitted';
          });
          widget.onJobSubmitted?.call(jobId);
        }
      } else {
        final responseBody = response.body.isEmpty ? '' : ' ${response.body}';
        setState(() {
          _statusMessage = 'Error: ${response.statusCode}$responseBody';
        });
        widget.jobInProgress.value = false;
      }
    } on TimeoutException {
      setState(() {
        _statusMessage = 'Error: request timed out (30s)';
      });
      widget.jobInProgress.value = false;
    } catch (error) {
      setState(() {
        _statusMessage = 'Error: $error';
      });
      widget.jobInProgress.value = false;
    } finally {
      setState(() {
        _isSubmitting = false;
      });
    }
  }

}

class LogPanel extends StatefulWidget {
  const LogPanel({
    super.key,
    required this.pageName,
    this.latestJobId,
    this.jobInProgress,
  });

  final String pageName;
  final ValueListenable<String?>? latestJobId;
  final ValueNotifier<bool>? jobInProgress;

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
  double? _progress;
  String _progressLabel = '';
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
          if (_progress != null) ...[
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
              child: Text(
                _progressLabel,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 0),
              child: LinearProgressIndicator(value: _progress),
            ),
            if (_eta != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 4, 16, 0),
                child: Text('ETA: $_eta'),
              ),
          ],
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
      _progress = null;
      _progressLabel = '';
      _eta = null;
      _channel = channel;
    });
    _setJobInProgress(true);
    _addLog('Connecting to $jobId ...');

    _subscription = channel.stream.listen(
      (event) {
        _handleSocketEvent(event);
      },
      onError: (error) {
        _addLog('WebSocket error: $error', level: _LogLevel.error);
        _setJobInProgress(false);
      },
      onDone: () {
        _addLog('WebSocket closed', level: _LogLevel.warning);
        _setJobInProgress(false);
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
        final nextEta = etaSeconds == null ? null : '${etaSeconds.toString()} sec';
        final shouldUpdateProgress =
            _progress == null || (progress - _progress!).abs() >= 0.005;
        final shouldUpdateEta = nextEta != _eta;
        if (shouldUpdateProgress || shouldUpdateEta) {
          setState(() {
            if (shouldUpdateProgress) {
              _progress = progress;
              _progressLabel = '現在 ${(progress * 100).toStringAsFixed(1)}%';
            }
            if (shouldUpdateEta) {
              _eta = nextEta;
            }
          });
        }
        return;
      case 'error':
        _addLog('エラー: ${payload['message']}', level: _LogLevel.error);
        _setJobInProgress(false);
        return;
      case 'completed':
        _addLog('完了: ${jsonEncode(payload['result'])}', level: _LogLevel.success);
        _setJobInProgress(false);
        return;
      case 'log':
      default:
        _addLog('${payload['message']}');
        return;
    }
  }

  void _setJobInProgress(bool value) {
    final notifier = widget.jobInProgress;
    if (notifier == null || notifier.value == value) return;
    notifier.value = value;
  }

  void _addLog(String message, { _LogLevel level = _LogLevel.info }) {
    setState(() {
      _logs.add(_LogEntry(timestamp: DateTime.now(), message: message, level: level));
      const maxLogs = 200;
      if (_logs.length > maxLogs) {
        _logs.removeRange(0, _logs.length - maxLogs);
      }
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
