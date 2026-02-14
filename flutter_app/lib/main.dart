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
import 'package:video_player/video_player.dart';
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
  static final ValueNotifier<String> currentProjectType = ValueNotifier<String>('standard');
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

enum AppLogLevel { info, warn, error }

class AppLogEntry {
  AppLogEntry({
    required this.timestamp,
    required this.level,
    required this.message,
  });

  final DateTime timestamp;
  final AppLogLevel level;
  final String message;
}

class AppLogger {
  AppLogger._();

  static const int _maxLogs = 2000;
  static final List<AppLogEntry> _entries = <AppLogEntry>[];
  static final ValueNotifier<int> revision = ValueNotifier<int>(0);

  static List<AppLogEntry> get entries => List<AppLogEntry>.unmodifiable(_entries);

  static void info(String message) => _add(AppLogLevel.info, message);
  static void warn(String message) => _add(AppLogLevel.warn, message);

  static void error(String message, {Object? error, StackTrace? stackTrace}) {
    final details = StringBuffer(message);
    if (error != null) {
      details.write(' | error=$error');
    }
    if (stackTrace != null) {
      final lines = stackTrace.toString().split('\n');
      if (lines.isNotEmpty) {
        details.write(' | st=${lines.first}');
      }
    }
    _add(AppLogLevel.error, details.toString());
  }

  static void clear() {
    _entries.clear();
    revision.value += 1;
  }

  static void _add(AppLogLevel level, String message) {
    _entries.add(AppLogEntry(timestamp: DateTime.now(), level: level, message: message));
    if (_entries.length > _maxLogs) {
      _entries.removeRange(0, _entries.length - _maxLogs);
    }
    revision.value += 1;
  }
}

const List<ProjectFlowStep> kFlowSteps = [
  ProjectFlowStep(key: 'script', label: '台本作成（AI + 人間修正）'),
  ProjectFlowStep(key: 'base_video', label: '動画作成（ベース動画生成）'),
  ProjectFlowStep(key: 'title_description', label: '動画タイトル・説明文作成（AI）'),
  ProjectFlowStep(key: 'thumbnail', label: 'サムネイル作成（AI）'),
  ProjectFlowStep(key: 'ponchi', label: 'ポンチ絵（補足ビジュアル）案の作成'),
  ProjectFlowStep(key: 'final_edit', label: '動画編集（最終編集）'),
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
    final baseTextTheme = ThemeData(useMaterial3: true).textTheme.apply(fontSizeFactor: 0.85);
    return MaterialApp(
      title: appTitle,
      theme: ThemeData(
        colorScheme: colorScheme,
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF6F4FF),
        visualDensity: VisualDensity.compact,
        cardTheme: CardThemeData(
          elevation: 3,
          shadowColor: colorScheme.primary.withOpacity(0.2),
          surfaceTintColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          isDense: true,
          fillColor: Colors.white,
          hintStyle: TextStyle(
            fontSize: 12,
            height: 1.2,
            color: Colors.grey.shade600,
          ),
          labelStyle: const TextStyle(
            fontSize: 12,
            height: 1.2,
          ),
          floatingLabelStyle: const TextStyle(
            fontSize: 12,
            height: 1.1,
            fontWeight: FontWeight.w600,
          ),
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
          contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            backgroundColor: colorScheme.primary,
            foregroundColor: Colors.white,
            elevation: 2,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
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
        textTheme: baseTextTheme.copyWith(
          headlineSmall: baseTextTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700),
          titleMedium: baseTextTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
          bodyLarge: baseTextTheme.bodyLarge?.copyWith(fontSize: 13, height: 1.2),
          bodyMedium: baseTextTheme.bodyMedium?.copyWith(fontSize: 12, height: 1.2),
          bodySmall: baseTextTheme.bodySmall?.copyWith(fontSize: 11, height: 1.15),
        ),
        snackBarTheme: SnackBarThemeData(
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
        scrollbarTheme: const ScrollbarThemeData(
          thumbVisibility: MaterialStatePropertyAll(true),
          trackVisibility: MaterialStatePropertyAll(true),
          thickness: MaterialStatePropertyAll(10),
          radius: Radius.circular(8),
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
    '画像収集',
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
  Map<String, String> _currentFlowState = {for (final step in kFlowSteps) step.key: '未着手'};
  bool _flowStateLoading = false;
  double _bottomPanelRatio = 0.25;
  bool _bottomPanelCollapsed = false;

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

    _apiServerPort = await _selectApiServerPort();
    _updateApiBaseUrlForPort(_apiServerPort);

    final bundledExecutable = _findBundledApiServerExecutable();
    if (bundledExecutable != null) {
      try {
        _apiServerLaunchCommand =
            '$bundledExecutable --host 127.0.0.1 --port $_apiServerPort';
        final process = await Process.start(
          bundledExecutable,
          [
            '--host',
            '127.0.0.1',
            '--port',
            '$_apiServerPort',
          ],
          workingDirectory: File(bundledExecutable).parent.path,
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
            .listen((line) {
          if (_looksLikeApiErrorLine(line)) {
            _appendApiServerError(line);
            return;
          }
          AppLogger.info('APIログ: $line');
        });
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

  String? _findBundledApiServerExecutable() {
    final executableName =
        Platform.isWindows ? 'movie_maker_api.exe' : 'movie_maker_api';
    final exeDir = File(Platform.resolvedExecutable).parent.path;
    final candidates = [
      _joinFilePath([exeDir, 'api', executableName]),
      _joinFilePath([exeDir, executableName]),
    ];
    for (final candidate in candidates) {
      if (File(candidate).existsSync()) {
        return candidate;
      }
    }
    return null;
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
      _syncCurrentProjectType();
      await _loadCurrentProjectFlowState();
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
    _syncCurrentProjectType();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('project.current_id', normalized);
    await _loadCurrentProjectFlowState();
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

  void _syncCurrentProjectType() {
    final current = _currentProjectSummary();
    final nextType = current?.projectType == 'flow' ? 'flow' : 'standard';
    if (ProjectState.currentProjectType.value != nextType) {
      ProjectState.currentProjectType.value = nextType;
    }
  }

  bool get _isFlowProjectSelected => _currentProjectSummary()?.isFlowProject ?? false;

  String _pageLabel(int index) {
    if (!_isFlowProjectSelected) {
      return _pages[index];
    }
    switch (index) {
      case 1:
        return '台本作成';
      case 2:
        return 'ベース動画作成';
      case 3:
        return 'タイトル・説明';
      case 4:
        return 'サムネイル';
      case 5:
        return 'ポンチ絵案';
      case 6:
        return '最終編集';
      case 11:
        return 'AIフロー進捗';
      default:
        return _pages[index];
    }
  }

  String? _flowStepKeyForMenuIndex(int index) {
    return switch (index) {
      1 => 'script',
      2 => 'base_video',
      3 => 'title_description',
      4 => 'thumbnail',
      5 => 'ponchi',
      6 => 'final_edit',
      _ => null,
    };
  }

  Future<void> _loadCurrentProjectFlowState() async {
    if (!_isFlowProjectSelected) {
      if (mounted) {
        setState(() {
          _currentFlowState = {for (final step in kFlowSteps) step.key: '未着手'};
          _flowStateLoading = false;
        });
      }
      return;
    }
    final project = _currentProjectSummary();
    if (project == null) {
      return;
    }
    if (mounted) {
      setState(() {
        _flowStateLoading = true;
      });
    }
    try {
      final uri = ApiConfig.httpUri('/projects/${project.id}/flow');
      final response = await http.get(uri).timeout(const Duration(seconds: 20));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        return;
      }
      final body = jsonDecode(response.body) as Map<String, dynamic>;
      final flow = body['flow_state'] as Map<String, dynamic>? ?? {};
      if (!mounted) return;
      setState(() {
        _currentFlowState = {
          for (final step in kFlowSteps)
            step.key: (flow[step.key] as String? ?? '未着手'),
        };
      });
    } catch (_) {
      // ignore flow-state fetch errors to avoid blocking legacy pages.
    } finally {
      if (mounted) {
        setState(() {
          _flowStateLoading = false;
        });
      }
    }
  }

  Future<void> _updateCurrentProjectFlowStep(String step, String status) async {
    final project = _currentProjectSummary();
    if (project == null || !project.isFlowProject) {
      return;
    }
    setState(() {
      _flowStateLoading = true;
    });
    try {
      final uri = ApiConfig.httpUri('/projects/${project.id}/flow');
      final response = await http.put(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'step': step, 'status': status}),
      ).timeout(const Duration(seconds: 20));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        return;
      }
      if (!mounted) return;
      setState(() {
        _currentFlowState[step] = status;
      });
    } catch (_) {
      // ignore flow-state update errors in shell-level UI.
    } finally {
      if (mounted) {
        setState(() {
          _flowStateLoading = false;
        });
      }
    }
  }

  Widget _flowStepStatusBadge({
    required String stepKey,
    required String currentStatus,
  }) {
    final selectedStatus = kFlowStatuses.contains(currentStatus)
        ? currentStatus
        : '未着手';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.primary.withOpacity(0.08),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(
          color: Theme.of(context).colorScheme.primary.withOpacity(0.35),
        ),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: selectedStatus,
          isDense: true,
          style: const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
          items: kFlowStatuses
              .map(
                (status) => DropdownMenuItem<String>(
                  value: status,
                  child: Text('状態: $status'),
                ),
              )
              .toList(),
          onChanged: _flowStateLoading
              ? null
              : (value) {
                  if (value == null || value == selectedStatus) return;
                  _updateCurrentProjectFlowStep(stepKey, value);
                },
        ),
      ),
    );
  }

  int? _nextMenuIndexForFlowStep(String stepKey) {
    switch (stepKey) {
      case 'script':
        return 2;
      case 'base_video':
        return 3;
      case 'title_description':
        return 4;
      case 'thumbnail':
        return 5;
      case 'ponchi':
        return 6;
      default:
        return null;
    }
  }

  Widget _buildFlowNextButton(String stepKey) {
    final nextIndex = _nextMenuIndexForFlowStep(stepKey);
    if (nextIndex == null) {
      return const SizedBox.shrink();
    }
    return Align(
      alignment: Alignment.bottomRight,
      child: FilledButton.icon(
        onPressed: () {
          setState(() {
            _selectedIndex = nextIndex;
          });
        },
        icon: const Icon(Icons.arrow_forward),
        label: Text('次へ（${_pageLabel(nextIndex)}）'),
      ),
    );
  }

  Widget _wrapFlowStep({
    required String stepKey,
    required String description,
    required Widget child,
  }) {
    if (!_isFlowProjectSelected) {
      return child;
    }
    final currentStatus = _currentFlowState[stepKey] ?? '未着手';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            const Spacer(),
            _flowStepStatusBadge(stepKey: stepKey, currentStatus: currentStatus),
          ],
        ),
        const SizedBox(height: 8),
        Text(description),
        const SizedBox(height: 8),
        if (_flowStateLoading) const LinearProgressIndicator(),
        const SizedBox(height: 16),
        Expanded(
          child: Scrollbar(
            thumbVisibility: true,
            trackVisibility: true,
            child: child,
          ),
        ),
        const SizedBox(height: 12),
        _buildFlowNextButton(stepKey),
      ],
    );
  }

  Future<void> _openFlowPage() async {
    setState(() {
      _selectedIndex = 11;
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
    AppLogger.warn('API通知: $message');
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
    AppLogger.error('APIエラー詳細: $line');
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

  bool _looksLikeApiErrorLine(String line) {
    final normalized = line.trim();
    if (normalized.isEmpty) {
      return false;
    }
    if (normalized.contains('Traceback')) {
      return true;
    }
    final upper = normalized.toUpperCase();
    return upper.contains(' ERROR ') ||
        upper.contains('CRITICAL') ||
        upper.contains('EXCEPTION') ||
        upper.startsWith('ERROR') ||
        upper.contains('[ERROR]');
  }

  void _setApiServerError(String message) {
    AppLogger.error('APIエラー: $message');
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

  void _resizeBottomPanel(double deltaDy, double totalHeight) {
    if (totalHeight <= 0) {
      return;
    }
    final minRatio = 120 / totalHeight;
    final maxRatio = 0.6;
    final next = (_bottomPanelRatio - (deltaDy / totalHeight)).clamp(minRatio, maxRatio);
    setState(() {
      _bottomPanelRatio = next;
      if (_bottomPanelCollapsed) {
        _bottomPanelCollapsed = false;
      }
    });
  }

  void _toggleBottomPanelCollapsed() {
    setState(() {
      _bottomPanelCollapsed = !_bottomPanelCollapsed;
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
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final totalHeight = constraints.maxHeight;
                  final minRatio = totalHeight <= 0 ? 0.0 : 120 / totalHeight;
                  final clampedRatio = _bottomPanelRatio.clamp(minRatio, 0.6);
                  final panelHeight = _bottomPanelCollapsed ? 38.0 : totalHeight * clampedRatio;
                  return Column(
                    children: [
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
                              return _buildMenuItem(index: index, label: _pageLabel(index));
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
                  Expanded(
                    flex: 1,
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
                ],
                      ),
                      ),
                      MouseRegion(
                        cursor: SystemMouseCursors.resizeUpDown,
                        child: GestureDetector(
                          behavior: HitTestBehavior.opaque,
                          onVerticalDragUpdate: (details) {
                            _resizeBottomPanel(details.delta.dy, totalHeight);
                          },
                          child: Container(
                            height: 8,
                            color: Colors.black.withOpacity(0.06),
                            alignment: Alignment.center,
                            child: Container(
                              width: 48,
                              height: 3,
                              decoration: BoxDecoration(
                                color: Colors.black26,
                                borderRadius: BorderRadius.circular(99),
                              ),
                            ),
                          ),
                        ),
                      ),
                      SizedBox(
                        height: panelHeight,
                        child: Padding(
                          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                          child: Card(
                            child: LogPanel(
                              pageName: _pageLabel(_selectedIndex),
                              latestJobId: _latestJobId,
                              jobInProgress: _videoJobInProgress,
                              collapsed: _bottomPanelCollapsed,
                              onToggleCollapse: _toggleBottomPanelCollapsed,
                            ),
                          ),
                        ),
                      ),
                    ],
                  );
                },
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
            child: Builder(
              builder: (context) {
                final stepKey = _flowStepKeyForMenuIndex(index);
                final stepStatus = stepKey == null ? null : _currentFlowState[stepKey];
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '○ $label',
                      style: TextStyle(
                        fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
                        color: isSelected ? Colors.black87 : Colors.black54,
                      ),
                    ),
                    if (_isFlowProjectSelected && stepStatus != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 2),
                        child: Text(
                          '状態: $stepStatus',
                          style: TextStyle(
                            fontSize: 10,
                            color: isSelected ? Colors.black54 : Colors.black45,
                          ),
                        ),
                      ),
                  ],
                );
              },
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
        return _wrapFlowStep(
          stepKey: 'script',
          description: 'AI生成した台本は必ず人が確認・編集してから確定してください。自動で次工程には進みません。',
          child: ScriptGenerateForm(
            checkApiHealth: _checkApiHealthAndUpdate,
          ),
        );
      case 2:
        return _wrapFlowStep(
          stepKey: 'base_video',
          description: '既存の動画設定をそのまま使ってベース動画を生成します。ここでは最終編集を行いません。',
          child: VideoGenerateForm(
            checkApiHealth: _checkApiHealthAndUpdate,
            jobInProgress: _videoJobInProgress,
            onJobSubmitted: (jobId) {
              _latestJobId.value = jobId;
            },
          ),
        );
      case 3:
        return _wrapFlowStep(
          stepKey: 'title_description',
          description: '台本に基づいて候補を作成し、ユーザーが確認・選択して採用します。',
          child: TitleGenerateForm(
            checkApiHealth: _checkApiHealthAndUpdate,
          ),
        );
      case 4:
        return _wrapFlowStep(
          stepKey: 'thumbnail',
          description: 'タイトルを踏まえてサムネイル候補を生成し、プレビュー確認後に採用してください。',
          child: MaterialsGenerateForm(
            checkApiHealth: _checkApiHealthAndUpdate,
          ),
        );
      case 5:
        return _wrapFlowStep(
          stepKey: 'ponchi',
          description: 'SRTをもとに提案を作成し、開始/終了時間や画像・サイズ・位置は必ず手動で調整します。',
          child: PonchiGenerateForm(
            checkApiHealth: _checkApiHealthAndUpdate,
          ),
        );
      case 6:
        return _wrapFlowStep(
          stepKey: 'final_edit',
          description: 'ポンチ絵設定を反映して最終動画を出力します。必要に応じて再編集してください。',
          child: const VideoEditForm(),
        );
      case 7:
        return const DetailedEditForm();
      case 8:
        return const ImageCollectForm();
      case 9:
        return const SettingsForm();
      case 10:
        return const AboutPanel();
      case 11:
        return FlowProjectPanel(
          selectedProject: _currentProjectSummary(),
          checkApiHealth: _checkApiHealthAndUpdate,
        );
      default:
        return PlaceholderPanel(title: _pageLabel(_selectedIndex));
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
            content: Text('「${project.name}」を削除します。フォルダごと消去されます。本当によろしいですか？'),
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
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('削除しました！')),
        );
      }
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
              OutlinedButton.icon(
                onPressed: _openLargeEditorDialog,
                icon: const Icon(Icons.open_in_full),
                label: const Text('大画面で編集'),
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


  bool get _isFlowProject => ProjectState.currentProjectType.value == 'flow';

  String _flowScriptPrefsKey() => 'flow.script_path.${ProjectState.currentProjectId.value}';

  Future<void> _persistFlowScript(String text) async {
    final outputPath = _outputController.text.trim().isEmpty
        ? 'flow_confirmed_script.txt'
        : _outputController.text.trim();
    final file = File(outputPath);
    await file.writeAsString(text);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_flowScriptPrefsKey(), file.path);
  }

  Future<void> _openLargeEditorDialog() async {
    final initial = _outputTextController.text;
    final controller = TextEditingController(text: initial);
    final action = await showDialog<String>(
      context: context,
      builder: (context) {
        return Dialog(
          child: SizedBox(
            width: 980,
            height: 760,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text('台本エディタ（大画面）', style: Theme.of(context).textTheme.titleLarge),
                  const SizedBox(height: 12),
                  Expanded(
                    child: TextField(
                      controller: controller,
                      maxLines: null,
                      expands: true,
                      decoration: const InputDecoration(
                        hintText: 'ここで台本を自由に編集してください。',
                        alignLabelWithHint: true,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    alignment: WrapAlignment.end,
                    children: [
                      TextButton(
                        onPressed: () => Navigator.pop(context, 'cancel'),
                        child: const Text('キャンセル'),
                      ),
                      ElevatedButton(
                        onPressed: () => Navigator.pop(context, 'apply'),
                        child: const Text('反映'),
                      ),
                      if (_isFlowProject)
                        FilledButton(
                          onPressed: () => Navigator.pop(context, 'save_flow'),
                          child: const Text('AIフロー台本として保存'),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
    final edited = controller.text;
    controller.dispose();
    if (action == null || action == 'cancel') {
      return;
    }
    setState(() {
      _outputTextController.text = edited;
    });
    if (action == 'save_flow') {
      try {
        await _persistFlowScript(edited);
        _showSnackBar('AIフロー用の確定台本として保存しました。');
      } catch (error) {
        _showSnackBar('台本保存に失敗しました: $error');
      }
      return;
    }
    _showSnackBar('編集内容を反映しました。');
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
      if (_isFlowProject) {
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString(_flowScriptPrefsKey(), file.path);
      }
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
  final _selectedTitleController = TextEditingController();

  bool get _isFlowProject => ProjectState.currentProjectType.value == 'flow';

  String _flowScriptPrefsKey() => 'flow.script_path.${ProjectState.currentProjectId.value}';

  String _selectedTitlePrefsKey() => 'title_generate.selected_title.${ProjectState.currentProjectId.value}';

  Future<String?> _resolveScriptPathForTitle() async {
    if (!_isFlowProject) {
      return _scriptPathController.text.trim();
    }
    final prefs = await SharedPreferences.getInstance();
    return (prefs.getString(_flowScriptPrefsKey()) ?? '').trim();
  }

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
    final prefs = await SharedPreferences.getInstance();
    final selectedTitle = (prefs.getString(_selectedTitlePrefsKey()) ?? '').trim();
    if (!mounted) return;
    setState(() {
      _provider = provider ?? _provider;
      _geminiModel = geminiModel ?? _geminiModel;
      _chatGptModel = chatGptModel ?? _chatGptModel;
      _claudeModel = claudeModel ?? _claudeModel;
      _selectedTitleController.text = selectedTitle;
    });
  }

  Future<void> _persistSelectedTitle(String title) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_selectedTitlePrefsKey(), title);
  }

  @override
  void dispose() {
    _scriptPathController.dispose();
    _countController.dispose();
    _instructionsController.dispose();
    _outputController.dispose();
    _selectedTitleController.dispose();
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
          ValueListenableBuilder<String>(
            valueListenable: ProjectState.currentProjectType,
            builder: (context, projectType, _) {
              if (projectType == 'flow') {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: const [
                    Text('AIフローでは①で確定した台本を自動利用します。'),
                    SizedBox(height: 12),
                  ],
                );
              }
              return Column(
                children: [
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
                ],
              );
            },
          ),
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
          const SizedBox(height: 12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: TextFormField(
                  controller: _selectedTitleController,
                  decoration: const InputDecoration(
                    labelText: '採用する動画タイトル（⑥出力名に使用）',
                    hintText: 'ここに最終的なタイトルを入力してください。',
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Padding(
                padding: const EdgeInsets.only(top: 2),
                child: FilledButton(
                  onPressed: _confirmSelectedTitle,
                  child: const Text('確定'),
                ),
              ),
            ],
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
    final scriptPath = await _resolveScriptPathForTitle();
    if (scriptPath == null || scriptPath.isEmpty) {
      _showSnackBar(_isFlowProject
          ? 'AIフローの台本が未保存です。①台本作成で「AIフロー台本として保存」を実行してください。'
          : '台本ファイルを指定してください。');
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
        'provider': _provider,
        'script_path': scriptPath,
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
        final outputText = data['text'] as String? ?? '';
        setState(() {
          _outputController.text = outputText;
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

  Future<void> _confirmSelectedTitle() async {
    final title = _selectedTitleController.text.trim();
    if (title.isEmpty) {
      _showSnackBar('採用する動画タイトルを入力してください。');
      return;
    }
    await _persistSelectedTitle(title);
    _showSnackBar('採用タイトルを確定保存しました。');
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
  late final VoidCallback _projectListener;
  bool _isSubmitting = false;

  String _generatedImagePathPrefsKey() => 'generated_image_path.${ProjectState.currentProjectId.value}';

  String _generatedImageMimePrefsKey() => 'generated_image_mime.${ProjectState.currentProjectId.value}';

  String _generatedImageBase64PrefsKey() => 'generated_image_base64.${ProjectState.currentProjectId.value}';

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('materials_generate.', scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_modelController, 'model');
    _persistence.registerController(_promptController, 'prompt');
    _persistence.registerController(_outputController, 'output_dir');
    _projectListener = () {
      _loadGeneratedImageState();
    };
    ProjectState.currentProjectId.addListener(_projectListener);
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    await _loadGeneratedImageState();
    await _loadModelConfig();
  }

  Future<void> _persistGeneratedImageState() async {
    await _persistence.setString(_generatedImagePathPrefsKey(), _generatedImagePath ?? '');
    await _persistence.setString(_generatedImageMimePrefsKey(), _generatedImageMimeType ?? '');
    final imageBase64 = (_generatedImageBytes != null && _generatedImageBytes!.isNotEmpty)
        ? base64Encode(_generatedImageBytes!)
        : '';
    await _persistence.setString(_generatedImageBase64PrefsKey(), imageBase64);
  }

  Future<void> _loadGeneratedImageState() async {
    final path = (await _persistence.readString(_generatedImagePathPrefsKey()) ?? '').trim();
    final mime = (await _persistence.readString(_generatedImageMimePrefsKey()) ?? '').trim();
    final imageBase64 = (await _persistence.readString(_generatedImageBase64PrefsKey()) ?? '').trim();
    Uint8List? bytes;
    if (imageBase64.isNotEmpty) {
      try {
        bytes = base64Decode(imageBase64);
      } catch (_) {
        bytes = null;
      }
    }
    if (!mounted) return;
    setState(() {
      _generatedImagePath = path.isEmpty ? null : path;
      _generatedImageMimeType = mime.isEmpty ? null : mime;
      _generatedImageBytes = bytes;
    });
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
    ProjectState.currentProjectId.removeListener(_projectListener);
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
        _persistGeneratedImageState();
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

String _newPonchiRowId() => DateTime.now().microsecondsSinceEpoch.toString();

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
  static const double _compactFontSize = 12;
  final _formKey = GlobalKey<FormState>();
  final _srtController = TextEditingController();
  final _outputController = TextEditingController(text: 'ponchi_images');
  final _geminiModelController = TextEditingController(text: 'gemini-2.0-flash');
  final _outputTextController = TextEditingController();
  final List<_PonchiPreviewItem> _previewItems = [];
  final List<_PonchiIdeaRow> _ideaRows = [];
  final Set<String> _rowGeneratingKeys = <String>{};
  late final InputPersistence _persistence;
  String _engine = 'Gemini';
  List<String> _ponchiModels = const ['gemini-2.0-flash'];
  bool _isSubmittingIdeas = false;
  bool _isSubmittingImages = false;
  late final VoidCallback _projectListener;

  bool get _isFlowProject => ProjectState.currentProjectType.value == 'flow';

  String _lastSrtPathPrefsKey() => 'video_generate.last_srt_path.${ProjectState.currentProjectId.value}';

  String _lastVideoPathPrefsKey() => 'video_generate.last_video_path.${ProjectState.currentProjectId.value}';

  String _ideaRowsPrefsKey() => 'ponchi.idea_rows.${ProjectState.currentProjectId.value}';

  String _ponchiOutputPrefsKey() => 'ponchi.output_text.${ProjectState.currentProjectId.value}';

  String _sharedOverlayRowsPrefsKey() => 'ponchi.overlay_rows.${ProjectState.currentProjectId.value}';

  Future<void> _loadFlowDefaultSrtPath() async {
    if (!_isFlowProject) {
      return;
    }
    final prefs = await SharedPreferences.getInstance();
    var srtPath = (prefs.getString(_lastSrtPathPrefsKey()) ?? '').trim();
    if (srtPath.isEmpty) {
      final videoPath = (prefs.getString(_lastVideoPathPrefsKey()) ?? '').trim();
      if (videoPath.isNotEmpty) {
        final videoFile = File(videoPath);
        final stem = videoFile.uri.pathSegments.isEmpty
            ? ''
            : videoFile.uri.pathSegments.last.split('.').first;
        if (stem.isNotEmpty) {
          srtPath = '${videoFile.parent.path}${Platform.pathSeparator}$stem.srt';
        }
      }
    }
    if (srtPath.isEmpty) {
      return;
    }
    if (!await File(srtPath).exists()) {
      return;
    }
    if (!mounted) return;
    setState(() {
      _srtController.text = srtPath;
    });
  }

  void _syncMarkdownTableFromRows() {
    final buffer = StringBuffer();
    buffer.writeln('| 開始 | 終了 | ポンチ絵内容 | 画像生成プロンプト |');
    buffer.writeln('|---|---|---|---|');
    for (final row in _ideaRows) {
      final visual = row.visualSuggestion.replaceAll('|', r'\|');
      final prompt = row.imagePrompt.replaceAll('|', r'\|');
      buffer.writeln('| ${row.start} | ${row.end} | $visual | $prompt |');
    }
    _outputTextController.text = buffer.toString();
    _persistPonchiState();
  }

  void _syncIdeaRowsImagePathFromPreviews() {
    for (final row in _ideaRows) {
      final preview = _findPreviewForRow(row);
      if (preview != null && preview.path.trim().isNotEmpty) {
        row.imagePath = preview.path.trim();
      }
    }
  }

  Future<void> _persistSharedOverlayRows() async {
    final prefs = await SharedPreferences.getInstance();
    final payload = _ideaRows
        .map((row) => {
              'id': row.id,
              'start': row.start,
              'end': row.end,
              'visual_suggestion': row.visualSuggestion,
              'image_prompt': row.imagePrompt,
              'image_path': row.imagePath,
              'x': row.x,
              'y': row.y,
              'w': row.w,
              'h': row.h,
              'opacity': row.opacity,
            })
        .toList();
    await prefs.setString(_sharedOverlayRowsPrefsKey(), jsonEncode(payload));
  }

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('ponchi_generate.', scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_srtController, 'srt_path');
    _persistence.registerController(_outputController, 'output_dir');
    _persistence.registerController(_geminiModelController, 'gemini_model');
    _projectListener = () {
      _loadFlowDefaultSrtPath();
      _loadPersistedPonchiState();
    };
    ProjectState.currentProjectId.addListener(_projectListener);
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    await _loadPonchiModelConfig();
    await _loadFlowDefaultSrtPath();
    await _loadPersistedPonchiState();
  }


  Future<void> _persistPonchiState() async {
    final rowsPayload = _ideaRows
        .map(
          (row) => {
            'id': row.id,
            'start': row.start,
            'end': row.end,
            'visual_suggestion': row.visualSuggestion,
            'image_prompt': row.imagePrompt,
            'image_path': row.imagePath,
            'x': row.x,
            'y': row.y,
            'w': row.w,
            'h': row.h,
            'opacity': row.opacity,
          },
        )
        .toList();
    await _persistence.setString(_ideaRowsPrefsKey(), jsonEncode(rowsPayload));
    await _persistence.setString(_ponchiOutputPrefsKey(), _outputTextController.text);
    await _persistSharedOverlayRows();
  }

  Future<void> _loadPersistedPonchiState() async {
    final rawRows = (await _persistence.readString(_ideaRowsPrefsKey()) ?? '').trim();
    final rawOutput = await _persistence.readString(_ponchiOutputPrefsKey());
    final rows = <_PonchiIdeaRow>[];
    if (rawRows.isNotEmpty) {
      try {
        final decoded = jsonDecode(rawRows);
        if (decoded is List) {
          for (final item in decoded) {
            if (item is! Map<String, dynamic>) {
              continue;
            }
            rows.add(
              _PonchiIdeaRow(
                id: (item['id'] as String? ?? '').trim().isEmpty
                    ? _newPonchiRowId()
                    : (item['id'] as String).trim(),
                start: (item['start'] as String? ?? '').trim(),
                end: (item['end'] as String? ?? '').trim(),
                visualSuggestion: (item['visual_suggestion'] as String? ?? '').trim(),
                imagePrompt: (item['image_prompt'] as String? ?? '').trim(),
                imagePath: (item['image_path'] as String? ?? '').trim(),
                x: (item['x'] as String? ?? '100').trim(),
                y: (item['y'] as String? ?? '200').trim(),
                w: (item['w'] as String? ?? '0').trim(),
                h: (item['h'] as String? ?? '0').trim(),
                opacity: (item['opacity'] as String? ?? '1.0').trim(),
              ),
            );
          }
        }
      } catch (_) {
        // ignore persisted parse errors.
      }
    }
    if (!mounted) return;
    setState(() {
      _ideaRows
        ..clear()
        ..addAll(rows);
      _outputTextController.text = rawOutput ?? '';
    });
    if (_ideaRows.isNotEmpty && _outputTextController.text.trim().isEmpty) {
      _syncMarkdownTableFromRows();
    }
    await _loadPersistedPonchiPreviews();
    _syncIdeaRowsImagePathFromPreviews();
    await _persistSharedOverlayRows();
  }

  String _ponchiPreviewItemsPrefsKey() => 'ponchi.preview_items.${ProjectState.currentProjectId.value}';

  Future<void> _persistPonchiPreviewItems() async {
    final payload = _previewItems
        .map(
          (item) => {
            'title': item.title,
            'subtitle': item.subtitle,
            'path': item.path,
            'bytes': (item.bytes != null && item.bytes!.isNotEmpty) ? base64Encode(item.bytes!) : '',
          },
        )
        .toList();
    await _persistence.setString(_ponchiPreviewItemsPrefsKey(), jsonEncode(payload));
  }

  Future<void> _loadPersistedPonchiPreviews() async {
    final raw = (await _persistence.readString(_ponchiPreviewItemsPrefsKey()) ?? '').trim();
    if (raw.isEmpty) {
      if (!mounted) return;
      setState(() {
        _previewItems.clear();
      });
      return;
    }
    final previews = <_PonchiPreviewItem>[];
    try {
      final decoded = jsonDecode(raw);
      if (decoded is List) {
        for (final item in decoded) {
          if (item is! Map<String, dynamic>) continue;
          final encoded = (item['bytes'] as String? ?? '').trim();
          Uint8List? bytes;
          if (encoded.isNotEmpty) {
            try {
              bytes = base64Decode(encoded);
            } catch (_) {
              bytes = null;
            }
          }
          previews.add(
            _PonchiPreviewItem(
              title: (item['title'] as String? ?? '').trim(),
              subtitle: (item['subtitle'] as String? ?? '').trim(),
              path: (item['path'] as String? ?? '').trim(),
              bytes: bytes,
            ),
          );
        }
      }
    } catch (_) {
      // ignore parse errors.
    }
    if (!mounted) return;
    setState(() {
      _previewItems
        ..clear()
        ..addAll(previews);
    });
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
    ProjectState.currentProjectId.removeListener(_projectListener);
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
          ValueListenableBuilder<String>(
            valueListenable: ProjectState.currentProjectType,
            builder: (context, projectType, _) {
              final isFlow = projectType == 'flow';
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (isFlow)
                    const Padding(
                      padding: EdgeInsets.only(bottom: 8),
                      child: Text('AIフローでは②動画作成で出力されたSRTをデフォルト読み込みします。'),
                    ),
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
                ],
              );
            },
          ),
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
                label: Text(_isSubmittingImages ? '生成中...' : '画像生成（ポンチ絵作成）'),
              ),
              OutlinedButton.icon(
                onPressed: () {
                  _outputTextController.clear();
                  setState(() {
                    _previewItems.clear();
                    _ideaRows.clear();
                  });
                  _persistPonchiState();
                  _persistPonchiPreviewItems();
                },
                icon: const Icon(Icons.clear),
                label: const Text('クリア'),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text('生成結果', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 12),
          TextFormField(
            controller: _outputTextController,
            maxLines: 8,
            decoration: const InputDecoration(
              hintText: '生成結果（Markdown表）がここに表示されます。',
            ),
          ),
          const SizedBox(height: 12),
          Text('提案テーブル（編集可）', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          if (_ideaRows.isEmpty)
            const Text('案出しを実行すると編集可能な表が表示されます。')
          else
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Table(
                border: TableBorder.all(color: Theme.of(context).dividerColor),
                defaultVerticalAlignment: TableCellVerticalAlignment.middle,
                columnWidths: const {
                  0: FixedColumnWidth(90),
                  1: FixedColumnWidth(90),
                  2: FixedColumnWidth(220),
                  3: FixedColumnWidth(300),
                  4: FixedColumnWidth(72),
                  5: FixedColumnWidth(72),
                  6: FixedColumnWidth(72),
                  7: FixedColumnWidth(72),
                  8: FixedColumnWidth(88),
                  9: FixedColumnWidth(120),
                  10: FixedColumnWidth(180),
                  11: FixedColumnWidth(240),
                  12: FixedColumnWidth(56),
                },
                children: [
                  _buildIdeaTableHeaderRow(context),
                  ..._ideaRows.map(_buildIdeaTableDataRow),
                ],
              ),
            ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }


  TableRow _buildIdeaTableHeaderRow(BuildContext context) {
    final headerStyle = Theme.of(context).textTheme.labelSmall?.copyWith(
          fontSize: _compactFontSize,
          fontWeight: FontWeight.w700,
        );

    Widget headerCell(String text) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        child: Text(text, style: headerStyle),
      );
    }

    return TableRow(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceVariant,
      ),
      children: [
        headerCell('開始'),
        headerCell('終了'),
        headerCell('ポンチ絵内容'),
        headerCell('画像生成プロンプト'),
        headerCell('X'),
        headerCell('Y'),
        headerCell('W'),
        headerCell('H'),
        headerCell('透明度'),
        headerCell('画像生成'),
        headerCell('プレビュー'),
        headerCell('画像情報'),
        headerCell('削除'),
      ],
    );
  }

  TableRow _buildIdeaTableDataRow(_PonchiIdeaRow row) {
    Widget cell(Widget child) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
        child: child,
      );
    }

    return TableRow(
      children: [
        cell(
          TextFormField(
            initialValue: row.start,
            style: const TextStyle(fontSize: _compactFontSize),
            decoration: _compactInputDecoration(hint: '開始'),
            onChanged: (v) {
              row.start = v;
              _syncMarkdownTableFromRows();
            },
          ),
        ),
        cell(
          TextFormField(
            initialValue: row.end,
            style: const TextStyle(fontSize: _compactFontSize),
            decoration: _compactInputDecoration(hint: '終了'),
            onChanged: (v) {
              row.end = v;
              _syncMarkdownTableFromRows();
            },
          ),
        ),
        cell(
          TextFormField(
            initialValue: row.visualSuggestion,
            maxLines: 2,
            style: const TextStyle(fontSize: _compactFontSize),
            decoration: _compactInputDecoration(hint: '内容'),
            onChanged: (v) {
              row.visualSuggestion = v;
              _syncMarkdownTableFromRows();
            },
          ),
        ),
        cell(
          TextFormField(
            initialValue: row.imagePrompt,
            maxLines: 2,
            style: const TextStyle(fontSize: _compactFontSize),
            decoration: _compactInputDecoration(hint: 'プロンプト'),
            onChanged: (v) {
              row.imagePrompt = v;
              _syncMarkdownTableFromRows();
            },
          ),
        ),
        cell(
          TextFormField(
            initialValue: row.x,
            style: const TextStyle(fontSize: _compactFontSize),
            decoration: _compactInputDecoration(hint: 'X'),
            onChanged: (v) {
              row.x = v;
              _persistPonchiState();
            },
          ),
        ),
        cell(
          TextFormField(
            initialValue: row.y,
            style: const TextStyle(fontSize: _compactFontSize),
            decoration: _compactInputDecoration(hint: 'Y'),
            onChanged: (v) {
              row.y = v;
              _persistPonchiState();
            },
          ),
        ),
        cell(
          TextFormField(
            initialValue: row.w,
            style: const TextStyle(fontSize: _compactFontSize),
            decoration: _compactInputDecoration(hint: 'W'),
            onChanged: (v) {
              row.w = v;
              _persistPonchiState();
            },
          ),
        ),
        cell(
          TextFormField(
            initialValue: row.h,
            style: const TextStyle(fontSize: _compactFontSize),
            decoration: _compactInputDecoration(hint: 'H'),
            onChanged: (v) {
              row.h = v;
              _persistPonchiState();
            },
          ),
        ),
        cell(
          TextFormField(
            initialValue: row.opacity,
            style: const TextStyle(fontSize: _compactFontSize),
            decoration: _compactInputDecoration(hint: '透明度'),
            onChanged: (v) {
              row.opacity = v;
              _persistPonchiState();
            },
          ),
        ),
        cell(
          SizedBox(
            height: 36,
            child: ElevatedButton(
              onPressed: _isSubmittingImages || _rowGeneratingKeys.contains(_rowKey(row))
                  ? null
                  : () => _generateImageForRow(row),
              child: Text(
                (_isSubmittingImages || _rowGeneratingKeys.contains(_rowKey(row)))
                    ? '生成中'
                    : '生成',
                style: const TextStyle(fontSize: _compactFontSize),
              ),
            ),
          ),
        ),
        cell(_buildPreviewCell(row)),
        cell(
          SelectableText(
            row.imagePath.isEmpty ? '未設定' : row.imagePath,
            style: const TextStyle(fontSize: _compactFontSize),
          ),
        ),
        cell(
          IconButton(
            onPressed: () {
              setState(() {
                _ideaRows.remove(row);
              });
              _syncMarkdownTableFromRows();
              _persistPonchiPreviewItems();
            },
            icon: const Icon(Icons.delete_outline),
            tooltip: '行削除',
          ),
        ),
      ],
    );
  }

  _PonchiPreviewItem? _findPreviewForRow(_PonchiIdeaRow row) {
    final rowTitle = '${row.start}〜${row.end}'.trim();
    for (final item in _previewItems) {
      if (item.title.trim() == rowTitle) {
        return item;
      }
    }
    return null;
  }

  String _rowKey(_PonchiIdeaRow row) => '${row.start}|${row.end}|${row.imagePrompt}';

  Widget _buildPreviewCell(_PonchiIdeaRow row) {
    final preview = _findPreviewForRow(row);
    if (preview == null) {
      return const SizedBox(
        height: 72,
        child: Center(
          child: Text('未生成', style: TextStyle(fontSize: _compactFontSize)),
        ),
      );
    }
    final image = preview.bytes != null
        ? Image.memory(preview.bytes!, fit: BoxFit.cover)
        : Image.file(
            File(preview.path),
            fit: BoxFit.cover,
            errorBuilder: (_, __, ___) => const Center(child: Text('読込失敗')),
          );
    return GestureDetector(
      onTap: () => _openPreviewDialog(preview),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: SizedBox(height: 72, child: image),
      ),
    );
  }

  Future<void> _generateImageForRow(_PonchiIdeaRow row) async {
    final apiKey = ApiKeys.gemini.value;
    if (apiKey.isEmpty) {
      _showSnackBar('Gemini APIキーが未設定です。設定タブで入力してください。');
      return;
    }
    if (_outputController.text.trim().isEmpty) {
      _showSnackBar('出力フォルダを指定してください。');
      return;
    }
    final rowKey = _rowKey(row);
    setState(() {
      _rowGeneratingKeys.add(rowKey);
    });
    try {
      final healthy = await widget.checkApiHealth();
      if (!healthy) {
        _showSnackBar('API サーバーに接続できません。');
        return;
      }
      final payload = {
        'api_key': apiKey,
        'prompt': row.imagePrompt.trim().isEmpty
            ? row.visualSuggestion
            : row.imagePrompt.trim(),
        'output_dir': _outputController.text.trim(),
        'project_id': ProjectState.currentProjectId.value,
      };
      final response = await http
          .post(
            ApiConfig.httpUri('/materials/generate'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 180));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        _showSnackBar('行画像の生成に失敗しました: ${response.statusCode} ${response.body}');
        return;
      }
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final imagePath = (data['image_path'] as String? ?? '').trim();
      final imageBase64 = (data['image_base64'] as String? ?? '').trim();
      Uint8List? bytes;
      if (imageBase64.isNotEmpty) {
        bytes = base64Decode(imageBase64);
      }
      final generated = _PonchiPreviewItem(
        title: '${row.start}〜${row.end}',
        subtitle: row.visualSuggestion,
        path: imagePath,
        bytes: bytes,
      );
      if (imagePath.isNotEmpty) {
        row.imagePath = imagePath;
      }
      if (!mounted) return;
      if (generated.path.isEmpty && (generated.bytes == null || generated.bytes!.isEmpty)) {
        _showSnackBar('この行の画像が生成されませんでした。');
        return;
      }
      setState(() {
        _previewItems.removeWhere((item) => item.title == generated.title);
        _previewItems.add(generated);
      });
      _persistPonchiPreviewItems();
      await _persistSharedOverlayRows();
      await _openPreviewDialog(generated);
      _showSnackBar('行ごとの画像生成が完了しました。');
    } on TimeoutException {
      _showSnackBar('行画像生成リクエストがタイムアウトしました。');
    } catch (error) {
      _showSnackBar('行画像生成エラー: $error');
    } finally {
      if (mounted) {
        setState(() {
          _rowGeneratingKeys.remove(rowKey);
        });
      }
    }
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

  InputDecoration _compactInputDecoration({String? label, String? hint}) {
    return InputDecoration(
      labelText: label,
      hintText: hint,
      isDense: true,
      contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      labelStyle: const TextStyle(fontSize: _compactFontSize),
    );
  }

  Future<void> _openPreviewDialog(_PonchiPreviewItem item) async {
    await showDialog<void>(
      context: context,
      builder: (context) {
        final image = item.bytes != null
            ? Image.memory(item.bytes!, fit: BoxFit.contain)
            : Image.file(
                File(item.path),
                fit: BoxFit.contain,
                errorBuilder: (_, __, ___) => const Text('画像を読み込めませんでした。'),
              );
        return Dialog(
          insetPadding: const EdgeInsets.all(24),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(item.title, style: Theme.of(context).textTheme.titleMedium),
                if (item.subtitle.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  Text(item.subtitle, style: Theme.of(context).textTheme.bodySmall),
                ],
                const SizedBox(height: 12),
                Flexible(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxHeight: 560),
                    child: image,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
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
        final rows = <_PonchiIdeaRow>[];
        for (final item in items) {
          final map = item as Map<String, dynamic>;
          rows.add(
            _PonchiIdeaRow(
              id: _newPonchiRowId(),
              start: (map['start'] as String? ?? '').trim(),
              end: (map['end'] as String? ?? '').trim(),
              visualSuggestion: (map['visual_suggestion'] as String? ?? '').trim(),
              imagePrompt: (map['image_prompt'] as String? ?? '').trim(),
            ),
          );
        }
        setState(() {
          _ideaRows
            ..clear()
            ..addAll(rows);
        });
        _syncMarkdownTableFromRows();
        if (jsonPath != null && jsonPath.trim().isNotEmpty) {
          _outputTextController.text =
              '${_outputTextController.text}\n\n<!-- source_json: ${jsonPath.trim()} -->';
          _persistPonchiState();
        }
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
                  GestureDetector(
                    onTap: () => _openPreviewDialog(item),
                    child: SizedBox(
                      height: 180,
                      child: item.bytes != null
                          ? Image.memory(item.bytes!, fit: BoxFit.contain)
                          : Image.file(
                              File(item.path),
                              fit: BoxFit.contain,
                              errorBuilder: (_, __, ___) => const Text('画像を読み込めませんでした。'),
                            ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton.icon(
                      onPressed: () => _openPreviewDialog(item),
                      icon: const Icon(Icons.zoom_in),
                      label: const Text('拡大プレビュー'),
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
          _syncIdeaRowsImagePathFromPreviews();
        });
        _persistPonchiPreviewItems();
        await _persistSharedOverlayRows();
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


class _PonchiIdeaRow {
  _PonchiIdeaRow({
    required this.id,
    required this.start,
    required this.end,
    required this.visualSuggestion,
    required this.imagePrompt,
    this.imagePath = '',
    this.x = '100',
    this.y = '200',
    this.w = '0',
    this.h = '0',
    this.opacity = '1.0',
  });

  String id;
  String start;
  String end;
  String visualSuggestion;
  String imagePrompt;
  String imagePath;
  String x;
  String y;
  String w;
  String h;
  String opacity;
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
  final _outputDirController = TextEditingController();
  late final InputPersistence _persistence;
  double _previewX = 0;
  double _previewY = 0;
  double _previewOverlayW = 0;
  double _previewOverlayH = 0;
  double _previewOpacity = 1.0;
  String _previewOverlayPath = '';
  String? _selectedTitle;
  final List<Map<String, String>> _linkedPonchiRows = [];
  late final VoidCallback _projectListener;
  VideoPlayerController? _videoPreviewController;
  String _videoPreviewPath = '';
  bool _videoPreviewInitializing = false;
  String? _videoPreviewError;
  bool _isExporting = false;
  double? _exportProgress;
  String _exportStatusMessage = '待機中';
  String? _exportJobId;

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('video_edit.', scopeListenable: ProjectState.currentProjectId);
    _persistence.registerController(_inputVideoController, 'input_video');
    _persistence.registerController(_outputDirController, 'output_dir');
    _projectListener = () {
      _loadLinkedPonchiRows();
    };
    ProjectState.currentProjectId.addListener(_projectListener);
    _initPersistence();
  }

  Future<void> _initPersistence() async {
    await _persistence.init();
    final previewX = await _persistence.readDouble('preview_x');
    final previewY = await _persistence.readDouble('preview_y');
    final previewW = await _persistence.readDouble('preview_w');
    final previewH = await _persistence.readDouble('preview_h');
    final previewOpacity = await _persistence.readDouble('preview_opacity');
    final previewOverlayPath = await _persistence.readString('preview_overlay_path');
    final prefs = await SharedPreferences.getInstance();
    final selectedTitle = (prefs.getString(_selectedTitlePrefsKey()) ?? '').trim();
    if (!mounted) return;
    setState(() {
      _previewX = previewX ?? _previewX;
      _previewY = previewY ?? _previewY;
      _previewOverlayW = previewW ?? _previewOverlayW;
      _previewOverlayH = previewH ?? _previewOverlayH;
      _previewOpacity = previewOpacity ?? _previewOpacity;
      _previewOverlayPath = previewOverlayPath ?? _previewOverlayPath;
      _selectedTitle = selectedTitle.isEmpty ? null : selectedTitle;
    });
    await _loadLinkedPonchiRows();
    await _initInputVideoPreview(_inputVideoController.text);
  }

  String _selectedTitlePrefsKey() => 'title_generate.selected_title.${ProjectState.currentProjectId.value}';

  String _sanitizeFileName(String raw) {
    final replaced = raw.replaceAll(RegExp(r'[\\/:*?"<>|]'), '_').trim();
    return replaced.isEmpty ? 'output' : replaced;
  }

  String _resolvedOutputFileName() {
    final title = (_selectedTitle ?? '').trim();
    return '${_sanitizeFileName(title.isEmpty ? 'output' : title)}.mp4';
  }

  String _resolvedOutputPath() {
    final dir = _outputDirController.text.trim().isEmpty
        ? Directory.current.path
        : _outputDirController.text.trim();
    return '$dir${Platform.pathSeparator}${_resolvedOutputFileName()}';
  }

  Future<void> _persistLinkedPonchiRows() async {
    final prefs = await SharedPreferences.getInstance();
    final payload = _linkedPonchiRows
        .map(
          (row) => {
            'id': (row['id'] ?? '').trim(),
            'start': (row['start'] ?? '').trim(),
            'end': (row['end'] ?? '').trim(),
            'visual_suggestion': (row['visual'] ?? '').trim(),
            'image_path': (row['image'] ?? '').trim(),
            'x': (row['x'] ?? '').trim(),
            'y': (row['y'] ?? '').trim(),
            'w': (row['w'] ?? '').trim(),
            'h': (row['h'] ?? '').trim(),
            'opacity': (row['opacity'] ?? '').trim(),
          },
        )
        .toList();
    await prefs.setString(_sharedOverlayRowsPrefsKey(), jsonEncode(payload));
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  String _sharedOverlayRowsPrefsKey() => 'ponchi.overlay_rows.${ProjectState.currentProjectId.value}';

  Future<void> _loadLinkedPonchiRows() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = (prefs.getString(_sharedOverlayRowsPrefsKey()) ?? '').trim();
    final rows = <Map<String, String>>[];
    if (raw.isNotEmpty) {
      try {
        final decoded = jsonDecode(raw);
        if (decoded is List) {
          for (final item in decoded) {
            if (item is! Map<String, dynamic>) {
              continue;
            }
            rows.add({
              'id': (item['id'] as String? ?? '').trim(),
              'start': (item['start'] as String? ?? '').trim(),
              'end': (item['end'] as String? ?? '').trim(),
              'visual': (item['visual_suggestion'] as String? ?? '').trim(),
              'image': (item['image_path'] as String? ?? '').trim(),
              'x': (item['x'] as String? ?? '100').trim(),
              'y': (item['y'] as String? ?? '200').trim(),
              'w': (item['w'] as String? ?? '0').trim(),
              'h': (item['h'] as String? ?? '0').trim(),
              'opacity': (item['opacity'] as String? ?? '1.0').trim(),
            });
          }
        }
      } catch (_) {
        // ignore parse errors
      }
    }
    if (!mounted) return;
    setState(() {
      _linkedPonchiRows
        ..clear()
        ..addAll(rows);
    });
  }

  Future<void> _startExport() async {
    if (_isExporting) {
      return;
    }
    final selectedTitle = (_selectedTitle ?? '').trim();
    if (selectedTitle.isEmpty) {
      _showSnackBar('③タイトル・説明文作成で動画タイトルを入力してください。');
      return;
    }

    final inputPath = _inputVideoController.text.trim();
    if (inputPath.isEmpty || !File(inputPath).existsSync()) {
      _showSnackBar('入力動画が見つかりません。');
      return;
    }

    final outputPath = _resolvedOutputPath();
    final outputDir = Directory(File(outputPath).parent.path);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }

    final overlays = _buildExportOverlays();

    setState(() {
      _isExporting = true;
      _exportProgress = 0.05;
      _exportStatusMessage = '書き出し準備中...';
    });
    AppLogger.info('最終編集: 書き出し準備開始 overlays=${overlays.length}');

    try {
      setState(() {
        _exportProgress = 0.2;
        _exportStatusMessage = 'FFmpeg コマンドを構築中...';
      });
      setState(() {
        _exportProgress = 0.35;
        _exportStatusMessage = '最終書き出しジョブを開始中...';
      });
      final response = await http
          .post(
            ApiConfig.httpUri('/video/final-export-job'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'input_path': inputPath,
              'output_path': outputPath,
              'overlays': overlays,
            }),
          )
          .then((resp) async {
            if (resp.statusCode != 404) return resp;
            return _postWithApiPrefixFallback(
              '/video/final-export-job',
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode({
                'input_path': inputPath,
                'output_path': outputPath,
                'overlays': overlays,
              }),
            );
          })
          .timeout(const Duration(minutes: 10));

      if (response.statusCode < 200 || response.statusCode >= 300) {
        final detail = (() {
          try {
            final data = jsonDecode(response.body) as Map<String, dynamic>;
            return (data['detail'] ?? response.body).toString();
          } catch (_) {
            return response.body;
          }
        })();
        AppLogger.error('最終編集: 書き出し失敗', error: 'status=${response.statusCode} detail=$detail');
        if (!mounted) return;
        setState(() {
          _exportProgress = 0.0;
          _exportStatusMessage = '失敗: $detail';
        });
        _showSnackBar('書き出しに失敗しました: $detail');
        return;
      }

      String? jobId;
      try {
        final body = jsonDecode(response.body) as Map<String, dynamic>;
        jobId = (body['job_id'] as String?)?.trim();
      } catch (_) {}
      if (jobId == null || jobId.isEmpty) {
        if (!mounted) return;
        setState(() {
          _exportProgress = 0.0;
          _exportStatusMessage = '失敗: job_id が取得できませんでした';
        });
        _showSnackBar('書き出しに失敗しました: job_id が取得できませんでした');
        return;
      }

      if (!mounted) return;
      setState(() {
        _exportJobId = jobId;
        _exportStatusMessage = '最終書き出し中... (job: $jobId)';
      });

      Map<String, dynamic>? lastStatus;
      var consecutiveFetchFailures = 0;
      for (var i = 0; i < 1200; i += 1) {
        await Future.delayed(const Duration(milliseconds: 500));
        http.Response statusResponse;
        try {
          statusResponse = await _getWithApiPrefixFallback('/jobs/$jobId')
              .timeout(const Duration(seconds: 5));
        } catch (e) {
          consecutiveFetchFailures += 1;
          if (consecutiveFetchFailures >= 20) {
            AppLogger.error('最終編集: ジョブ状態取得連続失敗', error: e);
            break;
          }
          if (mounted) {
            setState(() {
              _exportStatusMessage = '最終書き出し中... status=running (通信再試行 ${consecutiveFetchFailures}/20)';
            });
          }
          continue;
        }
        if (statusResponse.statusCode < 200 || statusResponse.statusCode >= 300) {
          consecutiveFetchFailures += 1;
          if (consecutiveFetchFailures >= 20) {
            break;
          }
          continue;
        }

        consecutiveFetchFailures = 0;
        final statusData = jsonDecode(statusResponse.body) as Map<String, dynamic>;
        lastStatus = statusData;
        final progress = (statusData['progress'] as num?)?.toDouble();
        final status = (statusData['status'] as String? ?? '').trim();
        final progressText = progress == null ? '--' : '${(progress * 100).toStringAsFixed(1)}%';
        if (mounted) {
          setState(() {
            _exportProgress = progress == null ? _exportProgress : progress.clamp(0.0, 1.0).toDouble();
            _exportStatusMessage =
                '最終書き出し中... status=${status.isEmpty ? 'running' : status} / progress=$progressText';
          });
        }
        if (status == 'completed' || status == 'error') {
          break;
        }
      }

      final status = (lastStatus?['status'] as String? ?? '').trim();
      if (status != 'completed') {
        final error = (lastStatus?['error'] ?? 'タイムアウトまたは状態取得失敗（ジョブ状態の取得が継続できませんでした）').toString();
        if (!mounted) return;
        setState(() {
          _exportProgress = 0.0;
          _exportStatusMessage = '失敗: $error';
        });
        _showSnackBar('書き出しに失敗しました: $error');
        return;
      }

      String finalOutputPath = outputPath;
      try {
        final result = (lastStatus?['result'] as Map<String, dynamic>?);
        finalOutputPath = (result?['output_path'] as String? ?? outputPath).trim();
      } catch (_) {}

      setState(() {
        _exportProgress = 1.0;
        _exportStatusMessage = '書き出し完了';
      });
      _showSnackBar('書き出し完了: $finalOutputPath');
      await _initInputVideoPreview(finalOutputPath);
      _inputVideoController.text = finalOutputPath;
      await _persistence.setString('input_video', finalOutputPath);
    } catch (e, st) {
      AppLogger.error('最終編集: 書き出し失敗', error: e, stackTrace: st);
      if (!mounted) return;
      setState(() {
        _exportProgress = 0.0;
        _exportStatusMessage = '失敗: $e';
      });
      _showSnackBar('書き出しに失敗しました: $e');
    } finally {
      if (!mounted) return;
      setState(() {
        _isExporting = false;
        _exportJobId = null;
      });
    }
  }

  double _parseDouble(String? value, double fallback) {
    return double.tryParse((value ?? '').trim()) ?? fallback;
  }

  double _parseTimeSeconds(String? value) {
    final raw = (value ?? '').trim();
    if (raw.isEmpty) {
      return 0.0;
    }
    final normalized = raw.replaceAll(',', '.');
    final parts = normalized.split(':');
    if (parts.length == 1) {
      return double.tryParse(parts[0]) ?? 0.0;
    }
    if (parts.length == 2) {
      final m = int.tryParse(parts[0]) ?? 0;
      final s = double.tryParse(parts[1]) ?? 0.0;
      return m * 60 + s;
    }
    final h = int.tryParse(parts[parts.length - 3]) ?? 0;
    final m = int.tryParse(parts[parts.length - 2]) ?? 0;
    final s = double.tryParse(parts[parts.length - 1]) ?? 0.0;
    return h * 3600 + m * 60 + s;
  }

  List<Map<String, dynamic>> _buildExportOverlays() {
    final overlays = <Map<String, dynamic>>[];
    for (final row in _linkedPonchiRows) {
      final imagePath = (row['image'] ?? '').trim();
      if (imagePath.isEmpty || !File(imagePath).existsSync()) {
        continue;
      }
      final start = _parseTimeSeconds(row['start']);
      final end = _parseTimeSeconds(row['end']);
      if (end <= start) {
        continue;
      }
      overlays.add({
        'image': imagePath,
        'start': start,
        'end': end,
        'x': _parseDouble(row['x'], 0).round(),
        'y': _parseDouble(row['y'], 0).round(),
        'w': _parseDouble(row['w'], 0).round(),
        'h': _parseDouble(row['h'], 0).round(),
        'opacity': _parseDouble(row['opacity'], 1.0).clamp(0.0, 1.0),
      });
    }
    return overlays;
  }

  List<String> _buildFfmpegArgs({
    required String inputPath,
    required String outputPath,
    required List<Map<String, dynamic>> overlays,
  }) {
    final args = <String>['-y', '-i', inputPath];
    if (overlays.isEmpty) {
      args.addAll(['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-c:a', 'copy', outputPath]);
      return args;
    }

    for (final overlay in overlays) {
      args.addAll(['-loop', '1', '-i', overlay['image'] as String]);
    }

    final filters = <String>['[0:v]setpts=PTS-STARTPTS[v0]'];
    var prevLabel = 'v0';
    for (var i = 0; i < overlays.length; i += 1) {
      final overlay = overlays[i];
      final inputIndex = i + 1;
      final imgLabel = 'ov$i';
      final outLabel = 'v${i + 1}';
      final opacity = (overlay['opacity'] as double).toStringAsFixed(3);
      final w = overlay['w'] as int;
      final h = overlay['h'] as int;
      final start = (overlay['start'] as double).toStringAsFixed(3);
      final end = (overlay['end'] as double).toStringAsFixed(3);
      final x = overlay['x'] as int;
      final y = overlay['y'] as int;

      var imageFilter = '[$inputIndex:v]format=rgba,colorchannelmixer=aa=$opacity';
      if (w > 0 && h > 0) {
        imageFilter += ',scale=$w:$h';
      }
      imageFilter += '[$imgLabel]';
      filters.add(imageFilter);

      filters.add(
        '[$prevLabel][$imgLabel]overlay=x=$x:y=$y:enable=between(t\\,$start\\,$end)[$outLabel]',
      );
      prevLabel = outLabel;
    }

    args.addAll([
      '-filter_complex',
      filters.join(';'),
      '-map',
      '[$prevLabel]',
      '-map',
      '0:a?',
      '-c:v',
      'libx264',
      '-preset',
      'medium',
      '-crf',
      '18',
      '-c:a',
      'copy',
      '-movflags',
      '+faststart',
      outputPath,
    ]);
    return args;
  }

  void _previewLinkedRow(Map<String, String> row) {
    setState(() {
      _previewOverlayPath = row['image'] ?? '';
      _previewX = _parseDouble(row['x'], _previewX);
      _previewY = _parseDouble(row['y'], _previewY);
      _previewOverlayW = _parseDouble(row['w'], _previewOverlayW);
      _previewOverlayH = _parseDouble(row['h'], _previewOverlayH);
      _previewOpacity = _parseDouble(row['opacity'], _previewOpacity).clamp(0, 1).toDouble();
    });
    _persistence.setDouble('preview_x', _previewX);
    _persistence.setDouble('preview_y', _previewY);
    _persistence.setDouble('preview_w', _previewOverlayW);
    _persistence.setDouble('preview_h', _previewOverlayH);
    _persistence.setDouble('preview_opacity', _previewOpacity);
    _persistence.setString('preview_overlay_path', _previewOverlayPath);
  }

  void _syncPreviewOverlayValueToLinkedRow(String key, String value) {
    final overlayPath = _previewOverlayPath.trim();
    if (overlayPath.isEmpty) {
      return;
    }
    for (final row in _linkedPonchiRows) {
      if ((row['image'] ?? '').trim() != overlayPath) {
        continue;
      }
      row[key] = value;
      break;
    }
    _persistLinkedPonchiRows();
  }

  Future<void> _initInputVideoPreview(String videoPath) async {
    final localPath = videoPath.trim();
    AppLogger.info('詳細動画編集: initialize開始 path=$localPath');

    if (localPath.isNotEmpty && _videoPreviewController != null && _videoPreviewPath == localPath) {
      AppLogger.info('詳細動画編集: 既存プレビューを再利用 path=$localPath');
      if (!mounted) return;
      setState(() {
        _videoPreviewInitializing = false;
        _videoPreviewError = null;
      });
      return;
    }

    final old = _videoPreviewController;
    if (old != null) {
      AppLogger.info('詳細動画編集: 既存プレビュー停止/破棄');
      await old.pause();
      await old.dispose();
    }
    _videoPreviewController = null;
    _videoPreviewPath = '';

    if (localPath.isEmpty) {
      AppLogger.warn('詳細動画編集: 動画未選択');
      if (!mounted) return;
      setState(() {
        _videoPreviewInitializing = false;
        _videoPreviewError = '動画未選択';
      });
      return;
    }

    final file = File(localPath);
    if (!file.existsSync()) {
      AppLogger.error('詳細動画編集: 動画ファイルが見つかりません path=$localPath');
      if (!mounted) return;
      setState(() {
        _videoPreviewInitializing = false;
        _videoPreviewError = '動画ファイルが見つかりません。PATH: $localPath';
      });
      return;
    }

    if (!mounted) return;
    setState(() {
      _videoPreviewInitializing = true;
      _videoPreviewError = null;
    });

    VideoPlayerController? controller;
    try {
      controller = VideoPlayerController.file(File(localPath));
      await controller.initialize();
      AppLogger.info('詳細動画編集: initialize成功 path=$localPath');
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _videoPreviewController = controller;
        _videoPreviewPath = localPath;
        _videoPreviewInitializing = false;
        _videoPreviewError = null;
      });
    } catch (e, st) {
      debugPrint(e.toString());
      debugPrint(st.toString());
      AppLogger.error('詳細動画編集: initialize失敗 path=$localPath', error: e, stackTrace: st);
      await controller?.dispose();
      if (!mounted) return;
      setState(() {
        _videoPreviewController = null;
        _videoPreviewPath = '';
        _videoPreviewInitializing = false;
        _videoPreviewError = 'プレビュー初期化に失敗しました。PATH: $localPath\nERROR: $e';
      });
    }
  }

  Future<void> _selectInputVideo() async {
    AppLogger.info('詳細動画編集: ファイル選択ダイアログを開きます');
    await _selectFile(
      _inputVideoController,
      const XTypeGroup(label: 'Video', extensions: ['mp4', 'mov', 'mkv']),
    );
    AppLogger.info('詳細動画編集: ファイル選択 path=${_inputVideoController.text.trim()}');
    await _initInputVideoPreview(_inputVideoController.text);
  }

  @override
  void dispose() {
    _inputVideoController.dispose();
    _outputDirController.dispose();
    _videoPreviewController?.dispose();
    _videoPreviewPath = '';
    ProjectState.currentProjectId.removeListener(_projectListener);
    _persistence.dispose();
    super.dispose();
  }

  DataCell _editableOverlayCell(Map<String, String> row, String key, {double width = 80}) {
    return DataCell(
      SizedBox(
        width: width,
        child: TextFormField(
          key: ValueKey('${row['id'] ?? ''}:$key:${row[key] ?? ''}'),
          initialValue: row[key] ?? '',
          decoration: const InputDecoration(isDense: true, border: OutlineInputBorder()),
          onChanged: (value) {
            setState(() {
              row[key] = value;
              final isPreviewTarget = (row['image'] ?? '').trim() == _previewOverlayPath.trim();
              if (!isPreviewTarget) {
                return;
              }
              if (key == 'x') {
                _previewX = _parseDouble(value, _previewX);
              } else if (key == 'y') {
                _previewY = _parseDouble(value, _previewY);
              } else if (key == 'w') {
                _previewOverlayW = _parseDouble(value, _previewOverlayW);
              } else if (key == 'h') {
                _previewOverlayH = _parseDouble(value, _previewOverlayH);
              } else if (key == 'opacity') {
                _previewOpacity = _parseDouble(value, _previewOpacity).clamp(0.0, 1.0);
              }
            });
            _persistLinkedPonchiRows();
          },
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final previewVideoSize = (_videoPreviewController != null &&
            _videoPreviewController!.value.isInitialized)
        ? _videoPreviewController!.value.size
        : null;
    final previewWidthLimit =
        (previewVideoSize != null && previewVideoSize.width > 0) ? previewVideoSize.width : 1920.0;
    final previewHeightLimit =
        (previewVideoSize != null && previewVideoSize.height > 0) ? previewVideoSize.height : 1080.0;

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
                    width: double.infinity,
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey.shade300),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        if (_videoPreviewInitializing) const LinearProgressIndicator(),
                        if (_videoPreviewError != null) ...[
                          const SizedBox(height: 8),
                          Text(
                            _videoPreviewError!,
                            style: TextStyle(color: Theme.of(context).colorScheme.error),
                          ),
                        ],
                        if (_videoPreviewController != null &&
                            _videoPreviewController!.value.isInitialized) ...[
                          const SizedBox(height: 8),
                          AspectRatio(
                            aspectRatio: _videoPreviewController!.value.aspectRatio,
                            child: LayoutBuilder(
                              builder: (context, constraints) {
                                final videoSize = _videoPreviewController!.value.size;
                                final scaleX = videoSize.width > 0
                                    ? constraints.maxWidth / videoSize.width
                                    : 1.0;
                                final scaleY = videoSize.height > 0
                                    ? constraints.maxHeight / videoSize.height
                                    : 1.0;
                                final overlayPath = _previewOverlayPath.trim();
                                final overlayExists =
                                    overlayPath.isNotEmpty && File(overlayPath).existsSync();
                                final overlayWidth = _previewOverlayW > 0 ? _previewOverlayW * scaleX : null;
                                final overlayHeight = _previewOverlayH > 0 ? _previewOverlayH * scaleY : null;

                                return Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    VideoPlayer(_videoPreviewController!),
                                    if (overlayExists)
                                      Positioned(
                                        left: _previewX * scaleX,
                                        top: _previewY * scaleY,
                                        child: IgnorePointer(
                                          child: Opacity(
                                            opacity: _previewOpacity.clamp(0.0, 1.0),
                                            child: SizedBox(
                                              width: overlayWidth,
                                              height: overlayHeight,
                                              child: Image.file(
                                                File(overlayPath),
                                                width: overlayWidth,
                                                height: overlayHeight,
                                                fit: BoxFit.fill,
                                                alignment: Alignment.topLeft,
                                                errorBuilder: (_, __, ___) => const SizedBox.shrink(),
                                              ),
                                            ),
                                          ),
                                        ),
                                      ),
                                  ],
                                );
                              },
                            ),
                          ),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 8,
                            children: [
                              ElevatedButton.icon(
                                onPressed: () {
                                  final c = _videoPreviewController!;
                                  if (c.value.isPlaying) {
                                    AppLogger.info('詳細動画編集: 再生停止');
                                    c.pause();
                                  } else {
                                    AppLogger.info('詳細動画編集: 再生開始');
                                    c.play();
                                  }
                                  setState(() {});
                                },
                                icon: Icon(
                                  _videoPreviewController!.value.isPlaying
                                      ? Icons.pause
                                      : Icons.play_arrow,
                                ),
                                label: Text(
                                  _videoPreviewController!.value.isPlaying ? '一時停止' : '再生',
                                ),
                              ),
                              OutlinedButton.icon(
                                onPressed: () => _initInputVideoPreview(_inputVideoController.text),
                                icon: const Icon(Icons.refresh),
                                label: const Text('再読み込み'),
                              ),
                            ],
                          ),
                        ] else
                          const SizedBox(
                            height: 180,
                            child: Center(child: Text('動画を選択してください')),
                          ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildSliderRow(
                    label: '画像X',
                    value: _previewX,
                    min: 0,
                    max: previewWidthLimit,
                    onChanged: (value) {
                      setState(() {
                        _previewX = value;
                      });
                      _persistence.setDouble('preview_x', value);
                      _syncPreviewOverlayValueToLinkedRow('x', value.toStringAsFixed(0));
                    },
                    displayValue: _previewX.toStringAsFixed(0),
                  ),
                  _buildSliderRow(
                    label: '画像Y',
                    value: _previewY,
                    min: 0,
                    max: previewHeightLimit,
                    onChanged: (value) {
                      setState(() {
                        _previewY = value;
                      });
                      _persistence.setDouble('preview_y', value);
                      _syncPreviewOverlayValueToLinkedRow('y', value.toStringAsFixed(0));
                    },
                    displayValue: _previewY.toStringAsFixed(0),
                  ),
                  _buildSliderRow(
                    label: '画像W',
                    value: _previewOverlayW,
                    min: 0,
                    max: previewWidthLimit,
                    onChanged: (value) {
                      setState(() {
                        _previewOverlayW = value;
                      });
                      _persistence.setDouble('preview_w', value);
                      _syncPreviewOverlayValueToLinkedRow('w', value.toStringAsFixed(0));
                    },
                    displayValue: _previewOverlayW.toStringAsFixed(0),
                  ),
                  _buildSliderRow(
                    label: '画像H',
                    value: _previewOverlayH,
                    min: 0,
                    max: previewHeightLimit,
                    onChanged: (value) {
                      setState(() {
                        _previewOverlayH = value;
                      });
                      _persistence.setDouble('preview_h', value);
                      _syncPreviewOverlayValueToLinkedRow('h', value.toStringAsFixed(0));
                    },
                    displayValue: _previewOverlayH.toStringAsFixed(0),
                  ),
                  _buildSliderRow(
                    label: '透明度',
                    value: _previewOpacity,
                    min: 0,
                    max: 1,
                    onChanged: (value) {
                      setState(() {
                        _previewOpacity = value;
                      });
                      _persistence.setDouble('preview_opacity', value);
                      _syncPreviewOverlayValueToLinkedRow('opacity', value.toStringAsFixed(3));
                    },
                    displayValue: _previewOpacity.toStringAsFixed(2),
                  ),
                  if (_previewOverlayPath.trim().isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: SelectableText('プレビュー対象画像: $_previewOverlayPath'),
                    ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _inputVideoController,
            onChanged: (value) {
              _persistence.setString('input_video', value);
            },
            decoration: InputDecoration(
              labelText: '入力動画（MP4）',
              suffixIcon: IconButton(
                icon: const Icon(Icons.video_file),
                onPressed: _selectInputVideo,
              ),
            ),
          ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerLeft,
            child: OutlinedButton.icon(
              onPressed: () => _initInputVideoPreview(_inputVideoController.text),
              icon: const Icon(Icons.play_circle_outline),
              label: const Text('この動画をプレビュー'),
            ),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _outputDirController,
            decoration: InputDecoration(
              labelText: '出力先ディレクトリ',
              suffixIcon: IconButton(
                icon: const Icon(Icons.folder),
                onPressed: () => _selectDirectory(_outputDirController),
              ),
            ),
          ),
          const SizedBox(height: 8),
          if ((_selectedTitle ?? '').trim().isEmpty)
            const Text('③でタイトルを入力すると、ファイル名が「{タイトル}.mp4」になります。')
          else
            SelectableText('出力ファイル名: ${_resolvedOutputFileName()}\n出力パス: ${_resolvedOutputPath()}'),
          const SizedBox(height: 16),
          Text('ポンチ絵提案テーブルとの共有データ', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          if (_linkedPonchiRows.isEmpty)
            const Text('ポンチ絵提案テーブルのデータがまだありません。')
          else
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: DataTable(
                columns: const [
                  DataColumn(label: Text('開始')),
                  DataColumn(label: Text('終了')),
                  DataColumn(label: Text('画像情報')),
                  DataColumn(label: Text('内容')),
                  DataColumn(label: Text('X')),
                  DataColumn(label: Text('Y')),
                  DataColumn(label: Text('W')),
                  DataColumn(label: Text('H')),
                  DataColumn(label: Text('透明度')),
                  DataColumn(label: Text('プレビュー反映')),
                ],
                rows: _linkedPonchiRows
                    .map(
                      (row) => DataRow(
                        cells: [
                          DataCell(Text(row['start'] ?? '')),
                          DataCell(Text(row['end'] ?? '')),
                          DataCell(SelectableText((row['image'] ?? '').isEmpty ? '未生成' : (row['image'] ?? ''))),
                          DataCell(SizedBox(width: 220, child: Text(row['visual'] ?? ''))),
                          _editableOverlayCell(row, 'x'),
                          _editableOverlayCell(row, 'y'),
                          _editableOverlayCell(row, 'w'),
                          _editableOverlayCell(row, 'h'),
                          _editableOverlayCell(row, 'opacity', width: 96),
                          DataCell(
                            TextButton(
                              onPressed: () => _previewLinkedRow(row),
                              child: const Text('反映'),
                            ),
                          ),
                        ],
                      ),
                    )
                    .toList(),
              ),
            ),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: _isExporting ? null : _startExport,
            icon: const Icon(Icons.movie),
            label: Text(_isExporting ? '書き出し中・・・' : '書き出し'),
          ),
          const SizedBox(height: 12),
          Text('進捗: $_exportStatusMessage'),
          Text(
            '進捗率: ${((_exportProgress ?? 0.0) * 100).toStringAsFixed(1)}%',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          if (_exportJobId != null && _exportJobId!.isNotEmpty)
            SelectableText('最終書き出し Job ID: $_exportJobId'),
          const SizedBox(height: 8),
          LinearProgressIndicator(value: _exportProgress),
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

class ImageCollectForm extends StatefulWidget {
  const ImageCollectForm({super.key});

  @override
  State<ImageCollectForm> createState() => _ImageCollectFormState();
}

class _ImageCollectFormState extends State<ImageCollectForm> {
  final _srtController = TextEditingController();
  final _imageOutputController = TextEditingController(text: '${Directory.current.path}/srt_images');
  final _searchApiKeyController = TextEditingController();
  final _defaultXController = TextEditingController(text: '100');
  final _defaultYController = TextEditingController(text: '200');
  final _defaultWController = TextEditingController(text: '0');
  final _defaultHController = TextEditingController(text: '0');
  final _defaultOpacityController = TextEditingController(text: '1.0');
  late final InputPersistence _persistence;
  String _searchProvider = 'Google';

  @override
  void initState() {
    super.initState();
    _persistence = InputPersistence('image_collect.', scopeListenable: ProjectState.currentProjectId);
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
    if (!mounted) return;
    setState(() {
      _searchProvider = searchProvider ?? _searchProvider;
    });
  }

  @override
  void dispose() {
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
    return ListView(
      children: [
        Text('画像収集', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 16),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
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
              ],
            ),
          ),
        ),
      ],
    );
  }
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
  String? _generatedVideoPath;
  VideoPlayerController? _previewController;
  bool _previewInitializing = false;
  String? _previewErrorMessage;
  String _lastJobLogsText = '';
  List<String> _voicevoxSpeakers = [];
  bool _voicevoxSpeakersLoading = false;
  String? _voicevoxSpeakersError;
  late final VoidCallback _voicevoxUrlListener;
  late final VoidCallback _projectListener;

  bool get _isFlowProject => ProjectState.currentProjectType.value == 'flow';

  String _flowScriptPrefsKey() => 'flow.script_path.${ProjectState.currentProjectId.value}';

  String _lastVideoPathPrefsKey() => 'video_generate.last_video_path.${ProjectState.currentProjectId.value}';

  String _lastSrtPathPrefsKey() => 'video_generate.last_srt_path.${ProjectState.currentProjectId.value}';

  String _lastJobLogsPrefsKey() => 'video_generate.last_job_logs.${ProjectState.currentProjectId.value}';

  Future<String?> _resolveScriptPathForVideo() async {
    if (!_isFlowProject) {
      return _scriptController.text.trim();
    }
    final prefs = await SharedPreferences.getInstance();
    return (prefs.getString(_flowScriptPrefsKey()) ?? '').trim();
  }



  String? _extractVideoPathFromLogs(String logsText) {
    final lines = logsText.split('\n').map((line) => line.trim()).toList().reversed;
    for (final line in lines) {
      final idx = line.indexOf('動画を書き出し中:');
      if (idx >= 0) {
        final candidate = line.substring(idx + '動画を書き出し中:'.length).trim();
        if (candidate.isNotEmpty) {
          return candidate;
        }
      }
    }
    return null;
  }

  Future<void> _initVideoPreview(String videoPath) async {
    final localPath = videoPath.trim();
    AppLogger.info('動画作成: initialize開始 path=$localPath');
    final old = _previewController;
    if (old != null) {
      AppLogger.info('動画作成: 再生停止');
      await old.pause();
      await old.dispose();
    }
    _previewController = null;

    if (localPath.isEmpty) {
      AppLogger.warn('動画作成: 動画未選択');
      if (!mounted) return;
      setState(() {
        _previewInitializing = false;
        _generatedVideoPath = null;
        _previewErrorMessage = '動画未選択';
      });
      return;
    }

    final file = File(localPath);
    if (!file.existsSync()) {
      AppLogger.error('動画作成: 動画ファイルが見つかりません path=$localPath');
      if (!mounted) return;
      setState(() {
        _previewInitializing = false;
        _generatedVideoPath = localPath;
        _previewErrorMessage = '動画ファイルが見つかりません。PATH: $localPath';
      });
      return;
    }

    if (!mounted) return;
    setState(() {
      _previewInitializing = true;
      _generatedVideoPath = localPath;
      _previewErrorMessage = null;
    });

    VideoPlayerController? controller;
    try {
      controller = VideoPlayerController.file(File(localPath));
      await controller.initialize();
      AppLogger.info('動画作成: initialize成功 path=$localPath');
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _previewController = controller;
        _previewInitializing = false;
        _previewErrorMessage = null;
      });
    } catch (e, st) {
      debugPrint(e.toString());
      debugPrint(st.toString());
      AppLogger.error('動画作成: initialize失敗 path=$localPath', error: e, stackTrace: st);
      await controller?.dispose();
      if (!mounted) return;
      setState(() {
        _previewController = null;
        _previewInitializing = false;
        _previewErrorMessage = 'プレビュー初期化に失敗しました。PATH: $localPath\nERROR: $e';
      });
    }
  }


  Future<void> _saveVideoArtifacts({String? videoPath, String? srtPath, String? logsText}) async {
    final prefs = await SharedPreferences.getInstance();
    if (videoPath != null) {
      await prefs.setString(_lastVideoPathPrefsKey(), videoPath);
    }
    if (srtPath != null) {
      await prefs.setString(_lastSrtPathPrefsKey(), srtPath);
    }
    if (logsText != null) {
      await prefs.setString(_lastJobLogsPrefsKey(), logsText);
    }
  }

  Future<void> _loadPersistedVideoArtifacts() async {
    final prefs = await SharedPreferences.getInstance();
    final savedVideoPath = (prefs.getString(_lastVideoPathPrefsKey()) ?? '').trim();
    final savedLogs = prefs.getString(_lastJobLogsPrefsKey()) ?? '';
    final recoveredFromLogs = _extractVideoPathFromLogs(savedLogs) ?? '';
    final resolvedPath = savedVideoPath.isNotEmpty ? savedVideoPath : recoveredFromLogs;
    if (savedVideoPath.isEmpty && recoveredFromLogs.isNotEmpty) {
      await _saveVideoArtifacts(videoPath: recoveredFromLogs);
    }
    if (!mounted) return;
    setState(() {
      _lastJobLogsText = savedLogs;
      _generatedVideoPath = resolvedPath.isEmpty ? null : resolvedPath;
      _previewErrorMessage = null;
    });
    if (resolvedPath.isNotEmpty) {
      await _initVideoPreview(resolvedPath);
    }
  }

  Future<void> _fetchAndPersistJobLogs(String jobId) async {
    try {
      final response = await http
          .get(ApiConfig.httpUri('/jobs/$jobId/logs'))
          .timeout(const Duration(seconds: 10));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        return;
      }
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final logs = (data['logs'] as List<dynamic>? ?? [])
          .map((e) => e.toString())
          .toList();
      final combined = logs.join('\n');
      if (!mounted) return;
      setState(() {
        _lastJobLogsText = combined;
      });
      await _saveVideoArtifacts(logsText: combined);
    } catch (e, st) {
      AppLogger.error('動画作成: ログ取得失敗 job=$jobId', error: e, stackTrace: st);
    }
  }

  Future<void> _watchJobUntilFinished({
    required String jobId,
  }) async {
    for (var i = 0; i < 180; i += 1) {
      if (!mounted || _jobId != jobId) {
        return;
      }
      try {
        await _fetchAndPersistJobLogs(jobId);
        final response = await http
            .get(ApiConfig.httpUri('/jobs/$jobId'))
            .timeout(const Duration(seconds: 10));
        if (response.statusCode < 200 || response.statusCode >= 300) {
          await Future<void>.delayed(const Duration(seconds: 2));
          continue;
        }
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final status = (data['status'] as String? ?? '').trim().toLowerCase();
        if (status == 'completed') {
          final result = data['result'] as Map<String, dynamic>? ?? {};
          final videoPath = (result['video_path'] as String? ?? '').trim();
          final srtPath = (result['srt_path'] as String? ?? '').trim();
          final logsPath = _extractVideoPathFromLogs(_lastJobLogsText) ?? '';
          final resolvedPath = videoPath.isNotEmpty ? videoPath : logsPath;
          if (resolvedPath.isNotEmpty) {
            await _initVideoPreview(resolvedPath);
          }
          await _saveVideoArtifacts(videoPath: resolvedPath, srtPath: srtPath);
          if (!mounted) return;
          setState(() {
            _statusMessage = resolvedPath.isEmpty
                ? 'Completed (動画パス取得待ち: ログを確認してください)'
                : 'Completed';
          });
          widget.jobInProgress.value = false;
          return;
        }
        if (status == 'error') {
          if (!mounted) return;
          setState(() {
            _statusMessage = 'Error: ジョブ失敗';
          });
          widget.jobInProgress.value = false;
          return;
        }
      } catch (e, st) {
        AppLogger.error('動画作成: ジョブ監視失敗 job=$jobId', error: e, stackTrace: st);
      }
      await Future<void>.delayed(const Duration(seconds: 2));
    }
    AppLogger.warn('動画作成: ジョブ監視タイムアウト job=$jobId');
  }

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
      _loadPersistedVideoArtifacts();
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
    _previewController?.dispose();
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
    final persistedVoicevoxUrl = _voicevoxUrlController.text.trim();
    final globalVoicevoxUrl = VoicevoxConfig.baseUrl.value.trim();
    final resolvedVoicevoxUrl = persistedVoicevoxUrl.isNotEmpty
        ? persistedVoicevoxUrl
        : (globalVoicevoxUrl.isNotEmpty ? globalVoicevoxUrl : 'http://127.0.0.1:50021');
    if (persistedVoicevoxUrl != resolvedVoicevoxUrl) {
      _voicevoxUrlController.text = resolvedVoicevoxUrl;
      await _persistence.setString('vv_url', resolvedVoicevoxUrl);
    }
    VoicevoxConfig.baseUrl.value = resolvedVoicevoxUrl;
    await _loadProjectSettings();
    await _loadPersistedVideoArtifacts();
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
    var baseUrl = _voicevoxUrlController.text.trim();
    if (baseUrl.isEmpty) {
      final globalVoicevoxUrl = VoicevoxConfig.baseUrl.value.trim();
      if (globalVoicevoxUrl.isNotEmpty) {
        baseUrl = globalVoicevoxUrl;
        _voicevoxUrlController.text = globalVoicevoxUrl;
      }
    }
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
          ValueListenableBuilder<String>(
            valueListenable: ProjectState.currentProjectType,
            builder: (context, projectType, _) {
              if (projectType == 'flow') {
                return const Padding(
                  padding: EdgeInsets.only(bottom: 12),
                  child: Text('AIフローでは①で確定した台本を自動利用します（原稿ファイル指定は不要）。'),
                );
              }
              return Column(
                children: [
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
                ],
              );
            },
          ),
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
          ExpansionTile(
            title: const Text('TTS エンジン設定'),
            initiallyExpanded: true,
            children: [
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
            ],
          ),
          const SizedBox(height: 12),
          ExpansionTile(
            title: const Text('字幕設定'),
            initiallyExpanded: true,
            children: [
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
                      decoration: const InputDecoration(labelText: '字幕背景の透明度(alpha 0-255)'),
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
            ],
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
          const SizedBox(height: 16),
          Text('生成動画プレビュー', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          if (_previewInitializing) const LinearProgressIndicator(),
          if (_generatedVideoPath != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(_generatedVideoPath!, style: Theme.of(context).textTheme.bodySmall),
            ),
          if (_previewErrorMessage != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                _previewErrorMessage!,
                style: const TextStyle(color: Colors.red),
              ),
            ),
          if (_previewController != null && _previewController!.value.isInitialized)
            Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                AspectRatio(
                  aspectRatio: 1,
                  child: Center(
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(
                        maxWidth: 1000,
                        maxHeight: 500,
                      ),
                      child: AspectRatio(
                        aspectRatio: _previewController!.value.aspectRatio,
                        child: VideoPlayer(_previewController!),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 8),
                VideoProgressIndicator(
                  _previewController!,
                  allowScrubbing: true,
                  padding: const EdgeInsets.symmetric(vertical: 4),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    FilledButton.icon(
                      onPressed: () {
                        final c = _previewController!;
                        if (c.value.isPlaying) {
                          AppLogger.info('動画作成: 再生停止');
                          c.pause();
                        } else {
                          AppLogger.info('動画作成: 再生開始');
                          c.play();
                        }
                        setState(() {});
                      },
                      icon: Icon(_previewController!.value.isPlaying ? Icons.pause : Icons.play_arrow),
                      label: Text(_previewController!.value.isPlaying ? '一時停止' : '再生'),
                    ),
                    const SizedBox(width: 8),
                    OutlinedButton.icon(
                      onPressed: () async {
                        final path = _generatedVideoPath;
                        if (path == null || path.isEmpty) return;
                        await _initVideoPreview(path);
                      },
                      icon: const Icon(Icons.refresh),
                      label: const Text('再読込'),
                    ),
                  ],
                ),
              ],
            )
          else
            const Text('動画生成完了後にここでプレビュー再生できます。'),
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
      _generatedVideoPath = null;
      _previewErrorMessage = null;
      _lastJobLogsText = '';
    });
    unawaited(_saveVideoArtifacts(videoPath: '', srtPath: '', logsText: ''));
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

    final scriptPath = await _resolveScriptPathForVideo();
    if (scriptPath == null || scriptPath.isEmpty) {
      setState(() {
        _statusMessage = _isFlowProject
            ? 'Error: AIフロー台本が未保存です。①台本作成で保存してください。'
            : 'Error: 原稿ファイルを指定してください。';
      });
      widget.jobInProgress.value = false;
      return;
    }

    final payload = {
      'api_key': ApiKeys.gemini.value,
      'script_path': scriptPath,
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
          unawaited(
            _watchJobUntilFinished(
              jobId: jobId,
            ),
          );
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
    this.collapsed = false,
    this.onToggleCollapse,
  });

  final String pageName;
  final ValueListenable<String?>? latestJobId;
  final ValueNotifier<bool>? jobInProgress;
  final bool collapsed;
  final VoidCallback? onToggleCollapse;

  @override
  State<LogPanel> createState() => _LogPanelState();
}

class _LogPanelState extends State<LogPanel> {
  static const _tabLogs = 'Logs';
  static const _tabErrors = 'Errors';
  static const _tabRequests = 'Requests';

  final List<String> _tabs = const [_tabLogs, _tabErrors, _tabRequests];
  final ScrollController _scrollController = ScrollController();
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  String? _currentJobId;
  double? _progress;
  String _progressLabel = '';
  String? _eta;
  bool _autoScroll = true;
  String _activeTab = _tabLogs;

  @override
  void initState() {
    super.initState();
    widget.latestJobId?.addListener(_handleLatestJobIdChanged);
    AppLogger.revision.addListener(_scrollToBottom);
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
    AppLogger.revision.removeListener(_scrollToBottom);
    _subscription?.cancel();
    _channel?.sink.close();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final visibleLogs = _visibleLogs();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: const Color(0xFFF6F8FC),
            border: Border(bottom: BorderSide(color: Colors.grey.shade300)),
          ),
          child: Row(
            children: [
              ..._tabs.map(
                (tab) => Padding(
                  padding: const EdgeInsets.only(right: 6),
                  child: ChoiceChip(
                    label: Text(tab),
                    selected: _activeTab == tab,
                    onSelected: (_) => setState(() {
                      _activeTab = tab;
                    }),
                  ),
                ),
              ),
              const Spacer(),
              IconButton(
                tooltip: 'Clear',
                icon: const Icon(Icons.clear_all, size: 18),
                onPressed: () {
                  AppLogger.clear();
                },
              ),
              IconButton(
                tooltip: 'Copy',
                icon: const Icon(Icons.copy, size: 18),
                onPressed: () async {
                  final text = visibleLogs
                      .map((entry) => '[${_formatTimestamp(entry.timestamp)}] ${entry.level.name.toUpperCase()} ${entry.message}')
                      .join('\n');
                  await Clipboard.setData(ClipboardData(text: text));
                },
              ),
              IconButton(
                tooltip: _autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF',
                icon: Icon(_autoScroll ? Icons.vertical_align_bottom : Icons.do_not_disturb_alt, size: 18),
                onPressed: () {
                  setState(() {
                    _autoScroll = !_autoScroll;
                  });
                },
              ),
              IconButton(
                tooltip: widget.collapsed ? 'Expand' : 'Collapse',
                icon: Icon(widget.collapsed ? Icons.unfold_less : Icons.unfold_more, size: 18),
                onPressed: widget.onToggleCollapse,
              ),
            ],
          ),
        ),
        if (widget.collapsed) const SizedBox.shrink() else ...[
        if (_currentJobId != null) ...[
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 4, 12, 0),
            child: Text('Job: $_currentJobId'),
          ),
          if (_progress != null) ...[
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
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
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 0),
              child: LinearProgressIndicator(value: _progress),
            ),
            if (_eta != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 4, 12, 0),
                child: Text('ETA: $_eta'),
              ),
          ],
        ],
        Expanded(
          child: Container(
            margin: const EdgeInsets.fromLTRB(12, 8, 12, 12),
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: const Color(0xFF101216),
              border: Border.all(color: Colors.black26),
              borderRadius: BorderRadius.circular(8),
            ),
            child: ValueListenableBuilder<int>(
              valueListenable: AppLogger.revision,
              builder: (context, _, __) {
                final logs = _visibleLogs();
                return ListView.builder(
                  controller: _scrollController,
                  itemCount: logs.length,
                  itemBuilder: (context, index) {
                    final entry = logs[index];
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 2),
                      child: Text(
                        '${_formatTimestamp(entry.timestamp)}  ${entry.level.name.toUpperCase().padRight(5)}  ${entry.message}',
                        style: TextStyle(
                          color: _logColor(entry.level),
                          fontFamily: 'monospace',
                          fontSize: 12,
                          height: 1.2,
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
      ],
    );
  }

  List<AppLogEntry> _visibleLogs() {
    final source = AppLogger.entries;
    return source.where((entry) {
      if (_activeTab == _tabErrors) {
        return entry.level == AppLogLevel.error;
      }
      if (_activeTab == _tabRequests) {
        return entry.message.contains('http') || entry.message.contains('API');
      }
      return true;
    }).toList(growable: false);
  }

  void _handleLatestJobIdChanged() {
    final jobId = widget.latestJobId?.value;
    if (jobId == null || jobId.isEmpty || jobId == _currentJobId) {
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
      _progress = null;
      _progressLabel = '';
      _eta = null;
      _channel = channel;
    });
    _setJobInProgress(true);
    AppLogger.info('Connecting to $jobId ...');

    _subscription = channel.stream.listen(
      (event) {
        _handleSocketEvent(event);
      },
      onError: (error) {
        AppLogger.error('WebSocket error', error: error);
        _setJobInProgress(false);
      },
      onDone: () {
        AppLogger.warn('WebSocket closed');
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
      AppLogger.info(event.toString());
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
        AppLogger.error('エラー: ${payload['message']}');
        _setJobInProgress(false);
        return;
      case 'completed':
        AppLogger.info('完了: ${jsonEncode(payload['result'])}');
        _setJobInProgress(false);
        return;
      case 'log':
      default:
        AppLogger.info('${payload['message']}');
        return;
    }
  }

  void _setJobInProgress(bool value) {
    final notifier = widget.jobInProgress;
    if (notifier == null || notifier.value == value) return;
    notifier.value = value;
  }

  void _scrollToBottom() {
    if (!_autoScroll || widget.collapsed) {
      return;
    }
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

  Color _logColor(AppLogLevel level) {
    switch (level) {
      case AppLogLevel.error:
        return const Color(0xFFFF8A80);
      case AppLogLevel.warn:
        return const Color(0xFFFFCC80);
      case AppLogLevel.info:
      default:
        return const Color(0xFFCFD8DC);
    }
  }
}

Future<void> _selectFile(TextEditingController controller, XTypeGroup typeGroup) async {
  AppLogger.info('ファイル選択を開始: ${typeGroup.label}');
  final file = await openFile(acceptedTypeGroups: [typeGroup]);
  if (file == null) {
    AppLogger.warn('ファイル選択キャンセル: ${typeGroup.label}');
    return;
  }
  controller.text = file.path;
  AppLogger.info('ファイル選択: ${file.path}');
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
