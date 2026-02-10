import 'package:flutter/foundation.dart' show ValueListenable;
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class InputPersistence {
  InputPersistence(
    this.prefix, {
    this.scopeListenable,
    this.scopeNamespace = 'project',
  });

  final String prefix;
  final ValueListenable<String>? scopeListenable;
  final String scopeNamespace;
  SharedPreferences? _prefs;
  final Map<TextEditingController, String> _controllers = {};
  final Map<TextEditingController, VoidCallback> _controllerListeners = {};
  VoidCallback? _scopeListener;
  bool _isApplyingStoredValues = false;

  void registerController(TextEditingController controller, String key) {
    _controllers[controller] = key;
  }

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _prefs = prefs;
    for (final entry in _controllers.entries) {
      void listener() {
        if (_isApplyingStoredValues) {
          return;
        }
        prefs.setString(_fullKey(entry.value), entry.key.text);
      }

      entry.key.addListener(listener);
      _controllerListeners[entry.key] = listener;
    }
    await _loadControllerValues();
    if (scopeListenable != null) {
      _scopeListener = () {
        _loadControllerValues();
      };
      scopeListenable!.addListener(_scopeListener!);
    }
  }

  Future<void> dispose() async {
    final listener = _scopeListener;
    if (listener != null && scopeListenable != null) {
      scopeListenable!.removeListener(listener);
      _scopeListener = null;
    }
    for (final entry in _controllerListeners.entries) {
      entry.key.removeListener(entry.value);
    }
    _controllerListeners.clear();
  }

  Future<String?> readString(String key) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    return prefs.getString(_fullKey(key));
  }

  Future<bool?> readBool(String key) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    return prefs.getBool(_fullKey(key));
  }

  Future<double?> readDouble(String key) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    return prefs.getDouble(_fullKey(key));
  }

  Future<int?> readInt(String key) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    return prefs.getInt(_fullKey(key));
  }

  Future<void> setString(String key, String value) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    await prefs.setString(_fullKey(key), value);
  }

  Future<void> setBool(String key, bool value) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    await prefs.setBool(_fullKey(key), value);
  }

  Future<void> setDouble(String key, double value) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    await prefs.setDouble(_fullKey(key), value);
  }

  Future<void> setInt(String key, int value) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    await prefs.setInt(_fullKey(key), value);
  }

  String _fullKey(String key) {
    final scope = scopeListenable?.value.trim();
    if (scope == null || scope.isEmpty) {
      return '$prefix$key';
    }
    return '$prefix$scopeNamespace.$scope.$key';
  }

  Future<void> _loadControllerValues() async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    _isApplyingStoredValues = true;
    try {
      for (final entry in _controllers.entries) {
        final stored = prefs.getString(_fullKey(entry.value));
        final nextText = stored ?? '';
        if (entry.key.text != nextText) {
          entry.key.text = nextText;
        }
      }
    } finally {
      _isApplyingStoredValues = false;
    }
  }
}
