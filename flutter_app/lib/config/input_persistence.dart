import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class InputPersistence {
  InputPersistence(this.prefix);

  final String prefix;
  SharedPreferences? _prefs;
  final Map<TextEditingController, String> _controllers = {};
  final List<VoidCallback> _disposeCallbacks = [];

  void registerController(TextEditingController controller, String key) {
    _controllers[controller] = key;
  }

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _prefs = prefs;
    for (final entry in _controllers.entries) {
      final stored = prefs.getString('$prefix${entry.value}');
      if (stored != null) {
        entry.key.text = stored;
      }
      void listener() {
        prefs.setString('$prefix${entry.value}', entry.key.text);
      }

      entry.key.addListener(listener);
      _disposeCallbacks.add(() => entry.key.removeListener(listener));
    }
  }

  Future<void> dispose() async {
    for (final callback in _disposeCallbacks) {
      callback();
    }
    _disposeCallbacks.clear();
  }

  Future<String?> readString(String key) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    return prefs.getString('$prefix$key');
  }

  Future<bool?> readBool(String key) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    return prefs.getBool('$prefix$key');
  }

  Future<double?> readDouble(String key) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    return prefs.getDouble('$prefix$key');
  }

  Future<int?> readInt(String key) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    return prefs.getInt('$prefix$key');
  }

  Future<void> setString(String key, String value) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    await prefs.setString('$prefix$key', value);
  }

  Future<void> setBool(String key, bool value) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    await prefs.setBool('$prefix$key', value);
  }

  Future<void> setDouble(String key, double value) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    await prefs.setDouble('$prefix$key', value);
  }

  Future<void> setInt(String key, int value) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    await prefs.setInt('$prefix$key', value);
  }
}
