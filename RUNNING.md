# 起動方法

## 1. Flutter UI（API サーバーも同時起動）

Flutter の UI は `flutter_app/` 配下です。

```bash
cd flutter_app
flutter pub get
flutter run
```

Flutter アプリの起動時に API サーバーも自動で立ち上がる想定です。

- ヘルスチェック: `http://localhost:8000/health`
- 動画生成: `POST http://localhost:8000/video/generate`
- ジョブ状態: `GET http://localhost:8000/jobs/{job_id}`
- ログ/進捗: `WS ws://localhost:8000/ws/jobs/{job_id}`

### UI 操作
- 左メニューからページを切り替えできます。
- 「動画生成」ページでフォームを入力し、`http://localhost:8000/video/generate` に送信します。
- 右ログの「WebSocket 接続」を押して Job ID を入力すると、`ws://localhost:8000/ws/jobs/{job_id}` のログを受信します。

## 2. API サーバーを手動で起動する場合

API サーバーだけを起動する場合は、リポジトリのルートで次のコマンドを実行してください。

```bash
python -m uvicorn backend.api_server:app --host 127.0.0.1 --port 8000
```

## 3. 動作の流れ（最小構成）

1. Flutter UI を起動（API サーバーも同時起動）。
2. 動画生成フォームを入力して送信。
3. 返ってきた Job ID を右ログに入力して WebSocket で進捗を確認。

## Windows の仮想メモリ（ページングファイル）について

このアプリ側から起動時にページングファイルの割り当て量を増やすことはできません。
Windows の設定画面でシステム全体の仮想メモリを増やしてください。

1. **設定** → **システム** → **詳細情報** → **システムの詳細設定**
2. **パフォーマンス** → **設定** → **詳細設定** → **仮想メモリ** → **変更**
3. 「すべてのドライブのページング ファイルのサイズを自動的に管理する」のチェックを外し、
   カスタムサイズを指定して再起動

## Windows 向け配布の具体例（PyInstaller + Flutter 連携）

### 1. PyInstaller の `.spec` 例

`backend/api_server.py` にある FastAPI アプリを PyInstaller で単体実行ファイル化する例です。
UI からは `movie_maker_api.exe` を起動し、`http://localhost:8000` を使う前提です。

> 注意: `datas` や `hiddenimports` はプロジェクト依存です。
> モデルファイルやテンプレート類があれば `datas` に追加してください。

```python
# movie_maker_api.spec
# ビルド例: pyinstaller --clean --noconfirm movie_maker_api.spec
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("backend")

block_cipher = None

a = Analysis(
    ["backend/api_server.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="movie_maker_api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="movie_maker_api",
)
```

### 2. Flutter 側の `Process.start()` 実装例

`flutter_app/lib/main.dart` の初期化処理（例: `initState` など）から、
同梱された API サーバー exe を起動する例です。

> 注意: 実際の配置場所に合わせてパスを調整してください。

```dart
import 'dart:io';
import 'package:path/path.dart' as p;

Process? _apiProcess;

Future<void> startApiServer() async {
  final exeDir = File(Platform.resolvedExecutable).parent.path;
  final apiExePath = p.join(exeDir, 'api', 'movie_maker_api.exe');

  if (!await File(apiExePath).exists()) {
    throw Exception('API サーバーが見つかりません: $apiExePath');
  }

  _apiProcess = await Process.start(
    apiExePath,
    ['--host', '127.0.0.1', '--port', '8000'],
    mode: ProcessStartMode.detachedWithStdio,
  );
}

Future<void> stopApiServer() async {
  _apiProcess?.kill();
  _apiProcess = null;
}
```

### 3. 配布フォルダの整理方法（例）

Flutter ビルド成果物に API サーバーを同梱する例です。

```
MovieMaker/
  MovieMaker.exe                  (Flutter)
  data/                           (Flutter の標準フォルダ)
  api/
    movie_maker_api.exe           (PyInstaller)
    ...必要な DLL / モデル / 設定...
```

この構成を前提に、Flutter から `api/movie_maker_api.exe` を起動します。
API 側は `http://localhost:8000` で待ち受ける想定です。
