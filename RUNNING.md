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
