# 起動方法

## 1. Python API サーバー

FastAPI の API サーバーを起動します。

```bash
python -m uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

- ヘルスチェック: `http://localhost:8000/health`
- 動画生成: `POST http://localhost:8000/video/generate`
- ジョブ状態: `GET http://localhost:8000/jobs/{job_id}`
- ログ/進捗: `WS ws://localhost:8000/ws/jobs/{job_id}`

## 2. Flutter UI

Flutter の UI は `flutter_app/` 配下です。

```bash
cd flutter_app
flutter pub get
flutter run
```

### UI 操作
- 左メニューからページを切り替えできます。
- 「動画生成」ページでフォームを入力し、`http://localhost:8000/video/generate` に送信します。
- 右ログの「WebSocket 接続」を押して Job ID を入力すると、`ws://localhost:8000/ws/jobs/{job_id}` のログを受信します。

## 3. 動作の流れ（最小構成）

1. API サーバーを起動。
2. Flutter UI を起動。
3. 動画生成フォームを入力して送信。
4. 返ってきた Job ID を右ログに入力して WebSocket で進捗を確認。
