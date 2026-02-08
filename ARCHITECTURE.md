# Flutter UI + Python API Refactor Plan

## Goal
Tkinter GUI を Flutter に置き換え、既存の Python 動画生成ロジックを API として再利用する構成に移行する。UI は「左メニュー + 中央フォーム + 右ログ/進捗」を維持する。Flutter 側は HTTP/WebSocket または標準入出力で接続できるように設計する。

## Proposed Architecture

### 1) Python API (HTTP/WebSocket)
- FastAPI を使い、動画生成・ログ配信を API 経由で提供する。
- 長時間処理はジョブ化し、HTTP でジョブ作成、WebSocket でログ/進捗をストリーム配信する。

**Implemented endpoints (prototype)**
- `POST /video/generate`
  - 動画生成ジョブを作成し、`job_id` を返す。
- `GET /jobs/{job_id}`
  - 進捗やエラー、結果の取得。
- `WS /ws/jobs/{job_id}`
  - ログ/進捗イベントをリアルタイム配信。

### 2) Flutter UI (Desktop / Web)
- **左メニュー**: 既存のページ構成をそのままタブ化。
- **中央フォーム**: 各ページの入力フォーム。API に対応する request payload を生成。
- **右ログ**: WebSocket から受信したログ/進捗を表示。

### 3) Standard I/O (Optional)
- ローカル CLI 連携が必要なら、
  - Flutter 側の `Process` で Python を起動し、JSON Lines で入出力する方式も可能。
  - ただし GUI で複数ジョブ/ログを扱うなら HTTP/WebSocket の方が扱いやすい。

## Migration Strategy
1. **API 層の追加**: `new_video_gui20.py` の `generate_video()` を API から呼び出す。
2. **Flutter UI 実装**: 既存の UI セクションを Flutter に再実装。
3. **機能の段階移行**: 動画生成→台本生成→タイトル生成→素材生成→動画編集の順に API 化。

## Notes
- 現在の Tkinter 実装は保持しつつ、API から既存ロジックを再利用することで段階移行を可能にする。
- 将来的には `generate_video()` とその依存関数を専用モジュールに切り出すことで API サーバー依存を低減できる。
