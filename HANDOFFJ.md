# USBカメラ → VLM/YOLO → Chrome 配信デモ 設計書

## 0. このドキュメントについて

Claude.ai (Web) で設計検討した内容を Claude Code (Linux/NucBox EVO X2) に引き継ぐためのハンドオフ文書。
同フォルダ内の `LLaVA設計図.pptx` (元の画面イメージ) と SVG 図 2 枚をあわせて参照すること。

- `LLaVA設計図.pptx` — Chrome 上での画面レイアウト案 (USB カメラ映像 + これはなに? Window + YOLO bbox)
- `01_pipeline_architecture.svg` — プロセス分割と IPC のアーキテクチャ図
- `02_camera_abstraction_layer.svg` — カメラ抽象化レイヤと letterbox 二段構成図

### 設計と実装の差分 (実装後追記)

このドキュメントは「Claude.ai での設計検討」をそのまま残しています。実装中に変わった主要な点:

- **映像配信は WebRTC (aiortc) ではなく MJPEG (`multipart/x-mixed-replace`) over HTTP**。完全オフライン環境 (Wi-Fi OFF / インターネット非接続) で Chrome が ICE host candidate を 1 件も emit しなくなり接続不能になる現象に直面したため、ICE を必要としない経路に切り替えました。詳細は §6 末尾の「実装後の補足」と `TECHNICALJ.md` §5 を参照。
- それ以外 (Capture / CAL / SHM / YOLO / VLM / WS bbox & caption) は本書の設計どおりに実装されています。

## 1. 実現したいこと

NucBox EVO X2 上で以下を同時に動かし、Chrome ブラウザで低遅延に表示するデモ:

1. USB カメラから映像取得 (30fps)
2. Chrome へ低遅延で映像表示 (WebRTC)
3. YOLO11m で物体検出 → bbox を 30fps で映像にオーバーレイ
4. Nemotron Nano Omni で 2 秒に 1 回「なにが見えますか?」を日本語で取得し、Chrome 上の専用ウィンドウに表示

画面イメージは `LLaVA設計図.pptx` を参照。

## 2. 動作環境 (前提)

| 項目 | 内容 |
|------|------|
| マシン | NucBox EVO X2 (AMD Ryzen AI MAX+ 395, gfx1151/RDNA 3.5) |
| メモリ | 48GB unified (BIOS で VRAM 割当済) |
| OS | Ubuntu 24.04.4 LTS (HWE kernel) |
| ROCm | 7.2.2 (`/opt/rocm` 経由 symlink) |
| 環境変数 | `HSA_OVERRIDE_GFX_VERSION=11.5.1` |
| 既存資産 | llama.cpp (ROCm/HIP 版、`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`) |
| 既存資産 | ROCm PyTorch wheels (repo.radeon.com 由来) |
| 既存資産 | WhisperX/Ollama/VOICEVOX のローカル AI パイプラインの経験あり |

## 3. アーキテクチャ概要

`01_pipeline_architecture.svg` 参照。プロセスを 4 つに分け、SharedMemory + asyncio.Queue で疎結合にする。

| プロセス | 役割 | 実装方針 |
|---------|------|----------|
| Capture | USB カメラからフレーム取得し SHM に書き込む | OpenCV + v4l2 + pyudev |
| YOLO11m | 30fps で物体検出、bbox を Queue に積む | Ultralytics + ROCm PyTorch |
| VLM | 0.5fps (2 秒に 1 回) で日本語キャプション生成 | llama.cpp (ROCm/HIP) + Nemotron Nano Omni |
| aiortc server | WebRTC で映像配信 + WS で bbox/caption 配信 | aiortc + FastAPI |

### IPC 設計の要点

- **SharedMemory は "latest frame slot" 1 つだけ**: Capture が常に上書き、YOLO/VLM は推論開始時にスナップショット (`numpy.copy()`)
- **キューには結果のみを積む** (フレーム自体はキューに入れない): 古いフレームでの推論積み残しを防ぐ
- **VLM へのフレーム受け渡しは JPEG 圧縮済みバイト列**: llama.cpp (mtmd) はファイル/バイト列入力なので、SHM の raw frame を `cv2.imencode('.jpg', ...)` してから渡す

## 4. カメラ抽象化レイヤ (CAL) - 重要

`02_camera_abstraction_layer.svg` 参照。要件として「画角の違うカメラ・差し込むポートが変わっても動く」必要があるため、以下を実装する。

### 4.1 デバイス動的検出

- `/dev/video*` の番号は接続順で変わるので **直接指定しない**
- udev の `/dev/v4l/by-id/usb-...` パスか VID/PID で識別
- 起動時に `v4l2-ctl --list-devices` 相当を実行 (Python なら `v4l2-python3` か `subprocess` で `v4l2-ctl` を叩く)
- 1 つの USB カメラが複数 `/dev/videoN` を作るので、`VIDIOC_QUERYCAP` で実際にキャプチャ可能なものを選別

### 4.2 フォーマット交渉

- `v4l2-ctl --list-formats-ext` で対応解像度/FPS/FourCC を確認
- 希望解像度 (1280×720@30fps) に最も近いものを選択
- **MJPEG を優先** (USB 帯域効率が YUYV より圧倒的に良い)
- フォールバック順: MJPG → YUYV

### 4.3 letterbox 二段構成

これが今回の追加要件で最も重要なポイント。

- **CAL 出力 = 1280×720 固定 (letterbox 済 RGB)**: aiortc の送出解像度と同じにする
- **YOLO11m 入力 = 640×640**: Ultralytics の `model.predict(imgsz=640)` が内部で letterbox + 逆変換してくれる (自前実装不要)
- **VLM 入力 = 448×448 程度**: モデル仕様に合わせて単純 resize (letterbox は CAL でやってある)
- **bbox 座標は SHM 正規化フレーム上の絶対座標** に揃えて WS 送信。Chrome 側は 1 段の縮尺だけ考慮すれば描画可能

### 4.4 ホットプラグ対応

- `pyudev.Monitor` で `subsystem='video4linux'` を購読
- remove 時: `cv2.VideoCapture` を安全に close し、黒フレーム + 「カメラ未接続」キャプションを生成
- add 時: 設定ポリシーで再選択し reopen
- **aiortc の VideoTrack は生かしたまま、内部のソースだけ差し替える** (Chrome 接続を切らない)
- `cv2.VideoCapture.read()` が抜き取り時にブロックする可能性あり → 別スレッドで動かしタイムアウト監視 (500ms)

### 4.5 設定ファイル例

```yaml
# config.yaml
camera:
  preferred:
    - by_id: usb-Logitech_HD_Pro_Webcam_C920*
    - vid_pid: "046d:0892"
  fallback: any
  format:
    width: 1280
    height: 720
    fps: 30
    fourcc_priority: [MJPG, YUYV]
  output:
    target: [1280, 720]
    keep_aspect: true
    fisheye_correct: false  # 魚眼補正は明示 ON のみ (機種別キャリブレ必要)
```

## 5. 推論バックエンドの選択 (確定事項)

| 用途 | バックエンド | 備考 |
|------|------------|------|
| Nemotron Nano Omni | llama.cpp (ROCm/HIP) | 既存ビルド資産あり、mtmd で画像入力 |
| YOLO11m | ROCm PyTorch (Ultralytics) | 退避プランあり (下記) |
| 映像配信 | WebRTC (aiortc) | 低遅延優先 |

### 退避プラン: YOLO を CPU/iGPU に逃がす

llama.cpp と PyTorch ROCm を同一 GPU 上で同居させると HIP コンテキスト切り替えコストで両方が劣化する可能性あり。Ultralytics の `model.export(format='onnx')` で ONNX 化しておけば、ONNX Runtime CPU/MIGraphX に切り替え可能。**最初から ONNX エクスポートも併設しておく**こと。

## 6. WebRTC 配信の現実

- aiortc の `VideoStreamTrack` をサブクラス化、`recv()` で最新フレームを返す
- エンコーダは VP8 (デフォルト) か H.264 — **CPU エンコードのみ** (aiortc は ROCm VCN 非対応)
- 1280×720@30fps VP8 なら Ryzen AI MAX+ 395 で余裕
- シグナリングは FastAPI で SDP offer/answer を交換 (20 行程度)
- 同一 LAN なら STUN/TURN 不要、host candidate だけで繋がる

### Chrome 側

- `<video>`: WebRTC track 受信、生映像のみ表示
- `<canvas>` overlay: bbox を WS 受信 → 描画 (30fps)
- `<div>`: caption を WS 受信 → 表示 (0.5fps)
- 同期は基本 "最新の bbox を描画し続ける" で OK。厳密にやるなら `requestVideoFrameCallback` + PTS 合わせ

### 実装後の補足: WebRTC → MJPEG への切替

実装してオフライン環境で動かしたところ、Chrome が ICE host candidate を 1 件も emit しない (`chrome://webrtc-internals` の `onicecandidate` が一度も発火しない、`iceState=new` のまま) という挙動を確認しました。Wi-Fi OFF で loopback 以外の interface が無く、かつ STUN 不要な構成だと Chrome の privacy 機構が gathering 自体を諦めるためです。aioice 側のループバック除外を patch しても、サーバ側に dummy STUN を入れても、trickle ICE を実装しても、Chrome が candidate を出さない限り解決しません。

そこで ICE を経由しない MJPEG over HTTP (`multipart/x-mixed-replace; boundary=frame`) に切り替えました:

- サーバ: `GET /stream.mjpg` で SHM の BGR フレームを `cv2.imencode('.jpg', ...)` してチャンクで yield
- Chrome: `<video>` → `<img src="/stream.mjpg">` に差し替え、`onload` / `onerror` で簡易再接続

ICE / STUN / TURN / aiortc / RTCPeerConnection が一切不要になり、同一 LAN / オフラインともに plain HTTP として素直に動作。レイテンシは WebRTC より一段悪化しますがデモ用途では許容範囲。詳細は `TECHNICALJ.md` §5。

## 7. 推奨する構築順序 (重要)

最初から全部繋ぐと切り分けが大変なので、段階的に立ち上げる。**特に Step 4 が Go/No-Go ポイント**。

1. **Capture + SHM**: 別プロセスから読めることを確認
2. **CAL 単体**: 異なるカメラを差し替えて letterbox 後の出力が安定するか
3. **YOLO11m 単独**: ROCm PyTorch で 30fps 出るか実測
4. **Nemotron Nano Omni 単独**: llama.cpp で 1 回の画像推論時間を測定 (2 秒に間に合うか)
5. **YOLO + VLM 同時実行**: tok/s と fps の劣化を測定 ← **Go/No-Go**
6. **aiortc サーバ**: 最後に追加し、Chrome から受信確認
7. **ホットプラグ**: 最後に pyudev 監視を追加

## 8. プロジェクト構造案

```
~/projects/webcam-vlm-yolo/
├── README.md
├── config.yaml
├── pyproject.toml          # uv または poetry
├── docs/
│   ├── 01_pipeline_architecture.svg
│   ├── 02_camera_abstraction_layer.svg
│   └── LLaVA設計図.pptx
├── src/
│   ├── capture/
│   │   ├── device_manager.py      # v4l2 列挙、by-id 選択
│   │   ├── format_negotiator.py   # 解像度/FPS/FourCC
│   │   ├── frame_normalizer.py    # letterbox + メタデータ
│   │   ├── hotplug_watcher.py     # pyudev 監視
│   │   └── shm_writer.py          # SharedMemory 書き込み
│   ├── inference/
│   │   ├── yolo_worker.py         # YOLO11m proc
│   │   └── vlm_worker.py          # llama.cpp 呼び出し
│   ├── server/
│   │   ├── app.py                 # FastAPI + aiortc
│   │   ├── webrtc_track.py        # VideoStreamTrack
│   │   └── ws_broadcaster.py      # bbox/caption push
│   └── web/
│       ├── index.html
│       ├── overlay.js
│       └── style.css
└── scripts/
    ├── benchmark_yolo.py
    ├── benchmark_vlm.py
    └── list_cameras.py
```

## 9. 未確定事項 (Claude Code で確認/相談すること)

- Nemotron Nano Omni の正確なモデル名と GGUF 入手元 (llama.cpp 対応有無)
- VLM のビジョンエンコーダ入力サイズ (448 か 384 か別か)
- USB カメラ実機での MJPEG 30fps が安定するかの実測
- Chrome での `requestVideoFrameCallback` を使った厳密同期が必要かの判断
