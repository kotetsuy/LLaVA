# LLaVA on ROCm — USB カメラ × YOLO11m × Nemotron Nano Omni × Chrome MJPEG

NucBox EVO X2 (Ryzen AI MAX+ 395 / Radeon 8060S, ROCm 7.2.1) 上で USB カメラ映像を Chrome ブラウザに MJPEG (`multipart/x-mixed-replace`) で配信し、同じ映像に対して YOLO11m の物体検出 (30fps) と Nemotron Nano Omni による日本語キャプション (0.5fps) をリアルタイムオーバーレイするデモ。

> **転送経路について**: もともと WebRTC (aiortc) を使っていましたが、完全オフライン環境 (Wi-Fi OFF / インターネット非接続) では Chrome が ICE host candidate を 1 件も emit しなくなり接続不能になる事象があったため、ICE を必要としない MJPEG over HTTP に移行しました。LAN 越し配信も plain HTTP なのでそのまま動作します。

設計の詳細は [`HANDOFFJ.md`](./HANDOFFJ.md) と [`TECHNICALJ.md`](./TECHNICALJ.md) を参照。

---

## 必要なもの

| 項目 | 想定値 |
|------|------|
| マシン | NucBox EVO X2 (AMD Ryzen AI MAX+ 395, gfx1151, 48GB unified) |
| OS | Ubuntu 24.04.4 LTS (HWE kernel) |
| ROCm | 7.2.1 (`/opt/rocm` symlink) |
| Python | 3.12 |
| パッケージ管理 | `uv` (ローカル: `~/.local/bin/uv`) |
| USB カメラ | UVC 対応のもの 1 台 |
| Chrome | 任意の最近のバージョン (同一マシンまたは LAN 内の別端末) |

事前にインストール済みであることを期待:

- `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `tmux` (`sudo apt install tmux`)
- ROCm 7.2.1 (`sudo apt install rocm` 等、AMD 公式手順)
- llama.cpp ROCm/HIP ビルド (`~/llama.cpp/build/bin/llama-server` と `llama-mtmd-cli` がビルド済み)

---

## セットアップ手順

### 1. リポジトリ取得

```bash
git clone <this-repo-url> ~/LLaVA
cd ~/LLaVA
```

### 2. Python 仮想環境と基本依存

```bash
uv venv
uv sync
```

これで `numpy / opencv-python / pyyaml / pyudev` がインストールされ、Step 1 (USB カメラ → SHM) と Step 2 (ホットプラグ対応 CAL) が動く状態になります。

### 3. ROCm 版 PyTorch (Step 3 以降に必要)

PyPI の torch は CUDA 版なので使えません。AMD の ROCm wheel を直接 `wget` してインストール:

```bash
mkdir -p ~/wheels && cd ~/wheels
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

cd ~/LLaVA
uv pip install ~/wheels/torch-*.whl ~/wheels/torchvision-*.whl \
               ~/wheels/torchaudio-*.whl ~/wheels/triton-*.whl
```

### 4. YOLO + ONNX (退避プラン用)

```bash
uv pip install -e .[yolo,onnx]
```

`ultralytics` (YOLO11m を初回 predict 時に自動ダウンロード) と `onnx / onnxruntime` (CPU 版) が入ります。

### 5. サーバ依存 (FastAPI + uvicorn + requests)

```bash
uv pip install -e .[webrtc]
```

extra 名は歴史的経緯で `webrtc` のままですが、現行サーバは MJPEG 配信なので aiortc は実質未使用です (依存解決のために一緒にインストールされるだけ)。

### 6. ROCm 環境変数

`~/.bashrc` などに追加して、新しいシェルで自動的に効くようにしておくと楽:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
```

(`start_all.sh` は内部で再 export するので、シェル設定を忘れていても tmux セッションでは効きます。)

### 7. Nemotron Nano Omni GGUF の準備

```bash
mkdir -p ~/nemotron-3
cd ~/nemotron-3

# unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF (Q4_K_XL)
huggingface-cli download \
  unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
  --local-dir Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF

huggingface-cli download \
  unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  mmproj-F16.gguf \
  --local-dir Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF
```

合計 ~24.5 GB。`config.yaml` の `vlm.model` / `vlm.mmproj` のパスがこれと一致することを確認してください。

### 8. (任意) GPU が見えていることを確認

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True AMD Radeon Graphics
```

---

## 起動と停止

### 一括起動 (推奨)

```bash
cd ~/LLaVA
./start_all.sh
```

これだけで:

1. tmux セッション `llava` を作成
2. window 0: `uv run capture-run` (USB カメラ → SHM、Step 1+2)
3. window 1: `uv run serve` (FastAPI + MJPEG `/stream.mjpg` + YOLO bbox + VLM caption WS、Step 6+7)
4. window 2: `llama-server --reasoning off` (Nemotron 常駐、Step 7b)
5. `http://localhost:8080/` がレスポンスを返すまで最大 30 秒待機
6. Chrome を自動で開く

オプション:

```bash
./start_all.sh --no-browser     # SSH 越しなどで自動 open 不要なとき
./start_all.sh --help
```

セッションへの接続:

```bash
tmux attach -t llava            # ログを直接見る
# Ctrl-b 0 / 1 / 2 で window 切替
# Ctrl-b d でセッションを生かしたまま離脱
```

### 停止

```bash
./stop_all.sh
```

各 window に `Ctrl-C` を送って 5 秒待ち、そのあと `tmux kill-session`。万一プロセスが残っていれば SIGINT → SIGKILL の段階で後始末します。

---

## ブラウザで確認

`./start_all.sh` 実行後、Chrome で `http://localhost:8080/` (または LAN 内別端末から `http://<NucBox の IP>:8080/`) を開くと:

- 中央の `<img src="/stream.mjpg">` に USB カメラ映像 (1280x720, MJPEG)
- 上に半透明 `<canvas>` で YOLO の bbox (色分け、30fps、`person 92%` 形式のラベル)
- 下に半透明ボックスで Nemotron の日本語キャプション (約 50 字、2 秒ごとに更新)
- ステータス行にストリーム状態 / bbox WS / caption の inference 時間と t/s

llama-server がモデルをロードする最初の ~10 秒は caption が `(no caption yet)` のまま、その後更新が始まります。

---

## ステップごとの個別実行 (デバッグ用)

`start_all.sh` を使わず一つずつ起動したいとき:

```bash
# Step 1+2: capture
uv run capture-run                          # 別ターミナル
uv run shm-reader-demo --ticks 10           # SHM 読出しのみ確認
uv run shm-reader-demo --save /tmp/snap.jpg # 1 枚保存
uv run list-cameras                         # /dev/v4l デバイス一覧

# Step 3: YOLO 単独
uv run benchmark-yolo --source synthetic    # synthetic 1280x720 ノイズ
uv run benchmark-yolo --source shm          # 上の capture を起動した上で
uv run export-yolo-onnx --verify            # ONNX 退避プラン

# Step 4: VLM 単独 (mtmd-cli subprocess)
uv run benchmark-vlm --image /tmp/snap.jpg

# Step 5: YOLO + VLM 同居
uv run benchmark-concurrent --frames 600
uv run benchmark-concurrent --no-vlm        # baseline

# Step 6+7: サーバ単体起動
uv run serve                                # T2 相当
~/llama.cpp/build/bin/llama-server -m ... --mmproj ... --reasoning off  # T3 相当
```

---

## トラブルシューティング

### `uv run capture-run` でカメラが見つからない

```bash
ls /dev/v4l/by-id              # USB カメラの symlink が出るか
v4l2-ctl --list-devices        # (要 sudo apt install v4l-utils)
```

`config.yaml` の `camera.preferred[].by_id` が手元のカメラ名と一致しない場合は `fallback: any` 経由で適当に選ばれます。特定のカメラを優先したいときは `by_id: usb-Vendor_Model*` を編集。

### ブラウザに映像が出ない / ステータスが `stream error` のまま

`http://localhost:8080/stream.mjpg` を直接開いて 200 で MJPEG が降ってくるかを確認してください。

- 200 + 黒画面 → `capture-run` がまだ SEARCHING の可能性。`tmux attach -t llava` で window 0 (capture) を見て、`-> CAPTURING dev=...` と `30 fps` が出ているか確認。出ていなければ `/dev/v4l/by-id/` の symlink と `config.yaml` の `camera.preferred[].by_id` が一致しているか見る
- 404 / 500 → serve window のログを確認 (`tmux attach -t llava` → Ctrl-b 1)
- LAN 別端末から繋がらない場合は `sudo ufw allow 8080` でファイアウォール開放

### caption が空のまま (`(no caption yet)`)

llama-server がまだモデルロード中 (~10 秒) か、`--reasoning off` を付け忘れたか。`tmux attach -t llava` → Ctrl-b 1 で `serve` window のログを確認:

```
vlm-runner: caption (1300ms) 'これは...'         ← OK
vlm-runner: empty caption after strip; raw='<think>...'  ← --reasoning off 不足
```

### `start_all.sh` が「session already exists」で失敗

```bash
./stop_all.sh                  # まず停止
./start_all.sh                 # 再起動
```

または `tmux kill-session -t llava` で強制終了。

### モデルロードが極端に遅い (初回 30 秒以上)

21 GB の GGUF を NVMe から SSD にコピー → ページキャッシュに乗せる時間。2 回目以降は ~10 秒に短縮されます。

### YOLO の fp16 を fp32 に戻したい

`config.yaml` の `yolo.half: true` を `false` に。fp32 のほうが精度はわずかに上がるが、Step 5 の同居測定で fp16 のほうが VLM 側の余裕が大きくなったので fp16 を採用しています。

---

## 開発時のセルフチェック

```bash
# 全 Python ファイルの構文チェック
python3 -m compileall -q src scripts && echo OK

# モジュール import 確認 (依存解決の確認も兼ねる)
uv run python -c "from src.server.app import app; print('imports OK')"
```

---

## ライセンス

[`LICENSE`](./LICENSE) を参照。
