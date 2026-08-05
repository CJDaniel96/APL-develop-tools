# APL Develop Tools

AOI（自動光學檢測）開發用的工具集合。目前包含六支 CLI：

| 工具 | 說明 |
| --- | --- |
| [`scripts/crop_images.py`](scripts/crop_images.py) | 依 AOI 機台輸出的 XML 所標記的區域，批次裁切原始影像 |
| [`scripts/crop_components.py`](scripts/crop_components.py) | 依元件清單篩選影像，從同名 XML 讀取 bbox 並裁切元件 |
| [`scripts/group_images.py`](scripts/group_images.py) | 將裁切後的影像依光源篩選，並依 Component Name 分類到子資料夾 |
| [`scripts/rotate_images.py`](scripts/rotate_images.py) | 依 pixel size 判斷方向，將影像統一旋轉為橫向或豎向 |
| [`scripts/yolo_classify.py`](scripts/yolo_classify.py) | 使用 Ultralytics YOLO 推論，依 bbox 與 label 規則分類保存到 OK/NG |
| [`scripts/e2e_infer.py`](scripts/e2e_infer.py) | end-to-end 推論驗證：方向校正 → YOLO → HOAM 類別路由 → anomaly 模型 |

## 環境需求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- Pillow >= 12.3.0（由 uv 自動安裝）
- Ultralytics >= 8.3.0（由 uv 自動安裝）
- `e2e_infer.py` 另外需要 `e2e` extra（anomalib、torch、faiss-cpu 等）

## 安裝

```bash
uv sync
```

要跑 `e2e_infer.py` 時改用：

```bash
uv sync --extra e2e
```

## crop_images.py

遞迴掃描 XML 目錄下的所有 `*.xml`，每個 XML 內含一或多個 `<Image>` 節點
（層級為 `root > Panel > Board > Component > CompImage > Image`）。每個節點帶有
一個 `PicPath` 與 `X1`/`Y1`/`X2`/`Y2` 區域座標。

對每個 `<Image>` 節點，程式會：

1. 以 `MAP` 路徑片段為錨點（anchor），把機台的 `PicPath` 還原成相對於
   image 目錄的路徑：

   ```
   MAP/{YYYYMMDD}/{StationID}/{ProductName}/{BoardID}/{Image}
   ```

   `BoardID` 允許含有空白。
2. 從 `<image-dir>/<相對路徑>` 讀取來源影像。
3. 裁切 `(X1, Y1, X2, Y2)` 區域。
4. 寫到 `<output-dir>/<相同的相對路徑>`，保留原本的目錄結構。

### 基本用法

```bash
uv run scripts/crop_images.py \
    --xml-dir    ./XML \
    --image-dir  ./IMG \
    --output-dir ./OUT
```

先用 `--dry-run` 確認要裁切的內容，不會實際寫檔：

```bash
uv run scripts/crop_images.py -x ./XML -i ./IMG -o ./OUT --dry-run
```

### 參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `-x`, `--xml-dir` | （必填） | 遞迴搜尋 `*.xml` 的目錄 |
| `-i`, `--image-dir` | （必填） | 來源影像根目錄（內含 `MAP\...` 樹狀結構） |
| `-o`, `--output-dir` | （必填） | 輸出根目錄，會鏡射 `MAP\...` 結構 |
| `--anchor` | `MAP` | 用來還原 `PicPath` 的路徑片段 |
| `--on-exists` | `suffix` | 輸出檔已存在時的處理方式：`suffix` / `skip` / `overwrite` |
| `--dry-run` | 關閉 | 只回報會裁切什麼，不寫檔 |
| `-v`, `--verbose` | 關閉 | DEBUG 等級日誌 |
| `-q`, `--quiet` | 關閉 | 只輸出 warning 與 error |

### 輸入格式的容錯

- `PicPath`、`X1`…`Y2` 不論是寫成 XML **attribute** 或 **子元素**都能讀取。
- 欄位名稱比對忽略**大小寫**與 **XML namespace**。
- `PicPath` 支援 Windows（`\`）與 POSIX（`/`）兩種分隔符。
- 錨點比對忽略大小寫；在 case-sensitive 的檔案系統上，找不到完全相符的路徑時
  會逐層以忽略大小寫的方式尋找來源檔。
- 座標會自動正規化（`X1 > X2` 也沒問題）並裁切（clamp）到影像邊界內。

### 檔名衝突處理

**同一次執行**中，兩個不同的區域永遠不會互相覆蓋：若某個輸出路徑在這次執行中
已經產生過，後續的裁切一律加上 `_1`、`_2`… 編號後綴。

`--on-exists` 只作用在**先前執行**就已存在於磁碟上的檔案：

- `suffix`（預設）— 加上編號後綴，保留舊檔。
- `skip` — 跳過，不重新產生。
- `overwrite` — 覆蓋既有檔案。

### 錯誤處理

單一節點或單一檔案的問題只會被記錄並計數，不會中斷整批執行。結束時會輸出統計：

```
INFO Done. Summary:
  xml_files         : 1     # 成功解析的 XML 檔數
  xml_errors        : 0     # 無法解析的 XML 檔數
  image_nodes       : 6     # 掃到的 <Image> 節點數
  written           : 3     # 已寫出的裁切圖（dry-run 時為「將會寫出」）
  skipped_existing  : 0     # 因輸出已存在而跳過
  no_pic_path       : 0     # 沒有 PicPath 的節點
  no_anchor         : 1     # PicPath 中找不到錨點片段
  missing_source    : 1     # 找不到來源影像
  bad_region        : 1     # 座標缺漏、非數值或區域為空
  read_errors       : 0     # 無法讀取或裁切的來源影像
```

離開碼（exit code）：成功為 `0`（找不到任何 XML 檔也是 `0`），
`--xml-dir` 或 `--image-dir` 不存在時為 `2`。

## crop_components.py

輸入根目錄預期含有 `XML` 資料夾，以及根目錄或 `NG` 等子目錄內的影像。元件清單
是 UTF-8 txt 檔，每行一個 Component Name（空白行會忽略）。影像檔名規則：

```
{數字代號}_{時間戳}_{日期}_{機台編號}_{component}_{小板號}_{component}_{小板號}_{光源}.jpg
```

程式會用清單中的名稱解析檔名，因此 Component Name 本身可以含底線。符合清單的影像
會依檔名 stem 尋找 `XML` 目錄下的同名 `.xml`，再從以下階層尋找資料：

```
Panel > Board > Component[CompName="{component}_{小板號}"]
  > CompImage > Image[X1, Y1, X2, Y2]
```

座標會正規化並限制在影像邊界內。輸出會保留相對於輸入根目錄的路徑，例如
`<input>/NG/a.jpg` 會寫成 `<output>/NG/a.jpg`。

### 基本用法

```bash
uv run scripts/crop_components.py \
    --input-dir ./DATA \
    --component-list ./components.txt \
    --output-dir ./CROPPED
```

預設 XML 目錄是 `<input-dir>/XML`；也可以另外指定：

```bash
uv run scripts/crop_components.py -i ./DATA -c ./components.txt \
    -o ./CROPPED --xml-dir ./OTHER_XML
```

建議先驗證檔名、XML 對應及 bbox，不實際寫檔：

```bash
uv run scripts/crop_components.py -i ./DATA -c ./components.txt \
    -o ./CROPPED --dry-run
```

### 參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `-i`, `--input-dir` | （必填） | 影像與 `XML` 所在的輸入根目錄 |
| `-c`, `--component-list` | （必填） | UTF-8 txt，每行一個 Component Name |
| `-o`, `--output-dir` | （必填） | 裁切結果目錄，保留來源相對路徑 |
| `-x`, `--xml-dir` | `<input-dir>/XML` | XML 搜尋目錄 |
| `--ext` | 常見影像格式 | 要掃描的影像副檔名 |
| `--ignore-case` | 關閉 | Component Name 與 XML `CompName` 忽略大小寫 |
| `--on-exists` | `suffix` | 已存在時：`suffix` / `skip` / `overwrite` |
| `--dry-run` | 關閉 | 只驗證與回報，不寫檔 |
| `-v`, `--verbose` | 關閉 | DEBUG 等級日誌 |
| `-q`, `--quiet` | 關閉 | 只輸出 warning 與 error |

如果同一個 Component 下有多個 `Image`，程式會優先比對 `PicPath`、`FileName`
等欄位中的影像檔名，其次比對光源欄位；仍無法唯一判定時會跳過，避免使用錯誤 bbox。
任一已選影像因缺 XML、缺 Component、bbox 無效或讀寫失敗時，整批仍會繼續，最後
回傳離開碼 `1`；輸入路徑／清單無效時為 `2`，全部成功時為 `0`。

## group_images.py

遞迴掃描 image 目錄（通常就是 `crop_images.py` 的輸出）下的所有影像，依檔名判斷
光源、篩選出指定光源的影像，再依 Component Name 分類複製到子資料夾。

檔名規則為：

```
{Component Name}_{Pad ID}_{光源}.jpg
```

光源為 `SolderLight` 或 `UniformLight`。輸出結構：

```
<output-dir>/{Component Name}/{檔名}            # 指定單一光源
<output-dir>/{光源}/{Component Name}/{檔名}      # --light all
```

來源影像一律是**複製**，不會搬移或刪除。

### 基本用法

```bash
uv run scripts/group_images.py \
    --image-dir  ./OUT \
    --output-dir ./GROUPED \
    --light SolderLight
```

先用 `--dry-run` 確認分類結果，不會實際寫檔：

```bash
uv run scripts/group_images.py -i ./OUT -o ./GROUPED -l SolderLight --dry-run
```

兩種光源一次分好（會多一層光源資料夾）：

```bash
uv run scripts/group_images.py -i ./OUT -o ./GROUPED -l all
```

### 參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `-i`, `--image-dir` | （必填） | 遞迴搜尋影像的目錄 |
| `-o`, `--output-dir` | （必填） | 輸出根目錄，每個 component 一個子資料夾 |
| `-l`, `--light` | （必填） | `SolderLight` / `UniformLight` / `all`，忽略大小寫 |
| `--ext` | `.jpg .jpeg .png .bmp .tif .tiff` | 要掃描的副檔名 |
| `--on-exists` | `suffix` | 輸出檔已存在時的處理方式：`suffix` / `skip` / `overwrite` |
| `--dry-run` | 關閉 | 只回報會複製什麼，不寫檔 |
| `-v`, `--verbose` | 關閉 | DEBUG 等級日誌 |
| `-q`, `--quiet` | 關閉 | 只輸出 warning 與 error |

### 檔名解析的容錯

- 光源比對忽略**大小寫**，且是**由右往左**找，因此 `crop_images.py` 加上的
  `_1`、`_2` 編號後綴（`C1_1_SolderLight_1.jpg`）仍能正確解析。
- Component Name 允許含底線：`U5_A_3_UniformLight.jpg` 會解析成
  component `U5_A`、pad `3`。
- 檔名不符規則（找不到光源、或前面不足以構成 `{Component}_{Pad}`）只會記錄
  warning 並計入 `unparsed`，不會中斷整批執行。
- 若 `--output-dir` 位於 `--image-dir` 之內，掃描時會自動排除輸出目錄，重跑不會
  把自己複製出來的檔案再分類一次。

### 檔名衝突處理

不同 board／日期底下可能有**同名**的裁切圖（例如兩塊板子都有
`C1_1_SolderLight.jpg`），而分類後它們會落在同一個 component 資料夾。

規則與 `crop_images.py` 一致：同一次執行中，兩個不同的來源檔永遠不會互相覆蓋，
後來者一律加上 `_1`、`_2`… 編號後綴；`--on-exists` 只作用在**先前執行**就已存在
於磁碟上的檔案（`suffix` / `skip` / `overwrite`）。

> **注意**：預設的 `suffix` 在重跑時會把整批影像再複製一份（因為同名檔已存在，
> 就會全部加後綴）。要重跑請用乾淨的 `--output-dir`，或加上 `--on-exists skip`。

### 統計

```
INFO Done. Summary:
  images_seen       : 10    # 掃到的影像檔數
  copied            : 6     # 已複製（dry-run 時為「將會複製」）
  skipped_existing  : 0     # 因輸出已存在而跳過
  other_light       : 3     # 光源不是指定的那一種
  unparsed          : 1     # 檔名不符命名規則
  copy_errors       : 0     # 複製失敗
```

離開碼（exit code）：成功為 `0`（找不到任何影像也是 `0`），
`--image-dir` 不存在時為 `2`。

## rotate_images.py

依影像實際 pixel size 判斷方向：`width > height` 是橫向（landscape），
`height > width` 是豎向（portrait）。選擇目標方向後，不符合的影像會旋轉 90°，
已符合方向及正方形影像則原樣複製到輸出路徑。判斷不採用 EXIF 顯示方向。

輸入可以是單一影像，也可以是資料夾；資料夾會遞迴搜尋並在輸出目錄保留相對路徑。

### 將資料夾內影像統一為橫向

```bash
uv run scripts/rotate_images.py \
    --input ./IMAGES \
    --output-dir ./LANDSCAPE \
    --target landscape
```

### 將單一影像統一為豎向

```bash
uv run scripts/rotate_images.py \
    --input ./photo.jpg \
    --output-dir ./PORTRAIT \
    --target portrait
```

先用 `--dry-run` 查看每張影像的 pixel size 與預計動作：

```bash
uv run scripts/rotate_images.py \
    -i ./IMAGES -o ./ROTATED -t landscape --dry-run
```

### 參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `-i`, `--input` | （必填） | 單一影像，或要遞迴搜尋的資料夾 |
| `-o`, `--output-dir` | （必填） | 輸出目錄；資料夾輸入會保留相對目錄結構 |
| `-t`, `--target` | （必填） | `landscape`（橫向）或 `portrait`（豎向） |
| `--rotation` | `clockwise` | 需要旋轉時採順時針或逆時針：`clockwise` / `counterclockwise` |
| `--ext` | `.jpg .jpeg .png .bmp .tif .tiff .webp` | 資料夾模式要掃描的副檔名 |
| `--on-exists` | `suffix` | 輸出檔已存在時：`suffix` / `skip` / `overwrite` |
| `--dry-run` | 關閉 | 只檢查與回報，不寫入檔案 |
| `-v`, `--verbose` | 關閉 | DEBUG 等級日誌 |
| `-q`, `--quiet` | 關閉 | 只輸出 warning 與 error |

正方形影像（`width == height`）旋轉 90° 後方向不變，因此會原樣複製，並在統計的
`square` 欄位中顯示。無法讀取或寫入任一影像時，程式會繼續處理其餘檔案，最後以
離開碼 `1` 結束；輸入路徑無效時為 `2`，全部成功時為 `0`。

## yolo_classify.py

使用 Ultralytics YOLO 模型遞迴推論影像，依判定結果存到 `OK` 或 `NG` 資料夾。
每個最終分類會同時保存畫有 bbox、label 與 confidence 的推論效果圖，以及可匯入
CVAT review 的原始影像與 Pascal VOC XML。

### 判定規則

影像必須同時符合以下條件才是 OK：

1. `--ok-label` 的偵測數量精確等於 `--ok-count`。
2. 每個 OK bbox 的中心點都位於影像中央區域。
3. 其他 label 的 bbox 沒有完整落在任一 OK bbox 內。

其他 label 位於所有 OK bbox 外時不影響結果；沒有偵測到 OK label、但有其他 label
時，直接判為 NG。若有多個有效 NG label，影像放在 class id 最小的 label 子資料夾。

輸出結構：

```text
<output-dir>/
├── OK/
│   ├── inference/
│   │   └── image.jpg     # 畫有 YOLO 偵測結果
│   └── original/
│       ├── image.jpg     # 未修改的原始影像
│       └── image.xml     # Pascal VOC annotation
└── NG/
    ├── scratch/          # NG label 名稱
    │   ├── inference/
    │   │   └── image.jpg
    │   └── original/
    │       ├── image.jpg
    │       └── image.xml
    ├── _ok_rule/         # OK 數量或中心位置不符，且沒有有效 NG label
    └── _no_detection/    # 完全沒有 detection
```

XML 會包含該張影像的所有模型 detection（不只影響 OK/NG 判定的框），方便在
CVAT 逐框 review、修正或補標。完全沒有 detection 時仍會產生不含 `<object>` 的
有效 Pascal VOC XML。當檔名衝突且使用預設 `--on-exists suffix` 時，效果圖、原圖
與 XML 會套用相同流水號，維持配對關係。

### 基本用法

例如模型中的 `component_ok` 必須出現一次：

```bash
uv sync
uv run scripts/yolo_classify.py \
    --model ./best.pt \
    --source ./IMAGES \
    --output-dir ./RESULTS \
    --ok-label component_ok \
    --ok-count 1
```

`--center-tolerance 0.25`（預設）代表 OK bbox 的中心點必須位於影像寬與高的中央
50% 區域。若要更嚴格限制在中央 20%，可設為 `0.10`：

```bash
uv run scripts/yolo_classify.py -m ./best.pt -s ./IMAGES -o ./RESULTS \
    --ok-label component_ok --ok-count 1 --center-tolerance 0.10
```

只推論並顯示每張圖的判定，不寫入影像：

```bash
uv run scripts/yolo_classify.py -m ./best.pt -s ./IMAGES -o ./RESULTS \
    --ok-label component_ok --ok-count 1 --dry-run
```

### 主要參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `-m`, `--model` | （必填） | `.pt` 模型路徑或 Ultralytics 模型名稱 |
| `-s`, `--source` | （必填） | 單一影像或要遞迴搜尋的目錄 |
| `-o`, `--output-dir` | （必填） | 輸出根目錄；分類下建立 `inference/`、`original/` |
| `--ok-label` | （必填） | 模型中代表 OK 的完整 label name，區分大小寫 |
| `--ok-count` | （必填） | 必須偵測到的精確 OK 數量，至少為 1 |
| `--center-tolerance` | `0.25` | bbox 中心相對影像中心的水平、垂直容許比例 |
| `--conf` | `0.25` | YOLO confidence threshold |
| `--iou` | `0.7` | YOLO NMS IoU threshold |
| `--imgsz` | `640` | 推論影像尺寸 |
| `--device` | 自動 | 例如 `cpu`、`mps` 或 CUDA device `0` |
| `--max-det` | `300` | 每張影像最大 detection 數 |
| `--on-exists` | `suffix` | 已存在時：`suffix` / `skip` / `overwrite` |
| `--dry-run` | 關閉 | 執行推論與分類，但不寫入影像 |

單張影像推論或儲存失敗不會中斷整批工作，最後離開碼為 `1`；來源／模型／label
無效時為 `2`；全部成功時為 `0`。

## e2e_infer.py

把 APL 的四個推論步驟串成一支程式，並輸出可逐張檢查的驗證報告。

### 流程

```text
1. 方向校正   讀取影像或資料夾，依 pixel size 判斷方向，把直的轉成橫的
2. YOLO 閘門  套用 yolo_classify.py 的 OK/NG 規則與複判素材輸出
              NG -> <output-dir>/NG/<label>/ 後結束；OK -> 進入下一步
3. 類別路由   anomaly_dino / patchcore：用訓練好的 HOAMV2 KNN index 判斷類別
              dinomaly：跳過（單一模型涵蓋所有類別）
4. 異常推論   以該類別對應的 anomalib checkpoint 推論
              -> <output-dir>/anomaly/[<class>/]<OK|NG>/
```

輸出結構：

```text
<output-dir>/
├── OK/                       # 只有加上 --save-ok-annotated 才會產生
│   ├── inference/image.jpg
│   └── original/image.jpg + image.xml
├── NG/<label>/               # 與 yolo_classify.py 相同的 CVAT 複判結構
│   ├── inference/image.jpg        # 畫上 bbox 的結果
│   └── original/image.jpg + image.xml
├── anomaly/
│   └── R1005/                # 有做類別路由時才有這層
│       ├── OK/
│       └── NG/
│           ├── image.jpg          # 實際送進模型的影像
│           └── image_result.png   # 原圖 | heat-map | pred mask
├── report.json               # 執行參數、統計、每張影像的完整軌跡
├── report.csv                # 同上，方便用 Excel 檢視
└── _work/                    # 旋轉後的暫存影像與送入 anomalib 的 symlink
```

`report.csv` 每列記錄一張影像走過的每一站：方向、是否旋轉、YOLO 判定與理由、
HOAM 類別、使用的 checkpoint、anomaly score 與判定、以及所有輸出路徑。

### 安裝

這支程式需要 anomalib 與 HOAM 的相依套件，屬於選用的 `e2e` extra：

```bash
uv sync --extra e2e
```

HOAM 本身不透過套件安裝，執行時用 `--hoam-root` 指向 HOAM repo 即可
（程式會把 `<hoam-root>/src` 加進 `sys.path`）。

### 基本用法

dinomaly：不做類別路由，`--anomaly-model-dir` 直接放 checkpoint。

```bash
uv run scripts/e2e_infer.py \
    --input ./IMAGES --output-dir ./RESULTS \
    --yolo-model ./models/yolo/best.pt \
    --ok-label component_ok --ok-count 1 \
    --anomaly-model dinomaly \
    --anomaly-model-dir ./models/dinomaly
```

patchcore / anomalydino：先用 HOAM 判類別，再挑對應資料夾底下的模型。

```bash
uv run scripts/e2e_infer.py \
    --input ./IMAGES --output-dir ./RESULTS \
    --yolo-model ./models/yolo/best.pt \
    --ok-label component_ok --ok-count 1 \
    --anomaly-model patchcore \
    --anomaly-model-dir ./models/patchcore \
    --anomaly-image-size 256 \
    --hoam-root ../HOAM \
    --hoam-model-path ./models/hoam/best.pt
```

此時 `--anomaly-model-dir` 的結構是每個類別一個子資料夾，名稱要和 HOAM 的
類別名稱一致；每個子資料夾底下遞迴尋找 `*.ckpt`，取最新的一個：

```text
./models/patchcore/
├── R1005/.../model.ckpt
└── C0402/.../model.ckpt
```

HOAM 的 `knn.index`、`dataset.pkl`、`mean_std.json`、`config_used.yaml` 預設
從 `--hoam-model-path` 所在目錄尋找（也就是 `hoam build-knn` 的輸出位置），
需要時再用 `--hoam-index` 等參數個別指定。

只跑方向校正與 YOLO，確認流程與判定，不寫入影像：

```bash
uv run scripts/e2e_infer.py -i ./IMAGES -o ./RESULTS \
    --yolo-model ./models/yolo/best.pt --ok-label component_ok --ok-count 1 \
    --anomaly-model dinomaly --anomaly-model-dir ./models/dinomaly --dry-run
```

### 主要參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `-i`, `--input` | （必填） | 單一影像或要遞迴搜尋的目錄 |
| `-o`, `--output-dir` | （必填） | 輸出根目錄 |
| `--work-dir` | `<output-dir>/_work` | 旋轉後影像與 anomalib 輸入的暫存目錄 |
| `--limit` | 無 | 只處理前 N 張，適合先跑小批驗證 |
| `--target-orientation` | `landscape` | 所有非正方形影像要轉成的方向 |
| `--yolo-model` | （必填） | YOLO `.pt` 模型路徑或名稱 |
| `--ok-label` / `--ok-count` | （必填） | 與 `yolo_classify.py` 完全相同的規則 |
| `--center-tolerance` | `0.25` | OK bbox 中心相對影像中心的容許比例 |
| `--save-ok-annotated` | 關閉 | 另外把 YOLO 判 OK 的複判素材寫到 `OK/` |
| `--anomaly-model` | （必填） | `dinomaly` / `anomaly_dino` / `patchcore`，接受 `anomalydino`、`patch-core` 等別名 |
| `--anomaly-model-dir` | （必填） | 模型根目錄；有類別路由時每個類別一個子資料夾 |
| `--anomaly-ckpt` | 無 | 指定單一 checkpoint，等於跳過類別路由 |
| `--anomaly-image-size` | 無 | `448` 或 `448x448`；未指定時 batch size 會被降為 1，避免不同尺寸無法合批 |
| `--anomaly-threshold` | 無 | 改用分數比較，取代 checkpoint 自帶的 OK/NG 判定 |
| `--no-result-image` | 關閉 | 不輸出 `*_result.png` |
| `--hoam-root` | 無 | HOAM repo 路徑（或其 `src`），未安裝 hoam 套件時必填 |
| `--hoam-model-path` | 見說明 | HOAMV2 `.pt`；`anomaly_dino` / `patchcore` 且未給 `--anomaly-ckpt` 時必填 |
| `--hoam-k` | `1` | KNN 取幾個鄰居，類別採多數決，同票時取最近的 |
| `--accelerator` / `--devices` | `auto` | Lightning 的 accelerator 與 device 設定 |
| `--on-exists` | `suffix` | 已存在時：`suffix` / `skip` / `overwrite` |
| `--dry-run` | 關閉 | 只跑方向校正與 YOLO 判定，不寫影像、不做異常推論 |

單張影像失敗只會記在該列的 `error` 欄位，不會中斷整批；最後只要有任何錯誤離開碼
為 `1`，參數或模型設定不合法為 `2`，全部成功為 `0`。

### macOS 注意事項

faiss 與 torch 各自帶了一份 OpenMP runtime，在 macOS 上執行 HOAM 路由時可能出現
`OMP: Error #15`。加上環境變數即可繼續執行：

```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run scripts/e2e_infer.py ...
```

## 開發

程式碼遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)：
80 字元行寬、型別註記、以及 `Args:` / `Returns:` 區段的 docstring。
