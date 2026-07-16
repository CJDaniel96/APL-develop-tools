# APL Develop Tools

AOI（自動光學檢測）開發用的工具集合。目前包含一支 CLI：

| 工具 | 說明 |
| --- | --- |
| [`scripts/crop_images.py`](scripts/crop_images.py) | 依 AOI 機台輸出的 XML 所標記的區域，批次裁切原始影像 |

## 環境需求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- Pillow >= 12.3.0（由 uv 自動安裝）

## 安裝

```bash
uv sync
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

## 開發

程式碼遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)：
80 字元行寬、型別註記、以及 `Args:` / `Returns:` 區段的 docstring。
