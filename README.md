# 我的網頁應用程式

一個使用Python Flask作為後端，HTML/CSS/JavaScript作為前端的簡單網頁應用程式。

## 專案結構
```
my_web_app/
├── backend/
│   ├── app.py              # Flask主應用程式
│   └── requirements.txt    # Python套件清單
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css  # 樣式檔案
│   │   ├── js/
│   │   │   └── main.js    # JavaScript功能
│   │   └── images/        # 圖片資料夾
│   └── templates/
│       └── index.html     # 主頁面
├── .env                   # 環境變數
└── README.md             # 專案說明
```

## 安裝與使用

### 1. 安裝Python (如果尚未安裝)
- 前往 https://www.python.org/downloads/ 下載最新版本
- 安裝時務必勾選 "Add Python to PATH"

### 2. 設置虛擬環境
```bash
# 進入專案目錄
cd my_web_app

# 創建虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
venv\Scripts\activate

# 啟動虛擬環境 (macOS/Linux)
source venv/bin/activate
```

### 3. 安裝套件
```bash
pip install -r backend/requirements.txt
```

### 4. 執行應用程式
```bash
python backend/app.py
```

### 5. 開啟瀏覽器
前往 http://localhost:5000 查看你的網頁

## 功能特色

- ✅ 前後端分離架構
- ✅ RESTful API設計
- ✅ 響應式網頁設計
- ✅ AJAX非同步請求
- ✅ 錯誤處理機制
- ✅ 美觀的UI介面

## 開發指引

### 新增API端點
在 `backend/app.py` 中新增新的路由：

```python
@app.route('/api/new-endpoint', methods=['GET', 'POST'])
def new_endpoint():
    return jsonify({"message": "新的API端點"})
```

### 修改前端樣式
編輯 `frontend/static/css/style.css` 來自訂網頁外觀

### 新增JavaScript功能
在 `frontend/static/js/main.js` 中新增互動功能

## 下一步開發建議

1. 新增資料庫支援 (SQLite/PostgreSQL)
2. 用戶認證系統
3. 檔案上傳功能
4. 更多API端點
5. 部署到雲端平台