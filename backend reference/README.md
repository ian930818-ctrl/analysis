# 訪視報告自動生成後端（Docker 版）

本專案是一個彈性、可擴充的 Prompt Engineering 框架，支援多種 LLM（如 Ollama、OpenAI），可根據設定檔自動切換模型，並支援 per-step prompt 設計。所有對話流程、模型參數、輸入檔案皆可由設定檔（run.json、person_graph.json、setting.json）集中管理。

## ⚠️ 重要安全提醒

**在上傳到 GitHub 之前，請務必：**
1. 複製 `api_key.example.json` 為 `api_key.json` 並填入您的 API 金鑰
2. 確認 `api_key.json` 已被 `.gitignore` 忽略
3. 絕對不要將真實的 API 金鑰提交到版本控制系統

---

## 目錄結構

- `app.py`：Flask API 伺服器，負責接收前端請求並執行主流程。
- `run.py`：主流程控制，依據 `run.json` 執行多步驟對話。
- `person_graph.py`：人物關係圖流程腳本，依據 `person_graph.json` 執行 AI 抽取人物關係。
- `prompt_core/`：核心邏輯，包含 prompt 管理與 LLM API 封裝。
- `requirements.txt`：Python 依賴套件。
- `Dockerfile`：Docker 建置腳本。
- `run.json`、`person_graph.json`、`setting.json`：流程與模型設定檔。
- `api_key.example.json`：API 金鑰範本檔案。

---

## 快速開始

1. 下載專案
   ```bash
   git clone <your-repository-url>
   cd <專案資料夾>/backend
   ```

2. 設定 API 金鑰
   ```bash
   cp api_key.example.json api_key.json
   # 編輯 api_key.json，填入您的真實 API 金鑰
   ```

3. 編輯設定檔
   - `setting.json`：設定可用 LLM 模型與 API 金鑰。
   - `run.json`、`person_graph.json`：設定對話流程與每步 prompt。

4. 安裝依賴
   ```bash
   pip install -r requirements.txt
   ```

5. 啟動 Flask 伺服器
   ```bash
   python app.py
   ```

6. （可選）用 Docker 部署
   ```bash
   docker build -t visit-report-backend .
   docker run -e MY_OPENAI_KEY=sk-abc123456789 -p 5050:5050 visit-report-backend
   ```

---

## API 使用說明

- **POST** `/run`  
  - `body` 範例： `{ "text": "請貼上逐字稿內容...", "sessionId": "optional-session-id" }`
  - 回傳：流式 AI 報告內容
- **POST** `/PersonGraph`  
  - `body` 範例： `{ "text": "請貼上逐字稿內容...", "sessionId": "optional-session-id" }`
  - 回傳：流式 AI 產生的人物關係 JSON
- **POST** `/PersonGraphChat`
  - `body` 範例： `{ "message": "修改指令", "currentGraph": "{...}", "transcript": "原始逐字稿", "sessionId": "session-id" }`
  - 回傳：修改後的人物關係圖
- **DELETE** `/cleanup/<session_id>`
  - 清理指定會話的暫存檔案

---

## 設定檔範例

- `setting.json`：
  ```json
  [
    {
      "id": "ollama_llama3.2",
      "platform": "ollama",
      "model": "llama3.2",
      "url": "http://127.0.0.1:11434"
    },
    {
      "id": "openai_gpt-4o",
      "platform": "openai",
      "model": "gpt-4o",
      "openai_api_key": "my_openai_key"
    }
  ]
  ```

- `api_key.json`：
  ```json
  {
    "my_openai_key": "your_actual_openai_api_key_here"
  }
  ```

- `run.json`、`person_graph.json`：請參考專案內範例，或根據需求自訂 prompt 與流程。

---

## 常見問題

- 請確認 `setting.json`、`run.json`、`person_graph.json` 格式正確。
- OpenAI 需填入有效 API key。
- 若需支援新平台，擴充 `prompt_core/chat.py` 的 ChatBot 類即可。
- 若用 Ollama，請確認模型已啟動且有 GPU 支援。

---

## 安全性注意事項

- 絕對不要將 `api_key.json` 提交到版本控制系統
- 在生產環境中使用環境變數來管理敏感資訊
- 定期輪換 API 金鑰
- 定期清理 `temp_sessions/` 目錄中的暫存檔案

---

> 參考來源：[basic-backend-design-for-auto-generating-social-work-visit-reports-using-Docker](https://github.com/iamleoluo/basic-backend-design-for-auto-generating-social-work-visit-reports-using-Docker.git)
