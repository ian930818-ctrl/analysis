# 使用官方 Python 映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製需求檔案和程式碼
COPY requirements.txt ./
COPY . ./

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 開放 5050 port
EXPOSE 5050

# 啟動 Flask
CMD ["sh", "-c", "echo '{\"my_openai_key\": \"'$MY_OPENAI_KEY'\"}' > api_key.json && python app.py"]