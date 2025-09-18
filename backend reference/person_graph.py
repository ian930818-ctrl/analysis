from prompt_core.prompt import PromptManager, PromptLibrary
import json
import sys
import argparse
import os
import re



def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='運行人物關係圖生成腳本')
    parser.add_argument('--session-id', required=True, help='會話ID')
    parser.add_argument('--input-file', required=True, help='輸入文件路徑')
    parser.add_argument('--config-file', default='person_graph.json', help='配置文件路徑')
    
    args = parser.parse_args()
    
    # 檢查輸入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"錯誤：輸入文件 {args.input_file} 不存在", file=sys.stderr)
        return
    
    # 檢查配置文件是否存在
    if not os.path.exists(args.config_file):
        print(f"錯誤：配置文件 {args.config_file} 不存在", file=sys.stderr)
        return

    with open(args.config_file, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    default_model_id = run_data.get("default_model_id")
    steps = run_data.get("steps", [])

    prompt_lib = PromptLibrary(args.config_file)
    pm = PromptManager(default_model_id=default_model_id)
    conversation_id = f"person_graph_conv_{args.session_id}"  # 使用會話ID創建唯一的對話ID
    pm.create_conversation(conversation_id)

    with open(args.input_file, "r", encoding="utf-8") as f:
        input_text = f.read()

    print(f"開始處理人物關係圖，會話 {args.session_id}", file=sys.stderr)

    for step in steps:
        label = step.get("label")
        model_id = step.get("model_id")
        temperature = step.get("temperature")
        prompt = prompt_lib.get_prompt(label)
        if not prompt:
            continue
        if "template" in prompt:
            q = prompt["template"].format(input=input_text)
        else:
            q = prompt.get("question", "")
        print("-----------------------------------------", file=sys.stderr)
        print(f"[人物關係圖 {args.session_id}] [Q] {q}", file=sys.stderr)
        for chunk in pm.chat(conversation_id, q, model_id=model_id, temperature=temperature, stream=True, as_generator=True):
            print(json.dumps({"content": chunk}), flush=True)
        print("\n", file=sys.stderr)
        print("-----------------------------------------", file=sys.stderr)

    print(f"人物關係圖會話 {args.session_id} 處理完成", file=sys.stderr)

if __name__ == "__main__":
    main() 