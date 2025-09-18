#!/usr/bin/env python3
"""
處遇計畫功能完整測試腳本
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import build_treatment_plan_prompt, generate_treatment_plan_config

def test_case_types():
    """測試不同案件類型的prompt生成"""
    print("🧪 測試不同案件類型的prompt生成...")
    
    case_types = [
        'family_mediation',
        'parent_child', 
        'marriage_counseling',
        'child_protection',
        'domestic_violence',
        'other_family'
    ]
    
    for case_type in case_types:
        print(f"\n📋 測試案件類型: {case_type}")
        
        service_fields = ['protection_services', 'women_family']
        custom_settings = {'style': 'formal', 'notes': f'測試{case_type}案件'}
        
        prompt = build_treatment_plan_prompt(case_type, service_fields, custom_settings)
        
        # 檢查是否包含案件類型特定內容
        assert len(prompt) > 500, f"{case_type}的prompt過短"
        assert '處遇計畫' in prompt, f"{case_type}缺少處遇計畫結構"
        assert f'測試{case_type}案件' in prompt, f"{case_type}自定義設定未包含"
        
        print(f"✅ {case_type} 測試通過")

def test_service_fields():
    """測試不同服務領域組合"""
    print("\n🧪 測試不同服務領域組合...")
    
    service_combinations = [
        ['judicial_correction'],
        ['protection_services', 'children_youth'], 
        ['medical_related', 'psychological_mental', 'disability'],
        ['women_family', 'elderly_longterm_care', 'new_residents']
    ]
    
    for i, fields in enumerate(service_combinations):
        print(f"\n📋 測試組合 {i+1}: {fields}")
        
        prompt = build_treatment_plan_prompt('family_mediation', fields, {})
        
        # 檢查是否包含服務領域策略
        assert '專業領域策略' in prompt, f"組合{i+1}缺少專業策略"
        assert len(prompt) > 400, f"組合{i+1}的prompt過短"
        
        print(f"✅ 組合 {i+1} 測試通過")

def test_custom_settings():
    """測試自定義設置功能"""
    print("\n🧪 測試自定義設置功能...")
    
    settings_tests = [
        {'style': 'formal', 'notes': '正式風格測試'},
        {'style': 'detailed', 'notes': '詳細風格測試'},
        {'style': 'concise', 'notes': '簡潔風格測試'},
        {'notes': '只有備註的測試'}
    ]
    
    for i, settings in enumerate(settings_tests):
        print(f"\n📋 測試設置 {i+1}: {settings}")
        
        prompt = build_treatment_plan_prompt('child_protection', ['protection_services'], settings)
        
        if 'notes' in settings:
            assert settings['notes'] in prompt, f"設置{i+1}備註未包含"
        
        print(f"✅ 設置 {i+1} 測試通過")

def test_config_generation():
    """測試配置文件生成"""
    print("\n🧪 測試配置文件生成...")
    
    # 測試基本配置生成
    config_file = generate_treatment_plan_config(
        'domestic_violence', 
        ['protection_services', 'psychological_mental'],
        {'style': 'detailed', 'notes': '家暴案件特殊處理'}
    )
    
    print(f"📁 生成的配置文件: {config_file}")
    
    # 檢查文件是否存在
    assert os.path.exists(config_file), "配置文件未成功生成"
    
    # 讀取並檢查內容
    with open(config_file, 'r', encoding='utf-8') as f:
        import json
        config = json.load(f)
    
    # 驗證配置結構
    assert 'steps' in config, "配置缺少steps"
    assert len(config['steps']) > 0, "配置steps為空"
    assert 'template' in config['steps'][0], "配置缺少template"
    
    # 檢查prompt內容
    template = config['steps'][0]['template']
    assert '家暴案件特殊處理' in template, "自定義備註未包含在配置中"
    assert '家庭暴力防治和創傷復原' in template, "案件類型重點未包含"
    
    print("✅ 配置文件生成測試通過")
    
    # 清理測試文件
    if os.path.exists(config_file):
        os.remove(config_file)
        print(f"🧹 清理測試文件: {config_file}")

def test_edge_cases():
    """測試邊緣情況"""
    print("\n🧪 測試邊緣情況...")
    
    # 測試空的服務領域
    prompt1 = build_treatment_plan_prompt('family_mediation', [], {})
    assert len(prompt1) > 200, "空服務領域導致prompt過短"
    print("✅ 空服務領域測試通過")
    
    # 測試無效案件類型 - 系統應該優雅處理，不包含特別重點
    prompt2 = build_treatment_plan_prompt('invalid_type', ['protection_services'], {})
    assert len(prompt2) > 200, "無效案件類型導致prompt過短"
    assert '特別重點' not in prompt2, "無效案件類型不應包含特別重點"
    print("✅ 無效案件類型測試通過")
    
    # 測試無效服務領域 - 系統應該優雅處理
    prompt3 = build_treatment_plan_prompt('family_mediation', ['invalid_field'], {})
    assert len(prompt3) > 200, "無效服務領域導致prompt過短"
    print("✅ 無效服務領域測試通過")
    
    # 測試空字串案件類型
    prompt4 = build_treatment_plan_prompt('', ['protection_services'], {})
    assert len(prompt4) > 200, "空字串案件類型導致prompt過短"
    print("✅ 空字串案件類型測試通過")
    
    # 測試None案件類型
    prompt5 = build_treatment_plan_prompt(None, ['protection_services'], {})
    assert len(prompt5) > 200, "None案件類型導致prompt過短"
    print("✅ None案件類型測試通過")
    
    print("✅ 邊緣情況測試通過")

if __name__ == '__main__':
    print("🚀 開始處遇計畫功能完整測試...\n")
    
    try:
        test_case_types()
        test_service_fields() 
        test_custom_settings()
        test_config_generation()
        test_edge_cases()
        
        print("\n🎉 所有測試通過！處遇計畫功能完整且穩定。")
        
    except Exception as e:
        print(f"\n❌ 測試失败: {e}")
        sys.exit(1)