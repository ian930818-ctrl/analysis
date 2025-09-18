#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Enhanced NLP Character Analysis System
Tests the ability to identify Chinese names and titles correctly
"""

import requests
import json
import sys

def test_enhanced_nlp():
    """Test the enhanced NLP system with various Chinese text samples"""
    
    # Test URL
    url = "http://localhost:5000/api/analyze-text"
    
    # Test cases with expected results
    test_cases = [
        {
            "name": "Basic Student-Teacher Test",
            "text": "小明是一個學生，他每天和王老師一起學習。小華也是學生，她和小明是好朋友。",
            "expected_characters": ["小明", "王老師", "小華"],
            "description": "Should identify 3 characters: 小明, 王老師, 小華"
        },
        {
            "name": "Multiple Teachers Test", 
            "text": "張老師教數學，李老師教英文。小志和小美都很認真學習。",
            "expected_characters": ["張老師", "李老師", "小志", "小美"],
            "description": "Should identify 4 characters with teacher-student roles"
        },
        {
            "name": "Title-Name Combinations",
            "text": "王教授很嚴格，陳醫生很友善。小明同學表現優秀。",
            "expected_characters": ["王教授", "陳醫生", "小明同學"],
            "description": "Should identify professional titles with names"
        },
        {
            "name": "Dialogue Test",
            "text": "小華問：「老師，這個問題怎麼解？」王老師回答：「我們先看看題目。」",
            "expected_characters": ["小華", "老師", "王老師"],
            "description": "Should identify characters in dialogue context"
        },
        {
            "name": "Family Relations",
            "text": "小明的爸爸是工程師，媽媽是護士。哥哥在讀大學。",
            "expected_characters": ["小明", "爸爸", "媽媽", "哥哥"],
            "description": "Should identify family member titles"
        }
    ]
    
    print("Enhanced NLP Character Analysis Test")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Input Text: {test_case['text']}")
        print(f"Expected: {test_case['expected_characters']}")
        
        try:
            # Send request
            response = requests.post(url, 
                json={"text": test_case['text']},
                headers={"Content-Type": "application/json"},
                timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                characters = result.get('characters', [])
                character_names = [char['name'] for char in characters]
                
                print(f"Found Characters: {character_names}")
                print(f"Character Count: {len(character_names)}")
                
                # Detailed character information
                for char in characters:
                    print(f"  - {char['name']}: {char['description']} (信心度: {char.get('confidence', 0):.2f})")
                    if char.get('methods'):
                        print(f"    提取方法: {', '.join(char['methods'])}")
                    if char.get('behaviors'):
                        for behavior in char['behaviors']:
                            print(f"    {behavior['category']}: {', '.join(behavior['actions'])}")
                
                # Check if expected characters were found
                found_expected = []
                for expected in test_case['expected_characters']:
                    if any(expected in name or name in expected for name in character_names):
                        found_expected.append(expected)
                
                success_rate = len(found_expected) / len(test_case['expected_characters']) * 100
                print(f"Success Rate: {success_rate:.1f}% ({len(found_expected)}/{len(test_case['expected_characters'])})")
                
                if success_rate >= 80:
                    print("✅ TEST PASSED")
                elif success_rate >= 50:
                    print("⚠️  TEST PARTIAL")
                else:
                    print("❌ TEST FAILED")
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
        except Exception as e:
            print(f"❌ Test error: {e}")
        
        print("-" * 40)
    
    print("\nTest Summary Complete")

if __name__ == "__main__":
    test_enhanced_nlp()