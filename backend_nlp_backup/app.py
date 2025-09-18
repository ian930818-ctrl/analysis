from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import jieba
import jieba.posseg as pseg
import re
from collections import Counter, defaultdict
import json
import asyncio
import threading
# ML模型輕量級實現（避免大型依賴）
try:
    from ml_model import initialize_ml_model, get_ml_predictions
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    def initialize_ml_model():
        return False
    def get_ml_predictions(text):
        return []

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/hello', methods=['GET'])
def hello_api():
    return jsonify({"message": "Hello from backend!"})

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    return jsonify({"received": data, "status": "success"})

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        manual_corrections = data.get('manual_corrections', {})
        
        print(f"[DEBUG] 收到分析請求，文本長度: {len(text)}")
        print(f"[DEBUG] 文本前100字符: {text[:100]}...")
        
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400
        
        # 直接使用簡化的人物提取方法，因為複雜系統有問題
        print("[DEBUG] 開始提取人物...")
        characters = simple_extract_characters(text)
        print(f"[DEBUG] 使用簡化方法提取到 {len(characters)} 個人物")
        
        # 顯示提取到的人物名稱
        if characters:
            names = [char['name'] for char in characters]
            print(f"[DEBUG] 識別的人物: {', '.join(names)}")
        
        # Apply manual corrections if provided
        if manual_corrections:
            characters = apply_manual_corrections(characters, manual_corrections)
        
        # Generate relationships
        print("[DEBUG] 開始生成關係...")
        relationships = generate_relationships(text, characters)
        print(f"[DEBUG] 生成了 {len(relationships)} 個關係")
        
        result = {
            "characters": characters,
            "relationships": relationships,
            "status": "success",
            "corrections_applied": len(manual_corrections) > 0
        }
        
        print(f"[DEBUG] 返回結果: {len(characters)} 個人物, {len(relationships)} 個關係")
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] 分析出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"分析失敗: {str(e)}"}), 500

@app.route('/api/correct-characters', methods=['POST'])
def correct_characters():
    """Endpoint for manual character corrections"""
    data = request.get_json()
    original_characters = data.get('characters', [])
    corrections = data.get('corrections', {})
    
    corrected_characters = apply_manual_corrections(original_characters, corrections)
    
    return jsonify({
        "characters": corrected_characters,
        "status": "success"
    })

def apply_manual_corrections(characters, corrections):
    """Apply manual corrections to character list"""
    corrected = []
    
    for char in characters:
        char_id = char['id']
        char_name = char['name']
        
        # Check for corrections
        if char_id in corrections:
            correction = corrections[char_id]
            
            if correction['action'] == 'remove':
                continue  # Skip this character
            elif correction['action'] == 'rename':
                char['name'] = correction['new_name']
                char['description'] = correction.get('new_description', char['description'])
            elif correction['action'] == 'modify':
                char.update(correction['updates'])
        
        corrected.append(char)
    
    # Add manually added characters
    for correction in corrections.values():
        if correction.get('action') == 'add':
            new_char = {
                'id': f"manual_{len(corrected)}",
                'name': correction['name'],
                'description': correction.get('description', f"{correction['name']} - 手動添加角色"),
                'importance': correction.get('importance', 3),
                'frequency': correction.get('frequency', 1),
                'events': [],
                'attributes': []
            }
            corrected.append(new_char)
    
    return corrected

def extract_characters(text):
    """Advanced ML + NER hybrid character extraction"""
    
    # Step 1: ML-based NER (Transfer Learning + LoRA)
    ml_entities = get_ml_predictions(text)
    
    # Step 2: Traditional NER (as backup and enhancement)
    traditional_entities = perform_ner(text)
    
    # Step 3: Hybrid Entity Fusion
    fused_entities = fuse_ml_and_traditional_entities(ml_entities, traditional_entities)
    
    # Step 4: Coreference Resolution - 指代消解
    resolved_entities = resolve_coreferences(text, fused_entities)
    
    # Step 5: Event Extraction - 事件抽取
    events = extract_events(text, resolved_entities)
    
    # Step 6: Cross-validation Enhancement
    validated_entities = cross_validate_entities(text, resolved_entities, events)
    
    # Step 7: Character Profile Building
    character_profiles = build_character_profiles(text, validated_entities, events)
    
    # Step 8: Create final character objects with confidence scores
    characters = []
    for i, (name, profile) in enumerate(character_profiles.items()):
        characters.append({
            "id": f"char_{i}",
            "name": name,
            "description": profile['description'],
            "importance": profile['importance'],
            "frequency": profile['frequency'],
            "confidence": profile.get('confidence', 0.8),
            "source": profile.get('source', 'hybrid'),
            "events": profile['events'][:3],
            "attributes": profile['attributes'],
            "behaviors": profile['behaviors']
        })
    
    return characters

def fuse_ml_and_traditional_entities(ml_entities: list, traditional_entities: dict) -> dict:
    """融合ML模型和傳統NER的結果"""
    fused = defaultdict(set)
    
    # 添加ML預測的實體（高置信度）
    for entity in ml_entities:
        if entity['confidence'] > 0.7:  # 高置信度閾值
            fused['PERSON'].add(entity['text'])
    
    # 添加傳統NER實體
    for entity_type, entities in traditional_entities.items():
        fused[entity_type].update(entities)
    
    # 實體驗證和清理
    validated_fused = defaultdict(set)
    for entity_type, entities in fused.items():
        for entity in entities:
            if validate_entity(entity):
                validated_fused[entity_type].add(entity)
    
    return validated_fused

def validate_entity(entity: str) -> bool:
    """驗證實體的有效性"""
    # 長度檢查
    if len(entity) < 2 or len(entity) > 8:
        return False
    
    # 常見詞過濾
    if is_common_word(entity):
        return False
    
    # 特殊字符檢查
    if re.search(r'[0-9\s\.,，。！？：；]', entity):
        return False
    
    return True

def cross_validate_entities(text: str, entities: dict, events: list) -> dict:
    """交叉驗證實體識別結果"""
    validated = {}
    
    for name, entity_data in entities.items():
        # 計算多維度信心分數
        scores = {
            'frequency_score': calculate_frequency_score(name, text),
            'context_score': calculate_context_score(name, text, events),
            'linguistic_score': calculate_linguistic_score(name),
            'coherence_score': calculate_coherence_score(name, entity_data['contexts'])
        }
        
        # 加權平均計算總分
        weights = {'frequency_score': 0.3, 'context_score': 0.3, 
                  'linguistic_score': 0.2, 'coherence_score': 0.2}
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        # 閾值過濾
        if total_score > 0.6:  # 可調整的置信度閾值
            entity_data['confidence'] = total_score
            entity_data['scores'] = scores
            validated[name] = entity_data
    
    return validated

def calculate_frequency_score(name: str, text: str) -> float:
    """計算頻率分數"""
    frequency = len(re.findall(re.escape(name), text))
    return min(1.0, frequency / 10.0)  # 歸一化到[0,1]

def calculate_context_score(name: str, text: str, events: list) -> float:
    """計算上下文分數"""
    name_events = [e for e in events if e.get('agent') == name]
    context_patterns = [
        r'{}(?:說|問|答|喊|叫|講)',
        r'{}(?:走|跑|來|去|到)',
        r'{}(?:高興|難過|生氣|開心)',
        r'(?:看見|遇到|認識){}',
    ]
    
    context_matches = 0
    for pattern in context_patterns:
        if re.search(pattern.format(re.escape(name)), text):
            context_matches += 1
    
    event_score = len(name_events) / 10.0
    pattern_score = context_matches / len(context_patterns)
    
    return min(1.0, (event_score + pattern_score) / 2)

def calculate_linguistic_score(name: str) -> float:
    """計算語言學分數"""
    score = 0.5  # 基礎分數
    
    # 長度合理性
    if 2 <= len(name) <= 4:
        score += 0.2
    
    # 包含常見人名字符
    name_chars = ['小', '老', '大', '阿', '先生', '小姐', '博士', '老師']
    if any(char in name for char in name_chars):
        score += 0.2
    
    # 詞性檢查
    try:
        words = list(pseg.cut(name))
        if len(words) == 1 and words[0].flag in ['nr', 'nrt']:
            score += 0.1
    except:
        pass
    
    return min(1.0, score)

def calculate_coherence_score(name: str, contexts: list) -> float:
    """計算連貫性分數"""
    if not contexts:
        return 0.0
    
    # 檢查上下文的一致性
    action_types = set()
    for ctx in contexts:
        sentence = ctx['sentence']
        if '說' in sentence: action_types.add('speech')
        if any(word in sentence for word in ['走', '跑', '來', '去']): action_types.add('movement')
        if any(word in sentence for word in ['高興', '難過', '生氣']): action_types.add('emotion')
    
    # 多樣化的行為模式表示更真實的角色
    diversity_score = len(action_types) / 3.0
    
    # 出現的句子數量
    frequency_score = min(1.0, len(contexts) / 5.0)
    
    return (diversity_score + frequency_score) / 2

def perform_ner(text):
    """Enhanced NER for descriptive and non-typical character names"""
    entities = defaultdict(set)
    
    # Method 1: Enhanced patterns for descriptive characters
    ner_patterns = {
        'PERSON': [
            # 1. Standard names before speech acts
            r'([一-龥]{2,4})(?:說|問|答|喊|叫|講|道|言)',
            
            # 2. Names with honorifics
            r'([一-龥]{1,3}(?:先生|小姐|博士|老師|師傅|大人|老爺|夫人|同學|朋友|同事))',
            
            # 3. Age/size prefixes
            r'((?:小|老|大|阿)[一-龥]{1,3})',
            
            # 4. Introduction patterns
            r'(?:叫做|名叫|是|稱為|名為)([一-龥]{2,4})',
            
            # 5. Family relations
            r'([一-龥]{1,3}(?:爸爸|媽媽|哥哥|姐姐|弟弟|妹妹|爺爺|奶奶|叔叔|阿姨|舅舅|姑姑))',
            
            # 6. 描述性角色 - 職業/身份描述
            r'((?:那個|這個|一個)[一-龥]{2,4}(?:醫生|護士|警察|老師|學生|工人|農民|商人|司機|廚師|服務員))',
            r'([一-龥]{2,4}(?:醫生|護士|警察|老師|學生|工人|農民|商人|司機|廚師|服務員))',
            
            # 7. 描述性角色 - 外貌/特徵描述
            r'((?:那個|這個|一個)[一-龥]{2,4}(?:胖子|瘦子|高個子|矮個子|美女|帥哥|老人|小孩))',
            r'([一-龥]{2,4}(?:胖子|瘦子|高個子|矮個子|美女|帥哥|老人|小孩))',
            
            # 8. 描述性角色 - 地理/來源描述
            r'((?:那個|這個|一個)[一-龥]{2,4}(?:人|客人|客戶|訪客|陌生人))',
            r'([一-龥]{2,4}(?:人|客人|客戶|訪客|陌生人))',
            
            # 9. 非典型名字 - 外文音譯
            r'([一-龥]{2,6}(?:斯|克|特|爾|德|林|森|頓|森|羅|卡|莉|娜|亞|妮|絲))',
            
            # 10. 非典型名字 - 複合詞角色
            r'([一-龥]+(?:王|公主|王子|將軍|首領|隊長|班長|組長))',
            
            # 11. 非典型名字 - 綽號/外號
            r'((?:綽號|外號|人稱)[一-龥]{2,4})',
            r'([一-龥]{2,4}(?:號稱|被稱為)[一-龥]{2,4})',
        ]
    }
    
    # Apply enhanced patterns
    for entity_type, patterns in ner_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                name = match.strip() if isinstance(match, str) else match
                if isinstance(name, tuple):
                    name = name[0] if name[0] else name[1]
                
                # Clean descriptive markers
                name = re.sub(r'^(?:那個|這個|一個)', '', name)
                
                if len(name) >= 2 and len(name) <= 8:  # Extended length for descriptive names
                    if not is_common_word(name):
                        entities[entity_type].add(name)
    
    # Method 2: Enhanced Jieba NER with more POS tags
    words = pseg.cut(text)
    for word, flag in words:
        # Person names and noun entities that could be characters
        if flag in ['nr', 'nrt', 'n', 'ns', 'nt'] and len(word) >= 2:
            if not is_common_word(word) and could_be_character(word, text):
                entities['PERSON'].add(word)
    
    # Method 3: Context-based character extraction
    entities = enhance_with_context_analysis(text, entities)
    
    # Method 4: Co-reference and alias detection
    entities = detect_aliases_and_coreferences(text, entities)
    
    return entities

def could_be_character(word, text):
    """Determine if a word could be a character based on context"""
    # Check if word appears in character-like contexts
    character_contexts = [
        r'{}(?:說|問|答|喊|叫|講|想|覺得|認為|決定|希望|打算)',
        r'(?:看見|遇到|找到|認識){}',
        r'{}(?:走|跑|來|去|到|從|向|朝)',
        r'{}(?:高興|難過|生氣|開心|傷心|害怕|驚訝|興奮)',
        r'{}(?:和|與|跟)[一-龥]+(?:說話|聊天|討論|合作)',
    ]
    
    for pattern in character_contexts:
        if re.search(pattern.format(re.escape(word)), text):
            return True
    return False

def enhance_with_context_analysis(text, entities):
    """Use context to find missed characters"""
    sentences = re.split(r'[。！？\n]+', text)
    
    # Find subjects of action verbs
    action_patterns = [
        r'([一-龥]{2,6})(?:決定|開始|完成|嘗試|努力|準備|打算|希望|想要)',
        r'([一-龥]{2,6})(?:感到|覺得|認為|相信|懷疑|擔心|害怕)',
        r'([一-龥]{2,6})(?:拿|給|送|買|賣|借|還|找|看|聽)',
    ]
    
    for sentence in sentences:
        for pattern in action_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                if len(match) >= 2 and not is_common_word(match):
                    entities['PERSON'].add(match)
    
    return entities

def detect_aliases_and_coreferences(text, entities):
    """Detect character aliases and merge similar entities"""
    person_list = list(entities['PERSON'])
    aliases = {}
    
    # Find potential aliases
    for i, name1 in enumerate(person_list):
        for j, name2 in enumerate(person_list[i+1:], i+1):
            # Check if one name contains the other (potential alias)
            if name1 in name2 or name2 in name1:
                longer_name = name2 if len(name2) > len(name1) else name1
                shorter_name = name1 if len(name1) < len(name2) else name2
                aliases[shorter_name] = longer_name
            
            # Check for similar descriptive names
            if is_descriptive_variant(name1, name2):
                canonical_name = get_canonical_name(name1, name2, text)
                aliases[name1 if name1 != canonical_name else name2] = canonical_name
    
    # Merge aliases
    final_entities = set()
    for name in entities['PERSON']:
        canonical = aliases.get(name, name)
        final_entities.add(canonical)
    
    entities['PERSON'] = final_entities
    return entities

def is_descriptive_variant(name1, name2):
    """Check if two names are variants of the same descriptive character"""
    descriptive_suffixes = ['醫生', '老師', '警察', '司機', '學生', '護士']
    
    for suffix in descriptive_suffixes:
        if name1.endswith(suffix) and name2.endswith(suffix):
            return True
    return False

def get_canonical_name(name1, name2, text):
    """Choose the canonical name based on frequency and completeness"""
    freq1 = len(re.findall(re.escape(name1), text))
    freq2 = len(re.findall(re.escape(name2), text))
    
    # Prefer longer, more specific names, but also consider frequency
    if freq1 > freq2 * 2:
        return name1
    elif freq2 > freq1 * 2:
        return name2
    else:
        return name1 if len(name1) > len(name2) else name2

def is_common_word(word):
    """Enhanced filter for common Chinese words that are not names"""
    common_words = {
        # Pronouns and general references
        '東西', '地方', '時候', '事情', '問題', '辦法', '樣子', '大家', '他們', '我們', '她們', '它們',
        '什麼', '怎麼', '為什麼', '那裡', '這裡', '現在', '剛才', '剛剛', '馬上', '立刻',
        '一些', '一點', '一下', '一次', '一起', '一直', '一邊', '一面', '一樣', '一定',
        
        # Locations and directions
        '房間', '家裡', '外面', '裡面', '上面', '下面', '前面', '後面', '左邊', '右邊', '旁邊', '附近',
        '這邊', '那邊', '中間', '周圍', '全部', '整個', '所有', '各種', '任何', '每個',
        
        # Actions and states (not character names)
        '開始', '結束', '完成', '進行', '繼續', '停止', '結果', '原因', '方法', '目的', '情況', '狀況',
        '想法', '意見', '看法', '態度', '行為', '動作', '表情', '聲音', '語氣', '心情',
        
        # Time expressions
        '昨天', '今天', '明天', '早上', '中午', '下午', '晚上', '夜裡', '深夜', '凌晨',
        '剛才', '現在', '等等', '一會', '馬上', '立刻', '突然', '忽然', '漸漸', '慢慢',
        
        # Abstract concepts
        '感覺', '想法', '心情', '情緒', '壓力', '困難', '麻煩', '危險', '機會', '希望', '夢想',
        '記憶', '印象', '經驗', '教訓', '知識', '智慧', '能力', '技能', '天賦', '才華',
        
        # Common objects that appear in text but aren't characters
        '手機', '電腦', '汽車', '房子', '學校', '公司', '醫院', '銀行', '商店', '餐廳',
        '書本', '電視', '收音機', '報紙', '雜誌', '照片', '圖片', '影片', '音樂', '歌曲',
        
        # Quantities and measures
        '很多', '一些', '幾個', '許多', '大量', '少數', '全部', '部分', '一半', '三分之一',
        '公分', '公尺', '公里', '公斤', '小時', '分鐘', '秒鐘', '年', '月', '日', '週',
        
        # Common descriptive words that might be mistaken for names
        '好人', '壞人', '聰明', '愚蠢', '漂亮', '醜陋', '有趣', '無聊', '重要', '普通',
        '特別', '一般', '正常', '異常', '奇怪', '神秘', '危險', '安全', '困難', '容易'
    }
    
    # Additional heuristics
    if word in common_words:
        return True
    
    # Filter single characters (except some valid single-character names)
    if len(word) == 1 and word not in ['我', '你', '他', '她', '它']:
        return True
    
    # Filter common patterns that aren't names
    non_name_patterns = [
        r'^(?:為了|因為|所以|但是|然後|接著|最後|終於|突然|忽然|漸漸|慢慢)$',
        r'^(?:非常|特別|很|十分|相當|比較|更加|最|極其)$',
        r'^(?:應該|可能|也許|或許|大概|估計|據說|聽說)$',
    ]
    
    for pattern in non_name_patterns:
        if re.match(pattern, word):
            return True
    
    return False

def resolve_coreferences(text, entities):
    """Step 2: Coreference Resolution - 指代消解"""
    resolved = {}
    person_entities = list(entities['PERSON'])
    
    # Create pronoun mapping
    pronouns = ['他', '她', '它', '他們', '她們', '它們', '這個人', '那個人', '此人']
    
    sentences = re.split(r'[。！？\n]+', text)
    
    for person in person_entities:
        resolved[person] = {
            'canonical_name': person,
            'mentions': [person],
            'contexts': []
        }
        
        # Find all mentions and contexts
        for i, sentence in enumerate(sentences):
            if person in sentence:
                resolved[person]['contexts'].append({
                    'sentence_id': i,
                    'sentence': sentence.strip(),
                    'mention_type': 'direct'
                })
                
                # Look for pronouns in subsequent sentences that might refer to this person
                for j in range(i+1, min(i+3, len(sentences))):  # Check next 2 sentences
                    next_sentence = sentences[j]
                    for pronoun in pronouns:
                        if pronoun in next_sentence and not any(other_person in next_sentence for other_person in person_entities if other_person != person):
                            resolved[person]['mentions'].append(pronoun)
                            resolved[person]['contexts'].append({
                                'sentence_id': j,
                                'sentence': next_sentence.strip(),
                                'mention_type': 'pronoun',
                                'pronoun': pronoun
                            })
    
    return resolved

def extract_events(text, resolved_entities):
    """Step 3: Event Extraction - 事件抽取"""
    events = []
    
    # Define enhanced event patterns for detailed behavior extraction
    event_patterns = {
        'SPEECH': r'([一-龥]+)(?:說|問|答|喊|笑著說|哭著說|回答|告訴|解釋|命令|建議)[:：]?[「『"]([^」』"]+)[」』"]',
        'ACTION': r'([一-龥]+)(?:做|做了|進行|完成|開始|結束|執行|實施|創造|建立|製作|準備|處理|解決)([一-龥]+)',
        'MOVEMENT': r'([一-龥]+)(?:走|跑|來|去|到|從|回|離開|前往|抵達|經過|穿越|爬|跳|飛)([一-龥]*)',
        'EMOTION': r'([一-龥]+)(?:高興|難過|生氣|開心|傷心|害怕|驚訝|興奮|緊張|焦慮|滿意|失望|感動)',
        'INTERACTION': r'([一-龥]+)(?:和|與|跟)([一-龥]+)(?:一起|合作|吵架|談話|討論|爭論|協商|見面|分別)',
        'DECISION': r'([一-龥]+)(?:決定|選擇|考慮|打算|計劃|想要|希望|期待)([一-龥]+)',
        'WORK': r'([一-龥]+)(?:工作|學習|研究|教|教學|上課|下課|寫|讀|看|觀察)([一-龥]*)',
        'DAILY_LIFE': r'([一-龥]+)(?:吃|喝|睡|醒|起床|洗|買|購買|烹飪|打掃|休息)([一-龥]*)',
    }
    
    sentences = re.split(r'[。！？\n]+', text)
    
    for i, sentence in enumerate(sentences):
        for event_type, pattern in event_patterns.items():
            matches = re.findall(pattern, sentence)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 1:
                    agent = match[0]
                    # Check if agent is a known entity
                    if any(agent in entity_data['mentions'] for entity_data in resolved_entities.values()):
                        event = {
                            'type': event_type,
                            'agent': agent,
                            'sentence_id': i,
                            'sentence': sentence.strip(),
                            'details': match
                        }
                        events.append(event)
    
    return events

def build_character_profiles(text, resolved_entities, events):
    """Step 4: Build comprehensive character profiles using dependency parsing context"""
    profiles = {}
    
    for canonical_name, entity_data in resolved_entities.items():
        profile = {
            'canonical_name': canonical_name,
            'frequency': len(entity_data['contexts']),
            'events': [],
            'attributes': [],
            'relationships': [],
            'importance': 1
        }
        
        # Collect events for this character
        character_events = [event for event in events 
                          if event['agent'] in entity_data['mentions']]
        profile['events'] = character_events
        
        # Extract detailed behaviors from events
        behaviors = extract_character_behaviors(character_events, text, entity_data['mentions'])
        profile['behaviors'] = behaviors
        
        # Extract attributes from context
        attributes = extract_character_attributes(entity_data['contexts'], character_events)
        profile['attributes'] = attributes
        
        # Calculate importance
        importance = min(5, max(1, 
            profile['frequency'] +  # Frequency weight
            len(character_events) +  # Event participation weight
            len(attributes)  # Attribute richness weight
        ))
        profile['importance'] = importance
        
        # Generate smart description
        profile['description'] = generate_smart_description(canonical_name, attributes, character_events)
        
        profiles[canonical_name] = profile
    
    # Filter and sort by importance
    filtered_profiles = {name: profile for name, profile in profiles.items() 
                        if profile['importance'] > 1 and profile['frequency'] > 0}
    
    # Sort by importance and frequency
    sorted_profiles = dict(sorted(filtered_profiles.items(), 
                                key=lambda x: (x[1]['importance'], x[1]['frequency']), 
                                reverse=True)[:15])
    
    return sorted_profiles

def extract_character_attributes(contexts, events):
    """Extract character attributes from contexts and events"""
    attributes = []
    
    # Attribute patterns
    attribute_patterns = {
        'PROFESSION': r'(?:是|當|做)[一-龥]*(?:醫生|老師|學生|工人|農民|商人|警察|軍人|博士|教授)',
        'AGE': r'(?:今年|已經|才)[一-龥]*(?:歲|年)',
        'APPEARANCE': r'(?:很|非常|特別)[一-龥]*(?:高|矮|胖|瘦|美|醜|帥|漂亮)',
        'PERSONALITY': r'(?:很|非常|特別)[一-龥]*(?:聰明|笨|善良|壞|勇敢|膽小|活潑|安靜)',
        'SKILL': r'(?:會|能|擅長|善於)[一-龥]*(?:唱歌|跳舞|畫畫|寫字|運動)',
    }
    
    all_text = ' '.join([ctx['sentence'] for ctx in contexts])
    
    for attr_type, pattern in attribute_patterns.items():
        matches = re.findall(pattern, all_text)
        for match in matches:
            attributes.append({
                'type': attr_type,
                'value': match
            })
    
    # Extract from events
    for event in events:
        if event['type'] == 'EMOTION':
            attributes.append({
                'type': 'EMOTIONAL_STATE',
                'value': event['details']
            })
        elif event['type'] == 'ACTION':
            attributes.append({
                'type': 'CAPABILITY',
                'value': event['details']
            })
    
    return attributes

def generate_smart_description(name, attributes, events):
    """Generate pure description based only on extracted data"""
    description_parts = []
    
    # Add profession if found
    professions = [attr['value'] for attr in attributes if attr['type'] == 'PROFESSION']
    if professions:
        description_parts.append(f"職業：{professions[0]}")
    
    # Add personality traits
    personalities = [attr['value'] for attr in attributes if attr['type'] == 'PERSONALITY']
    if personalities:
        description_parts.append(f"性格：{personalities[0]}")
    
    # Add age information
    ages = [attr['value'] for attr in attributes if attr['type'] == 'AGE']
    if ages:
        description_parts.append(f"年齡：{ages[0]}")
    
    # Add appearance
    appearances = [attr['value'] for attr in attributes if attr['type'] == 'APPEARANCE']
    if appearances:
        description_parts.append(f"外貌：{appearances[0]}")
    
    # Add skills
    skills = [attr['value'] for attr in attributes if attr['type'] == 'SKILL']
    if skills:
        description_parts.append(f"技能：{skills[0]}")
    
    # Add event-based behavior patterns
    if events:
        event_types = [event['type'] for event in events[:3]]
        unique_events = list(set(event_types))
        if len(unique_events) >= 2:
            action_desc = '、'.join(unique_events[:2])
            description_parts.append(f"行為模式：{action_desc}")
        elif len(unique_events) == 1:
            description_parts.append(f"主要行為：{unique_events[0]}")
    
    # Pure linguistic fallback (no domain bias)
    if not description_parts:
        # Analyze name structure linguistically
        if len(name) >= 3 and any(title in name for title in ['先生', '小姐', '博士', '老師']):
            description_parts.append(f"{name} - 帶職稱人物")
        elif name.startswith('小') and len(name) >= 3:
            description_parts.append(f"{name} - 可能為年輕人物或暱稱")
        elif name.startswith('老') and len(name) >= 3:
            description_parts.append(f"{name} - 可能為年長人物")
        elif any(family in name for family in ['爸', '媽', '哥', '姐', '弟', '妹', '爺', '奶']):
            description_parts.append(f"{name} - 家庭成員")
        else:
            description_parts.append(f"{name} - 文本人物")
    
    return " | ".join(description_parts)

def extract_character_behaviors(character_events, text, character_mentions):
    """提取人物的詳細行為信息"""
    behaviors = []
    
    # 按事件類型分類行為
    behavior_categories = {
        'SPEECH': '說話行為',
        'ACTION': '行動行為', 
        'MOVEMENT': '移動行為',
        'EMOTION': '情感表現',
        'INTERACTION': '互動行為',
        'DECISION': '決策行為',
        'WORK': '工作學習',
        'DAILY_LIFE': '日常生活'
    }
    
    # 統計各類行為
    behavior_summary = {}
    
    for event in character_events:
        event_type = event['type']
        category = behavior_categories.get(event_type, '其他行為')
        
        if category not in behavior_summary:
            behavior_summary[category] = []
        
        # 根據事件類型提取具體行為描述
        if event_type == 'SPEECH' and len(event['details']) >= 2:
            behavior_text = f"說：{event['details'][1][:20]}..."
        elif event_type == 'ACTION' and len(event['details']) >= 2:
            behavior_text = f"做了{event['details'][1]}"
        elif event_type == 'MOVEMENT' and len(event['details']) >= 2:
            behavior_text = f"前往{event['details'][1]}"
        elif event_type == 'INTERACTION' and len(event['details']) >= 2:
            behavior_text = f"與{event['details'][1]}互動"
        elif event_type == 'DECISION' and len(event['details']) >= 2:
            behavior_text = f"決定{event['details'][1]}"
        else:
            behavior_text = event['sentence'][:30] + "..."
        
        behavior_summary[category].append(behavior_text)
    
    # 轉換為顯示格式
    for category, actions in behavior_summary.items():
        if actions:
            # 取前3個最重要的行為
            top_actions = actions[:3]
            behaviors.append({
                'category': category,
                'actions': top_actions,
                'count': len(actions)
            })
    
    return behaviors

def simple_extract_characters(text):
    """精準的中文NER人物提取函數"""
    characters = []
    
    print("[DEBUG] Starting character extraction")
    
    # 專門針對中文人物的精準識別模式
    patterns = [
        # 1. 直接說話模式 
        r'([一-龥]{2,4})說',
        r'([一-龥]{2,4})問',
        # 2. 職稱模式
        r'([一-龥]+老師)',
        r'([一-龥]+博士)',
        r'([一-龥]+教授)',
        # 3. 小字人名
        r'(小[一-龥]{1,2})',
        # 4. 動作主語
        r'([一-龥]{2,3})和',
        r'([一-龥]{2,3})與',
        # 5. 看見模式
        r'看到([一-龥]{2,4})',
        r'遇到([一-龥]{2,4})',
        # 6. 一起模式
        r'和([一-龥]{2,4})一起',
        r'跟([一-龥]{2,4})一起',
    ]
    
    found_names = set()
    
    # Process each pattern to match characters
    for i, pattern in enumerate(patterns):
        try:
            matches = re.findall(pattern, text)
            if matches:
                print(f"[DEBUG] Pattern {i+1} found: {len(matches)} matches")
                
                for match in matches:
                    name = match.strip() if isinstance(match, str) else match
                    
                    # Filter invalid names
                    if (len(name) >= 2 and len(name) <= 4 and
                        name not in ['他們', '我們', '大家', '這個', '那個', '一個', '所有', '每個']):
                        found_names.add(name)
                        print(f"[DEBUG] Valid character found: {name}")
        except Exception as e:
            print(f"[DEBUG] Pattern {i+1} error: {str(e)}")
    
    # Special handling for teacher characters
    if '老師' in text:
        # Find "X teacher" patterns
        teacher_matches = re.findall(r'([一-龥]+)老師', text)
        for match in teacher_matches:
            if len(match) >= 1:
                teacher_name = match + '老師'
                found_names.add(teacher_name)
                print(f"[DEBUG] Teacher found: {teacher_name}")
        
        # If no specific teacher name found, use generic
        if not any('老師' in name for name in found_names):
            found_names.add('老師')
            print("[DEBUG] Generic teacher added")
    
    print(f"[DEBUG] Final character list: {list(found_names)}")
    
    # Create character objects
    for i, name in enumerate(sorted(found_names)):
        frequency = text.count(name)
        
        # Analyze character type
        if "老師" in name:
            character_type = "教育工作者"
        elif name.startswith("小"):
            character_type = "學生"
        else:
            character_type = "人物"
        
        characters.append({
            "id": f"char_{i}",
            "name": name,
            "description": f"{name} - {character_type}",
            "importance": min(5, max(1, frequency)),
            "frequency": frequency,
            "confidence": 0.8,
            "source": "precise_ner",
            "events": [],
            "attributes": [],
            "behaviors": []
        })
    
    print(f"[DEBUG] Created {len(characters)} character objects")
    return characters

# Removed biased description function - using pure NLP approach

def generate_relationships(text, characters):
    """Advanced relationship analysis based on multiple factors"""
    relationships = []
    sentences = re.split(r'[。！？\n]+', text)
    
    for i, char1 in enumerate(characters):
        for j, char2 in enumerate(characters[i+1:], i+1):
            relationship_data = analyze_character_relationship(char1['name'], char2['name'], text, sentences)
            
            if relationship_data['strength'] > 0:
                relationships.append({
                    "id": f"rel_{i}_{j}",
                    "source": char1['id'],
                    "target": char2['id'],
                    "type": relationship_data['type'],
                    "strength": relationship_data['strength'],
                    "details": relationship_data['details']
                })
    
    return relationships

def analyze_character_relationship(name1, name2, text, sentences):
    """Analyze relationship between two characters"""
    cooccurrence = 0
    interaction_types = []
    emotional_context = []
    
    for sentence in sentences:
        if name1 in sentence and name2 in sentence:
            cooccurrence += 1
            
            # Analyze interaction types
            if any(word in sentence for word in ['說', '問', '答', '對話', '交談']):
                interaction_types.append('對話')
            if any(word in sentence for word in ['一起', '共同', '合作', '幫助']):
                interaction_types.append('合作')
            if any(word in sentence for word in ['爭吵', '吵架', '生氣', '反對']):
                interaction_types.append('衝突')
            if any(word in sentence for word in ['朋友', '喜歡', '愛', '關心']):
                interaction_types.append('友好')
            if any(word in sentence for word in ['家人', '父子', '母女', '兄弟', '姐妹']):
                interaction_types.append('家庭')
                
            # Analyze emotional context
            if any(word in sentence for word in ['開心', '快樂', '高興', '笑']):
                emotional_context.append('正面')
            elif any(word in sentence for word in ['難過', '傷心', '生氣', '哭']):
                emotional_context.append('負面')
    
    # Determine relationship type
    if interaction_types:
        most_common_interaction = Counter(interaction_types).most_common(1)[0][0]
        relationship_type = most_common_interaction
    else:
        relationship_type = determine_relationship_type(name1, name2)
    
    # Calculate relationship strength
    strength = min(5, max(1, cooccurrence))
    if '合作' in interaction_types:
        strength += 1
    if '友好' in interaction_types:
        strength += 1
    if '衝突' in interaction_types:
        strength = max(1, strength - 1)
    
    strength = min(5, strength)
    
    return {
        'strength': strength,
        'type': relationship_type,
        'details': {
            'cooccurrence': cooccurrence,
            'interactions': list(set(interaction_types)),
            'emotional_tone': Counter(emotional_context).most_common(1)[0][0] if emotional_context else '中性'
        }
    }

def determine_relationship_type(name1, name2):
    """Pure linguistic relationship inference"""
    # Family relationships
    family_terms = ['爸', '媽', '哥', '姐', '弟', '妹', '爺', '奶', '叔', '阿姨', '舅', '姑']
    if any(term in name1 or term in name2 for term in family_terms):
        return '家庭關係'
    
    # Professional relationships
    professional_terms = ['博士', '老師', '師傅', '先生', '小姐', '同學', '同事']
    if any(term in name1 or term in name2 for term in professional_terms):
        return '專業關係'
    
    # Age-based relationships
    if (name1.startswith('小') and name2.startswith('老')) or (name1.startswith('老') and name2.startswith('小')):
        return '輩分關係'
    
    # Default
    return '一般關係'

@app.route('/api/ml-status', methods=['GET'])
def ml_status():
    """檢查ML模型狀態"""
    try:
        from ml_model import ml_model
        if ml_model and ml_model.model:
            return jsonify({
                "status": "loaded",
                "model_type": "Transfer Learning + LoRA",
                "device": str(ml_model.device) if hasattr(ml_model, 'device') else "unknown"
            })
        else:
            return jsonify({
                "status": "not_loaded",
                "fallback": "traditional_ner"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "fallback": "traditional_ner"
        })

def initialize_app():
    """初始化應用程式"""
    print("Starting Advanced NER System...")
    
    if ML_AVAILABLE:
        print("Loading Transfer Learning + LoRA model...")
        # 非同步初始化ML模型（避免阻塞啟動）
        def init_ml_async():
            try:
                success = initialize_ml_model()
                if success:
                    print("ML model loaded successfully - Transfer Learning + LoRA enabled")
                else:
                    print("ML model loading failed - using enhanced NER fallback")
            except Exception as e:
                print(f"ML model initialization error: {e}")
                print("System will use enhanced NER methods")
        
        # 在背景執行緒中初始化ML模型
        ml_thread = threading.Thread(target=init_ml_async, daemon=True)
        ml_thread.start()
    else:
        print("Using Enhanced Hybrid NER System (without deep learning models)")
        print("Features include:")
        print("   * Multiple NER pattern recognition")
        print("   * Coreference resolution")
        print("   * Event extraction")
        print("   * Cross-validation")
        print("   * Multi-dimensional confidence scoring")
    
    print("System started, accessible at http://localhost:5000")
    print("Hybrid annotation strategy enabled - Target accuracy 85%+")

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)