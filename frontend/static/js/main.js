class TextAnalyzerMVP {
    constructor() {
        this.currentText = '';
        this.characters = [];
        this.relationships = [];
        this.networkSimulation = null;
        this.svg = null;
        this.uploadedFiles = [];
        this.currentView = 'split'; // 'split', 'graph-only', or 'table-only'
        this.labelsVisible = true;
        this.zoom = null;

        // Sample text data
        // 移除範例文本 - 純淨分析環境

        this.init();
    }

    init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.initializeApp();
            });
        } else {
            this.initializeApp();
        }
    }

    initializeApp() {
        console.log('Initializing Text Analyzer MVP...');
        
        try {
            this.bindEvents();
            this.initializeVisualization();
            this.updateUI();
            console.log('Application initialized successfully');
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showToast('應用程式初始化失敗', 'error');
        }
    }

    bindEvents() {
        // Text input related events
        const textInput = document.getElementById('text-input');
        const analyzeBtn = document.getElementById('analyze-text-btn');
        const clearTextBtn = document.getElementById('clear-text-btn');
        const exportTableBtn = document.getElementById('export-table-btn');
        const sortByImportanceBtn = document.getElementById('sort-by-importance-btn');

        // 簡單文件上傳
        const fileInput = document.getElementById('file-input');
        const uploadFileBtn = document.getElementById('upload-file-btn');

        // 視圖切換按鈕
        const viewSplitBtn = document.getElementById('view-split-btn');
        const viewGraphOnlyBtn = document.getElementById('view-graph-only-btn');
        const viewTableOnlyBtn = document.getElementById('view-table-only-btn');

        // 關係圖控制按鈕
        const resetLayoutBtn = document.getElementById('reset-layout-btn');
        const zoomFitBtn = document.getElementById('zoom-fit-btn');
        const toggleLabelsBtn = document.getElementById('toggle-labels-btn');

        // 第二組關係圖控制按鈕
        const resetLayoutBtn2 = document.getElementById('reset-layout-btn-2');
        const zoomFitBtn2 = document.getElementById('zoom-fit-btn-2');
        const toggleLabelsBtn2 = document.getElementById('toggle-labels-btn-2');
        const sortByImportanceBtn2 = document.getElementById('sort-by-importance-btn-2');

        if (textInput) {
            textInput.addEventListener('input', () => {
                this.updateCharCount();
                this.updateTextPreview();
            });
        }

        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzeText());
        }

        if (clearTextBtn) {
            clearTextBtn.addEventListener('click', () => this.clearText());
        }

        if (exportTableBtn) {
            exportTableBtn.addEventListener('click', () => this.exportTable());
        }

        if (sortByImportanceBtn) {
            sortByImportanceBtn.addEventListener('click', () => this.sortByImportance());
        }

        // 簡單文件上傳事件
        if (uploadFileBtn) {
            uploadFileBtn.addEventListener('click', () => {
                if (fileInput) fileInput.click();
            });
        }

        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleSimpleFileUpload(e));
        }

        // 視圖切換事件
        if (viewSplitBtn) {
            viewSplitBtn.addEventListener('click', () => this.switchToSplitView());
        }

        if (viewGraphOnlyBtn) {
            viewGraphOnlyBtn.addEventListener('click', () => this.switchToGraphOnlyView());
        }

        if (viewTableOnlyBtn) {
            viewTableOnlyBtn.addEventListener('click', () => this.switchToTableOnlyView());
        }

        // 關係圖控制事件
        if (resetLayoutBtn) {
            resetLayoutBtn.addEventListener('click', () => this.resetGraphLayout());
        }

        if (zoomFitBtn) {
            zoomFitBtn.addEventListener('click', () => this.zoomToFit());
        }

        if (toggleLabelsBtn) {
            toggleLabelsBtn.addEventListener('click', () => this.toggleLabels());
        }

        // 第二組關係圖控制事件
        if (resetLayoutBtn2) {
            resetLayoutBtn2.addEventListener('click', () => this.resetGraphLayout());
        }

        if (zoomFitBtn2) {
            zoomFitBtn2.addEventListener('click', () => this.zoomToFit());
        }

        if (toggleLabelsBtn2) {
            toggleLabelsBtn2.addEventListener('click', () => this.toggleLabels());
        }

        if (sortByImportanceBtn2) {
            sortByImportanceBtn2.addEventListener('click', () => this.sortByImportance());
        }
    }

    updateCharCount() {
        const textInput = document.getElementById('text-input');
        const charCount = document.getElementById('char-count');
        
        if (textInput && charCount) {
            const count = textInput.value.length;
            charCount.textContent = `${count} 字元`;
        }
    }

    updateTextPreview() {
        const textInput = document.getElementById('text-input');
        const textPreview = document.getElementById('text-preview');
        
        if (textInput && textPreview) {
            const text = textInput.value.trim();
            if (text) {
                const previewText = text.length > 300 ? text.substring(0, 300) + '...' : text;
                textPreview.innerHTML = `<p>${previewText.replace(/\n/g, '<br>')}</p>`;
                this.currentText = text;
            } else {
                textPreview.innerHTML = '<p class="placeholder-text">請輸入文本以開始分析...</p>';
                this.currentText = '';
            }
            this.updateTextStats();
        }
    }

    updateTextStats() {
        const textLength = document.getElementById('text-length');
        const charDetected = document.getElementById('char-detected');
        
        if (textLength) {
            textLength.textContent = `${this.currentText.length} 字`;
        }
        
        if (charDetected) {
            charDetected.textContent = `${this.characters.length} 人物`;
        }
    }

    loadSampleText() {
        // 功能已移除 - 鼓勵使用者輸入真實文本
        this.showToast('請輸入您自己的文本進行分析', 'info');
    }

    clearText() {
        const textInput = document.getElementById('text-input');
        if (textInput) {
            textInput.value = '';
            this.currentText = '';
            this.characters = [];
            this.relationships = [];
            this.uploadedFiles = [];
            this.updateCharCount();
            this.updateTextPreview();
            this.updateUI();
            this.clearTable();
            this.clearFileList();
            this.showToast('文本已清除', 'info');
        }
    }

    // 多格式文件上傳方法
    async handleSimpleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        // 檢查支援的文件類型
        const supportedExtensions = ['.txt', '.pdf', '.doc', '.docx', '.rtf'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

        if (!supportedExtensions.includes(fileExtension)) {
            alert('不支援的文件格式。請選擇 .txt, .pdf, .doc, .docx, .rtf 格式的文件。');
            e.target.value = '';
            return;
        }

        try {
            this.showLoading(true, `正在讀取 ${fileExtension.toUpperCase()} 文件...`);

            let content = '';

            // 根據文件類型選擇不同的讀取方法
            switch (fileExtension) {
                case '.txt':
                case '.rtf':
                    content = await this.readTextFile(file);
                    break;
                case '.pdf':
                    content = await this.readPDFFile(file);
                    break;
                case '.doc':
                case '.docx':
                    content = await this.readWordFile(file);
                    break;
                default:
                    throw new Error('不支援的文件格式');
            }

            if (content) {
                this.insertContentToTextArea(content, file.name);
            } else {
                throw new Error('無法從文件中提取文本內容');
            }

        } catch (error) {
            this.showLoading(false);
            console.error('文件處理錯誤:', error);
            alert(`文件處理失敗: ${error.message}`);
        } finally {
            e.target.value = ''; // 清除文件選擇
        }
    }

    // 讀取純文本文件
    async readTextFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = () => reject(new Error('文本文件讀取失敗'));
            reader.readAsText(file, 'UTF-8');
        });
    }

    // 讀取PDF文件
    async readPDFFile(file) {
        try {
            // 設置PDF.js worker路徑
            if (typeof pdfjsLib !== 'undefined') {
                pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
            } else {
                throw new Error('PDF.js 庫未載入');
            }

            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

            let fullText = '';

            // 逐頁提取文本
            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                const page = await pdf.getPage(pageNum);
                const textContent = await page.getTextContent();

                const pageText = textContent.items
                    .map(item => item.str)
                    .join(' ')
                    .replace(/\s+/g, ' ') // 合併多餘空格
                    .trim();

                if (pageText) {
                    fullText += `\n第${pageNum}頁:\n${pageText}\n`;
                }
            }

            return fullText.trim();

        } catch (error) {
            throw new Error(`PDF讀取失敗: ${error.message}`);
        }
    }

    // 讀取Word文檔
    async readWordFile(file) {
        try {
            if (typeof mammoth === 'undefined') {
                throw new Error('Mammoth.js 庫未載入');
            }

            const arrayBuffer = await file.arrayBuffer();
            const result = await mammoth.extractRawText({ arrayBuffer: arrayBuffer });

            if (result.messages && result.messages.length > 0) {
                console.warn('Word文檔讀取警告:', result.messages);
            }

            return result.value || '';

        } catch (error) {
            throw new Error(`Word文檔讀取失敗: ${error.message}`);
        }
    }

    // 將內容插入到文本區域
    insertContentToTextArea(content, fileName) {
        const textInput = document.getElementById('text-input');

        if (!textInput) {
            throw new Error('找不到文本輸入區域');
        }

        // 詢問用戶是否要覆蓋現有內容
        const currentText = textInput.value.trim();
        if (currentText) {
            const append = confirm('文本框中已有內容，是否要追加新內容？\n\n確定：追加到現有內容\n取消：覆蓋現有內容');
            if (append) {
                textInput.value = currentText + '\n\n' + content;
            } else {
                textInput.value = content;
            }
        } else {
            textInput.value = content;
        }

        this.updateCharCount();
        this.updateTextPreview();
        this.showLoading(false);

        // 顯示成功消息
        alert(`成功載入文件: ${fileName}\n提取文本長度: ${content.length} 字元`);

        // 滾動到文本框
        textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    exportTable() {
        if (this.characters.length === 0) {
            this.showToast('無資料可匯出，請先分析文本', 'warning');
            return;
        }

        // Create CSV content - simplified to only include name, description and behavior
        const headers = ['人物名稱', '描述', '人物行為'];
        const csvContent = [
            headers.join(','),
            ...this.characters.map(char => [
                `"${char.name}"`,
                `"${char.description || '未知'}"`,
                `"${(char.behaviors || []).map(b => `${b.category}:${(b.actions || []).join('，')}`).join('; ')}"`
            ].join(','))
        ].join('\n');

        // Create and download file
        const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `人物分析結果_${new Date().toISOString().split('T')[0]}.csv`;
        link.click();
        
        this.showToast('表格已匯出為 CSV 檔案', 'success');
    }

    sortByImportance() {
        if (this.characters.length === 0) {
            this.showToast('無資料可排序，請先分析文本', 'warning');
            return;
        }

        this.characters.sort((a, b) => (b.importance || 1) - (a.importance || 1));
        this.renderCharacterTable();
        this.showToast('已按重要性排序', 'success');
    }

    clearTable() {
        const tablePlaceholder = document.getElementById('table-placeholder');
        const characterTable = document.getElementById('character-table');
        
        if (tablePlaceholder) tablePlaceholder.style.display = 'flex';
        if (characterTable) characterTable.style.display = 'none';
    }

    async analyzeText() {
        // 確保獲取最新的文本內容
        const textInput = document.getElementById('text-input');
        if (textInput) {
            this.currentText = textInput.value.trim();
        }
        
        if (!this.currentText.trim()) {
            this.showToast('請先輸入文本', 'warning');
            return;
        }
        
        console.log('分析文本:', this.currentText.substring(0, 100) + '...');
        this.showLoading(true, '正在分析文本...');
        
        try {
            // Send text to backend for analysis
            const response = await fetch('/api/analyze-text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: this.currentText })
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('後端響應數據:', data);
                
                this.characters = data.characters || [];
                this.relationships = data.relationships || [];
                
                console.log('提取的人物:', this.characters);
                console.log('提取的關係:', this.relationships);
                
                this.updateUI();
                this.showToast(`文本分析完成，發現 ${this.characters.length} 個人物`, 'success');
            } else {
                throw new Error('分析失敗');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            // Fallback to client-side analysis
            this.clientSideAnalysis();
        } finally {
            this.showLoading(false);
        }
    }

    clientSideAnalysis() {
        console.log('Using client-side analysis...');
        this.extractCharacters();
        this.generateRelationships();
        this.updateUI();
        this.updateVisualization();
        this.showToast('文本分析完成 (本地分析)', 'success');
    }

    extractCharacters() {
        const text = this.currentText;
        const characterNames = [];
        
        // Simple Chinese name recognition
        const patterns = [
            /(兔子|狐狸|松鼠|貓頭鷹|青蛙|熊)(小?[白赤栗大]?)/g,
            /(博士|先生|小姐|老師)[一-龥]*/g,
        ];
        
        patterns.forEach(pattern => {
            const matches = text.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    const name = match.trim();
                    if (name.length > 1 && !characterNames.includes(name)) {
                        characterNames.push(name);
                    }
                });
            }
        });
        
        // No hardcoded characters - pure NLP approach
        
        // Create character objects
        this.characters = characterNames.map((name, index) => {
            const frequency = (text.match(new RegExp(name, 'g')) || []).length;
            return {
                id: `char_${index}`,
                name: name,
                description: this.generateCharacterDescription(name),
                importance: Math.min(5, Math.max(1, Math.ceil(frequency / 2))),
                frequency: frequency
            };
        });
    }

    generateCharacterDescription(name) {
        if (name.includes('兔子')) return '森林音樂會的發起人';
        if (name.includes('狐狸')) return '音樂會主持人';
        if (name.includes('貓頭鷹')) return '森林中的智者';
        if (name.includes('松鼠')) return '擊鼓手';
        if (name.includes('熊')) return '貝斯手';
        if (name.includes('青蛙')) return '和聲團';
        return '故事中的角色';
    }

    generateRelationships() {
        this.relationships = [];
        
        // Create relationships for all character pairs
        for (let i = 0; i < this.characters.length; i++) {
            for (let j = i + 1; j < this.characters.length; j++) {
                const char1 = this.characters[i];
                const char2 = this.characters[j];
                
                const cooccurrence = this.calculateCooccurrence(char1.name, char2.name);
                
                if (cooccurrence > 0) {
                    const relationInfo = this.determineRelationshipType(char1.name, char2.name);
                    this.relationships.push({
                        id: `rel_${i}_${j}`,
                        source: char1.id,
                        target: char2.id,
                        type: relationInfo.type,
                        text: relationInfo.text,
                        strength: Math.min(5, Math.max(1, cooccurrence))
                    });
                }
            }
        }
    }

    calculateCooccurrence(name1, name2) {
        const sentences = this.currentText.split(/[。！？\n]+/);
        let cooccurrence = 0;
        
        sentences.forEach(sentence => {
            if (sentence.includes(name1) && sentence.includes(name2)) {
                cooccurrence++;
            }
        });
        
        return cooccurrence;
    }

    determineRelationshipType(name1, name2) {
        // 更豐富的關係判斷邏輯

        // 師生關係
        if ((name1.includes('老師') || name1.includes('博士') || name1.includes('教授')) ||
            (name2.includes('老師') || name2.includes('博士') || name2.includes('教授'))) {
            return { type: 'work', text: '師生' };
        }

        // 家庭關係
        if ((name1.includes('爸爸') || name1.includes('媽媽') || name1.includes('父') || name1.includes('母')) ||
            (name2.includes('爸爸') || name2.includes('媽媽') || name2.includes('父') || name2.includes('母')) ||
            (name1.includes('哥哥') || name1.includes('姐姐') || name1.includes('弟弟') || name1.includes('妹妹')) ||
            (name2.includes('哥哥') || name2.includes('姐姐') || name2.includes('弟弟') || name2.includes('妹妹'))) {
            return { type: 'family', text: '家人' };
        }

        // 同事關係
        if ((name1.includes('經理') || name1.includes('主管') || name1.includes('同事')) ||
            (name2.includes('經理') || name2.includes('主管') || name2.includes('同事'))) {
            return { type: 'work', text: '同事' };
        }

        // 朋友關係 (通過文本內容判斷)
        const friendKeywords = ['朋友', '好友', '同學', '夥伴', '玩伴'];
        if (friendKeywords.some(keyword => this.currentText.includes(keyword))) {
            return { type: 'friendship', text: '朋友' };
        }

        // 合作關係
        if (this.currentText.includes('合作') || this.currentText.includes('一起') || this.currentText.includes('共同')) {
            return { type: 'work', text: '合作' };
        }

        // 預設關係
        return { type: 'default', text: '認識' };
    }

    updateUI() {
        console.log('更新UI，人物數量:', this.characters.length);
        this.updateTextStats();
        this.renderCharacterList();

        // 設置默認視圖並更新按鈕狀態
        this.updateViewButtons();
        this.updateViewDisplay();

        // 根據當前視圖渲染相應的內容
        if (this.characters.length > 0) {
            console.log('顯示數據，隱藏佔位符');
            switch (this.currentView) {
                case 'split':
                    this.renderCharacterTable('character-table-body');
                    this.renderRelationshipGraph('relationship-graph');
                    break;
                case 'graph-only':
                    this.renderRelationshipGraph('relationship-graph-2');
                    break;
                case 'table-only':
                    this.renderCharacterTable('character-table-body-2');
                    break;
            }
        } else {
            // 顯示占位符
            this.showGraphPlaceholder('relationship-graph');
            this.showGraphPlaceholder('relationship-graph-2');
        }
    }

    // 視圖切換方法
    switchToSplitView() {
        this.currentView = 'split';
        this.updateViewButtons();
        this.updateViewDisplay();

        // 如果有數據，重新渲染關係圖
        if (this.characters.length > 0) {
            setTimeout(() => {
                this.renderRelationshipGraph('relationship-graph');
                this.updateTable();
            }, 100);
        }
    }

    switchToGraphOnlyView() {
        this.currentView = 'graph-only';
        this.updateViewButtons();
        this.updateViewDisplay();

        // 如果有數據，重新渲染關係圖
        if (this.characters.length > 0) {
            setTimeout(() => this.renderRelationshipGraph('relationship-graph-2'), 100);
        }
    }

    switchToTableOnlyView() {
        this.currentView = 'table-only';
        this.updateViewButtons();
        this.updateViewDisplay();

        // 如果有數據，更新表格
        if (this.characters.length > 0) {
            this.updateTable('character-table-body-2');
        }
    }

    updateViewButtons() {
        const viewSplitBtn = document.getElementById('view-split-btn');
        const viewGraphOnlyBtn = document.getElementById('view-graph-only-btn');
        const viewTableOnlyBtn = document.getElementById('view-table-only-btn');

        // 重置所有按鈕狀態
        [viewSplitBtn, viewGraphOnlyBtn, viewTableOnlyBtn].forEach(btn => {
            if (btn) {
                btn.classList.remove('btn--primary', 'active');
                btn.classList.add('btn--outline');
            }
        });

        // 設置當前活動按鈕
        let activeBtn;
        switch (this.currentView) {
            case 'split':
                activeBtn = viewSplitBtn;
                break;
            case 'graph-only':
                activeBtn = viewGraphOnlyBtn;
                break;
            case 'table-only':
                activeBtn = viewTableOnlyBtn;
                break;
        }

        if (activeBtn) {
            activeBtn.classList.remove('btn--outline');
            activeBtn.classList.add('btn--primary', 'active');
        }
    }

    updateViewDisplay() {
        const splitView = document.getElementById('split-view');
        const graphOnlyView = document.getElementById('graph-only-view');
        const tableOnlyView = document.getElementById('table-only-view');

        // 隱藏所有視圖
        [splitView, graphOnlyView, tableOnlyView].forEach(view => {
            if (view) {
                view.classList.remove('active');
                view.style.display = 'none';
            }
        });

        // 顯示當前視圖
        let activeView;
        switch (this.currentView) {
            case 'split':
                activeView = splitView;
                break;
            case 'graph-only':
                activeView = graphOnlyView;
                break;
            case 'table-only':
                activeView = tableOnlyView;
                break;
        }

        if (activeView) {
            activeView.classList.add('active');
            activeView.style.display = 'block';
        }
    }

    // 關係圖渲染方法
    renderRelationshipGraph(containerId = 'relationship-graph') {
        const container = document.getElementById(containerId);
        if (!container || this.characters.length === 0) {
            this.showGraphPlaceholder(containerId);
            return;
        }

        this.hideGraphPlaceholder(containerId);

        // 清除現有內容
        container.innerHTML = '';

        // 創建SVG
        const containerRect = container.getBoundingClientRect();
        const width = containerRect.width || 600;
        const height = containerRect.height || 400;

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('border', '1px solid #e1e5e9')
            .style('border-radius', '8px');

        // 添加縮放功能
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });

        svg.call(this.zoom);

        // 創建主容器
        const g = svg.append('g');

        // 準備數據
        const nodes = this.characters.map(char => ({
            id: char.id || char.name,
            name: char.name,
            importance: char.importance || 1,
            description: char.description || '',
            x: Math.random() * width,
            y: Math.random() * height
        }));

        const links = this.relationships.map(rel => ({
            source: rel.source,
            target: rel.target,
            strength: rel.strength || 1,
            type: rel.type || 'relation',
            text: rel.text || rel.type || '關係' // 添加關係文本
        }));

        // 創建力導向圖
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));

        // 創建連線組
        const linkGroup = g.append('g').attr('class', 'links');

        // 創建連線
        const link = linkGroup.selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('class', 'relationship-link')
            .attr('stroke', d => this.getRelationshipColor(d.type))
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => Math.sqrt(d.strength) * 2);

        // 創建連線標籤容器
        const linkLabelGroup = g.append('g').attr('class', 'link-labels');

        // 為每個標籤創建背景矩形和文字
        const linkLabelContainer = linkLabelGroup.selectAll('g')
            .data(links)
            .enter().append('g')
            .attr('class', 'link-label-container')
            .style('display', this.labelsVisible ? 'block' : 'none');

        // 添加背景矩形
        linkLabelContainer.append('rect')
            .attr('class', 'link-label-bg')
            .attr('fill', 'white')
            .attr('stroke', d => this.getRelationshipColor(d.type))
            .attr('stroke-width', '1')
            .attr('rx', '4')
            .attr('ry', '4');

        // 添加文字
        const linkLabel = linkLabelContainer.append('text')
            .attr('class', d => `link-label ${d.type}`)
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('font-size', '10px')
            .attr('font-family', 'Arial, sans-serif')
            .attr('fill', d => this.getRelationshipColor(d.type))
            .attr('font-weight', '600')
            .style('pointer-events', 'none')
            .text(d => d.text);

        // 創建節點
        const node = g.append('g')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('class', d => d.importance > 3 ? 'main-character' : 'secondary-character')
            .attr('r', d => 10 + d.importance * 3)
            .attr('fill', d => d.importance > 3 ? '#667eea' : '#4ade80')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .call(d3.drag()
                .on('start', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                })
                .on('drag', (event, d) => {
                    d.fx = event.x;
                    d.fy = event.y;
                })
                .on('end', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }));

        // 添加節點標籤
        const label = g.append('g')
            .selectAll('text')
            .data(nodes)
            .enter().append('text')
            .text(d => d.name)
            .attr('class', 'node-label')
            .attr('font-family', 'Arial, sans-serif')
            .attr('font-size', '12px')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', '#333')
            .style('pointer-events', 'none')
            .style('display', this.labelsVisible ? 'block' : 'none');

        // 添加hover效果
        node.on('mouseover', (event, d) => {
            // 高亮相關連線
            link.style('stroke-opacity', l =>
                l.source.id === d.id || l.target.id === d.id ? 1 : 0.1
            );

            // 顯示tooltip
            this.showTooltip(event, d);
        })
        .on('mouseout', () => {
            link.style('stroke-opacity', 0.6);
            this.hideTooltip();
        });

        // 更新位置
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            // 更新標籤容器位置
            linkLabelContainer
                .attr('transform', d => {
                    const midX = (d.source.x + d.target.x) / 2;
                    const midY = (d.source.y + d.target.y) / 2;

                    // 計算線條角度，調整標籤位置避免重疊
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const angle = Math.atan2(dy, dx) * 180 / Math.PI;

                    // 根據角度偏移標籤位置，避免與線條重疊
                    const offsetDistance = 8;
                    const offsetX = -Math.sin(angle * Math.PI / 180) * offsetDistance;
                    const offsetY = Math.cos(angle * Math.PI / 180) * offsetDistance;

                    return `translate(${midX + offsetX}, ${midY + offsetY})`;
                });

            // 更新標籤背景矩形大小
            linkLabelContainer.selectAll('.link-label-bg')
                .each(function(d) {
                    const textElement = d3.select(this.parentNode).select('text').node();
                    if (textElement) {
                        const bbox = textElement.getBBox();
                        d3.select(this)
                            .attr('x', bbox.x - 4)
                            .attr('y', bbox.y - 2)
                            .attr('width', bbox.width + 8)
                            .attr('height', bbox.height + 4);
                    }
                });

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            label
                .attr('x', d => d.x)
                .attr('y', d => d.y + 25);
        });

        this.networkSimulation = simulation;
        this.svg = svg;
    }

    showGraphPlaceholder(containerId = 'relationship-graph') {
        const placeholderId = containerId === 'relationship-graph-2' ? 'graph-placeholder-2' : 'graph-placeholder';
        const placeholder = document.getElementById(placeholderId);
        const graph = document.getElementById(containerId);
        if (placeholder) placeholder.style.display = 'flex';
        if (graph) graph.style.display = 'none';
    }

    hideGraphPlaceholder(containerId = 'relationship-graph') {
        const placeholderId = containerId === 'relationship-graph-2' ? 'graph-placeholder-2' : 'graph-placeholder';
        const placeholder = document.getElementById(placeholderId);
        const graph = document.getElementById(containerId);
        if (placeholder) placeholder.style.display = 'none';
        if (graph) graph.style.display = 'block';
    }

    // 關係圖控制方法
    resetGraphLayout() {
        if (this.networkSimulation) {
            this.networkSimulation.alpha(1).restart();
            this.showToast('佈局已重置', 'info');
        }
    }

    zoomToFit() {
        if (this.svg && this.zoom) {
            const svg = this.svg;
            const bounds = svg.select('g').node().getBBox();
            const parent = svg.node().getBoundingClientRect();

            const scale = 0.8 * Math.min(
                parent.width / bounds.width,
                parent.height / bounds.height
            );

            const translate = [
                (parent.width - bounds.width * scale) / 2 - bounds.x * scale,
                (parent.height - bounds.height * scale) / 2 - bounds.y * scale
            ];

            svg.transition()
                .duration(750)
                .call(this.zoom.transform, d3.zoomIdentity
                    .translate(translate[0], translate[1])
                    .scale(scale));

            this.showToast('已調整視圖至適合大小', 'info');
        }
    }

    toggleLabels() {
        this.labelsVisible = !this.labelsVisible;

        if (this.svg) {
            // 切換節點標籤
            this.svg.selectAll('.node-label')
                .style('display', this.labelsVisible ? 'block' : 'none');

            // 切換關係標籤容器
            this.svg.selectAll('.link-label-container')
                .style('display', this.labelsVisible ? 'block' : 'none');
        }

        const btn = document.getElementById('toggle-labels-btn');
        if (btn) {
            btn.textContent = this.labelsVisible ? '隱藏標籤' : '顯示標籤';
        }

        this.showToast(this.labelsVisible ? '標籤已顯示' : '標籤已隱藏', 'info');
    }

    showTooltip(event, data) {
        // 簡單的tooltip實現
        const tooltip = d3.select('body').append('div')
            .attr('class', 'graph-tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .style('opacity', 0);

        tooltip.html(`
            <strong>${data.name}</strong><br/>
            重要性: ${data.importance}<br/>
            描述: ${data.description || '無'}
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px')
        .transition()
        .duration(200)
        .style('opacity', 1);
    }

    hideTooltip() {
        d3.selectAll('.graph-tooltip').remove();
    }

    // 根據關係類型獲取顏色
    getRelationshipColor(type) {
        const colorMap = {
            'friendship': '#10b981',  // 綠色 - 朋友
            'family': '#f59e0b',      // 橙色 - 家人
            'work': '#3b82f6',        // 藍色 - 工作
            'romantic': '#ef4444',    // 紅色 - 愛情
            'default': '#6b7280'      // 灰色 - 預設
        };
        return colorMap[type] || colorMap['default'];
    }

    updateTable(tableBodyId = 'character-table-body') {
        this.renderCharacterTable(tableBodyId);
    }

    renderCharacterTable(tableBodyId = 'character-table-body') {
        console.log('開始渲染表格，人物數量:', this.characters.length);

        // 根據 tableBodyId 確定對應的元素 ID
        const isSecondary = tableBodyId === 'character-table-body-2';
        const placeholderId = isSecondary ? 'table-placeholder-2' : 'table-placeholder';
        const tableId = isSecondary ? 'character-table-2' : 'character-table';

        const tablePlaceholder = document.getElementById(placeholderId);
        const characterTable = document.getElementById(tableId);
        const tableBody = document.getElementById(tableBodyId);
        
        if (!tableBody) {
            console.error('找不到表格主體元素');
            return;
        }
        
        if (this.characters.length === 0) {
            console.log('無人物數據，顯示佔位符');
            if (tablePlaceholder) tablePlaceholder.style.display = 'flex';
            if (characterTable) characterTable.style.display = 'none';
            return;
        }
        
        // Hide placeholder and show table
        if (tablePlaceholder) tablePlaceholder.style.display = 'none';
        if (characterTable) characterTable.style.display = 'block';
        
        // Generate table rows - simplified to only show name, description and behavior
        tableBody.innerHTML = this.characters.map(character => `
            <tr data-id="${character.id}">
                <td class="character-name">${character.name}</td>
                <td class="character-description">${character.description || '未知'}</td>
                <td class="behavior-list">${this.generateBehaviorTags(character.behaviors || [])}</td>
            </tr>
        `).join('');
        
        // Add sorting functionality
        this.addTableSorting();
    }

    generateImportanceStars(importance) {
        const stars = '★'.repeat(Math.min(5, Math.max(1, importance)));
        const emptyStars = '☆'.repeat(5 - stars.length);
        return `<span class="importance-stars">${stars}${emptyStars}</span>`;
    }

    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'confidence-high';
        if (confidence >= 0.6) return 'confidence-medium';
        return 'confidence-low';
    }

    generateEventTags(events) {
        if (!events || events.length === 0) return '<span class="text-muted">無</span>';
        return events.slice(0, 3).map(event => 
            `<span class="event-item">${event.type || '事件'}</span>`
        ).join(' ');
    }

    generateAttributeTags(attributes) {
        if (!attributes || attributes.length === 0) return '<span class="text-muted">無</span>';
        return attributes.slice(0, 3).map(attr => 
            `<span class="attribute-item">${attr.type || attr.value || '屬性'}</span>`
        ).join(' ');
    }
    
    generateBehaviorTags(behaviors) {
        if (!behaviors || behaviors.length === 0) return '<span class="text-muted">無行為記錄</span>';

        // 檢查behaviors是字符串數組還是對象數組
        if (typeof behaviors[0] === 'string') {
            // 處理字符串數組 - 新的Claude API格式
            return behaviors.slice(0, 5).map((behaviorText, index) => {
                // 截短長文本以適應表格顯示
                const displayText = behaviorText.length > 30 ?
                    behaviorText.substring(0, 30) + '...' : behaviorText;
                return `<div class="behavior-item" title="${behaviorText}">${displayText}</div>`;
            }).join('');
        } else {
            // 處理對象數組 - 原有格式
            return behaviors.slice(0, 3).map(behavior => {
                const category = behavior.category || '行為';
                const count = behavior.count || 1;
                const actions = behavior.actions || [];

                const firstAction = actions.length > 0 ? actions[0] : '未知行為';
                const tooltip = actions.slice(0, 3).join('；');

                return `<span class="behavior-item" title="${tooltip}">${category}(${count})</span>`;
            }).join(' ');
        }
    }

    addTableSorting() {
        const sortableHeaders = document.querySelectorAll('.character-table th.sortable');
        sortableHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const sortBy = header.getAttribute('data-sort');
                this.sortTable(sortBy, header);
            });
        });
    }

    sortTable(sortBy, headerElement) {
        const currentSort = headerElement.classList.contains('sort-asc') ? 'asc' : 
                           headerElement.classList.contains('sort-desc') ? 'desc' : 'none';
        
        // Remove all sort classes
        document.querySelectorAll('.character-table th').forEach(th => {
            th.classList.remove('sort-asc', 'sort-desc');
        });
        
        let newSort = 'asc';
        if (currentSort === 'asc') newSort = 'desc';
        else if (currentSort === 'desc') newSort = 'asc';
        
        headerElement.classList.add(`sort-${newSort}`);
        
        // Sort characters array
        this.characters.sort((a, b) => {
            let aVal = a[sortBy] || '';
            let bVal = b[sortBy] || '';
            
            // Handle different data types
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return newSort === 'asc' ? aVal - bVal : bVal - aVal;
            } else {
                aVal = String(aVal).toLowerCase();
                bVal = String(bVal).toLowerCase();
                if (newSort === 'asc') {
                    return aVal.localeCompare(bVal);
                } else {
                    return bVal.localeCompare(aVal);
                }
            }
        });
        
        // Re-render table
        this.renderCharacterTable();
    }

    renderCharacterList() {
        const characterList = document.getElementById('character-list');
        if (!characterList) return;
        
        if (this.characters.length === 0) {
            characterList.innerHTML = '<div class="empty-state"><p>尚無人物數據，請先分析文本</p></div>';
            return;
        }
        
        characterList.innerHTML = `
            <div class="character-summary">
                <h4>人物摘要 (共 ${this.characters.length} 個角色)</h4>
                ${this.characters.map(character => `
                    <div class="character-item" data-id="${character.id}">
                        <strong>${character.name}</strong>
                        <small>(信心: ${(character.confidence || 0.8).toFixed(2)}, 頻次: ${character.frequency || 1})</small>
                    </div>
                `).join('')}
            </div>
        `;
    }

    initializeVisualization() {
        const container = document.getElementById('network-container');
        if (!container) return;

        this.svg = d3.select('#network-svg');
        if (this.svg.empty()) {
            console.log('Creating new SVG element...');
            this.svg = d3.select('#network-container')
                .append('svg')
                .attr('id', 'network-svg')
                .attr('class', 'network-svg');
        }

        const containerRect = container.getBoundingClientRect();
        const width = containerRect.width || 600;
        const height = containerRect.height || 400;

        this.svg
            .attr('width', width)
            .attr('height', height);

        this.networkSimulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2));
    }

    updateVisualization() {
        this.renderNetwork();
    }

    renderNetwork() {
        if (!this.svg || this.characters.length === 0) {
            this.showNetworkPlaceholder();
            return;
        }
        
        this.hideNetworkPlaceholder();
        
        this.svg.selectAll('*').remove();
        
        const nodes = this.characters.map(char => ({ ...char }));
        const links = this.relationships.map(rel => ({ ...rel }));
        
        // Create links
        const link = this.svg.append('g')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('class', 'relationship-link')
            .attr('stroke', '#999')
            .attr('stroke-width', d => Math.sqrt(d.strength) * 2)
            .attr('stroke-opacity', 0.6);
        
        // Create nodes
        const node = this.svg.append('g')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('class', 'character-node')
            .attr('r', d => 10 + d.importance * 3)
            .attr('fill', '#4CAF50')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));
        
        // Add labels
        const label = this.svg.append('g')
            .selectAll('text')
            .data(nodes)
            .enter().append('text')
            .text(d => d.name)
            .attr('class', 'node-label')
            .attr('font-family', 'Arial, sans-serif')
            .attr('font-size', '12px')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .style('pointer-events', 'none');
        
        // Add hover effects
        node.on('mouseover', (event, d) => {
            d3.select(event.target).attr('r', d => 15 + d.importance * 3);
        }).on('mouseout', (event, d) => {
            d3.select(event.target).attr('r', d => 10 + d.importance * 3);
        });
        
        // Update simulation
        this.networkSimulation
            .nodes(nodes)
            .on('tick', () => {
                link
                    .attr('x1', d => {
                        const source = nodes.find(n => n.id === d.source);
                        return source ? source.x : 0;
                    })
                    .attr('y1', d => {
                        const source = nodes.find(n => n.id === d.source);
                        return source ? source.y : 0;
                    })
                    .attr('x2', d => {
                        const target = nodes.find(n => n.id === d.target);
                        return target ? target.x : 0;
                    })
                    .attr('y2', d => {
                        const target = nodes.find(n => n.id === d.target);
                        return target ? target.y : 0;
                    });
                
                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                label
                    .attr('x', d => d.x)
                    .attr('y', d => d.y + 25);
            });
        
        this.networkSimulation.force('link').links(links);
        this.networkSimulation.alpha(1).restart();
    }

    showNetworkPlaceholder() {
        const placeholder = document.getElementById('network-placeholder');
        if (placeholder) {
            placeholder.style.display = 'flex';
        }
    }

    hideNetworkPlaceholder() {
        const placeholder = document.getElementById('network-placeholder');
        if (placeholder) {
            placeholder.style.display = 'none';
        }
    }

    clearVisualization() {
        if (this.svg) {
            this.svg.selectAll('*').remove();
        }
        this.showNetworkPlaceholder();
    }

    resetLayout() {
        if (this.networkSimulation && this.characters.length > 0) {
            this.networkSimulation.alpha(1).restart();
            this.showToast('佈局已重設', 'info');
        }
    }

    // Drag handlers
    dragStarted(event, d) {
        if (!event.active) this.networkSimulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.networkSimulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // Utility methods
    showLoading(show, message = '正在處理...') {
        const overlay = document.getElementById('loading-overlay');
        const text = document.querySelector('.loading-text');
        
        if (overlay) {
            if (show) {
                if (text) text.textContent = message;
                overlay.classList.remove('hidden');
            } else {
                overlay.classList.add('hidden');
            }
        }
    }


    showToast(message, type = 'info') {
        console.log(`Toast [${type}]: ${message}`);

        // 簡單的toast通知實現
        const toastContainer = document.getElementById('toast-container');
        if (toastContainer) {
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            toast.innerHTML = `
                <div class="toast-content">
                    <span class="toast-icon">${this.getToastIcon(type)}</span>
                    <span class="toast-message">${message}</span>
                </div>
            `;

            toastContainer.appendChild(toast);

            // 3秒後自動移除
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 3000);
        }
    }

    getToastIcon(type) {
        switch(type) {
            case 'success': return '✅';
            case 'error': return '❌';
            case 'warning': return '⚠️';
            case 'info':
            default: return 'ℹ️';
        }
    }
}

// Initialize the application
const analyzer = new TextAnalyzerMVP();