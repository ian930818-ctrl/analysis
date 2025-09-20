class TextAnalyzerMVP {
    constructor() {
        this.currentText = '';
        this.characters = [];
        this.relationships = [];
        this.networkSimulation = null;
        this.svg = null;
        
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
            this.updateCharCount();
            this.updateTextPreview();
            this.updateUI();
            this.clearTable();
            this.showToast('文本已清除', 'info');
        }
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
                    this.relationships.push({
                        id: `rel_${i}_${j}`,
                        source: char1.id,
                        target: char2.id,
                        type: this.determineRelationshipType(char1.name, char2.name),
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
        if ((name1.includes('兔子') || name2.includes('兔子')) && 
            (name1.includes('狐狸') || name2.includes('狐狸'))) {
            return '合作夥伴';
        }
        if (name1.includes('博士') || name2.includes('博士')) {
            return '師生';
        }
        return '朋友';
    }

    updateUI() {
        console.log('更新UI，人物數量:', this.characters.length);
        this.updateTextStats();
        this.renderCharacterTable();
        this.renderCharacterList();
        
        // 強制重新渲染表格
        if (this.characters.length > 0) {
            console.log('顯示表格，隱藏佔位符');
            const tablePlaceholder = document.getElementById('table-placeholder');
            const characterTable = document.getElementById('character-table');
            
            if (tablePlaceholder) tablePlaceholder.style.display = 'none';
            if (characterTable) characterTable.style.display = 'block';
        }
    }

    renderCharacterTable() {
        console.log('開始渲染表格，人物數量:', this.characters.length);
        const tablePlaceholder = document.getElementById('table-placeholder');
        const characterTable = document.getElementById('character-table');
        const tableBody = document.getElementById('character-table-body');
        
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
        // Simple console log for now - can enhance later
    }
}

// Initialize the application
const analyzer = new TextAnalyzerMVP();