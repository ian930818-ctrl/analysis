class TextAnalyzerMVP {
    constructor() {
        this.currentText = '';
        this.characters = [];
        this.relationships = [];
        this.networkSimulation = null;
        this.svg = null;
        
        // Sample text data
        this.sampleText = `一天傍晚，森林裡的兔子小白跑到河邊，興奮地喊：「大家快來！今晚我們舉辦一場音樂會！」

狐狸小赤甩著尾巴說：「音樂會？我不會樂器耶。」
「沒關係！」貓頭鷹博士拍拍翅膀說，「你可以負責主持。」

松鼠小栗搬來一大堆橡果當鼓，熊大用低沉的聲音當貝斯，甚至連小青蛙們都在河邊「呱呱」和聲。

當月亮升起，音樂響徹整個森林。兔子小白唱歌、狐狸小赤講笑話，松鼠和熊大打著節奏，大家一起跳舞。

最後，貓頭鷹博士說：「這場音樂會告訴我們，只要團結，就能創造美好的事情。」

森林裡傳來笑聲與掌聲，像星光一樣閃爍。`;
        
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
        const loadSampleBtn = document.getElementById('load-sample-btn');
        const clearTextBtn = document.getElementById('clear-text-btn');
        const resetLayoutBtn = document.getElementById('reset-layout-btn');
        
        if (textInput) {
            textInput.addEventListener('input', () => {
                this.updateCharCount();
                this.updateTextPreview();
            });
        }
        
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzeText());
        }
        
        if (loadSampleBtn) {
            loadSampleBtn.addEventListener('click', () => this.loadSampleText());
        }
        
        if (clearTextBtn) {
            clearTextBtn.addEventListener('click', () => this.clearText());
        }
        
        if (resetLayoutBtn) {
            resetLayoutBtn.addEventListener('click', () => this.resetLayout());
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
        const textInput = document.getElementById('text-input');
        if (textInput) {
            textInput.value = this.sampleText;
            this.updateCharCount();
            this.updateTextPreview();
            this.showToast('範例文本已載入', 'success');
        }
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
            this.clearVisualization();
            this.showToast('文本已清除', 'info');
        }
    }

    async analyzeText() {
        if (!this.currentText.trim()) {
            this.showToast('請先輸入文本', 'warning');
            return;
        }
        
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
                this.characters = data.characters || [];
                this.relationships = data.relationships || [];
                
                this.updateUI();
                this.updateVisualization();
                this.showToast('文本分析完成', 'success');
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
        
        // Manual addition for known characters
        const knownCharacters = ['兔子小白', '狐狸小赤', '貓頭鷹博士', '松鼠小栗', '熊大', '小青蛙們'];
        knownCharacters.forEach(name => {
            if (text.includes(name) && !characterNames.includes(name)) {
                characterNames.push(name);
            }
        });
        
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
        this.updateTextStats();
        this.renderCharacterList();
    }

    renderCharacterList() {
        const characterList = document.getElementById('character-list');
        if (!characterList) return;
        
        if (this.characters.length === 0) {
            characterList.innerHTML = '<div class="empty-state"><p>尚無人物數據，請先分析文本</p></div>';
            return;
        }
        
        characterList.innerHTML = this.characters.map(character => `
            <div class="character-item" data-id="${character.id}">
                <strong>${character.name}</strong>
                <small>(${character.description}, 重要性: ${character.importance}/5)</small>
            </div>
        `).join('');
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