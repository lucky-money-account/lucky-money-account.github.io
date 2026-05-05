const state = {
    selectedMode: 'general',
    selectedFile: null,
    serverOnline: false
};

const modeIcons = {
    general: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/></svg>',
    digit: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2"/><path d="M8 7h8l-4 10H8"/></svg>',
    animal: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2c-1.5 0-2.5 1-3 2C8 3 6.5 3.5 6 5c-.5 1.5 1 3 2 3.5C9 9 10 9 12 9s3 0 4-.5c1-.5 2.5-2 2-3.5C17.5 3.5 16 3 15 4c-.5-1-1.5-2-3-2z"/><circle cx="8.5" cy="6.5" r="1"/><circle cx="15.5" cy="6.5" r="1"/><path d="M6 8c-1.5 1-2.5 2.5-2 4.5.5 2 2 3 3.5 2.5C9 14.5 10 13 12 13s3 1.5 4.5 2c1.5.5 3-.5 3.5-2.5.5-2-.5-3.5-2-4.5"/></svg>',
    scene: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 20h18L12 4 3 20z"/><path d="M15 20l-3-6-3 6"/></svg>'
};

const iconClasses = {
    general: 'purple',
    digit: 'cyan',
    animal: 'orange',
    scene: 'green'
};

async function init() {
    await waitForServer();
    await loadModes();
    setupUpload();
    setupButtons();
    pollModelsStatus();
}

async function fetchWithRetry(url, options = {}, maxRetries = 5) {
    let lastError;
    for (let i = 0; i < maxRetries; i++) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 8000);
            const opts = { ...options, signal: controller.signal };
            const res = await fetch(url, opts);
            clearTimeout(timeoutId);
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data.error || '服务器错误');
            }
            return res;
        } catch (e) {
            lastError = e;
            if (e.name === 'AbortError') continue;
            const delay = Math.min(1000 * Math.pow(1.5, i), 8000);
            await sleep(delay);
        }
    }
    throw lastError;
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

async function waitForServer() {
    const banner = showStatusBanner('正在连接服务器...', 'loading');
    for (let i = 0; i < 60; i++) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 2000);
            const res = await fetch('/api/status', { signal: controller.signal });
            clearTimeout(timeoutId);
            if (res.ok) {
                state.serverOnline = true;
                banner.remove();
                return;
            }
        } catch (e) {}
        await sleep(1500);
    }
    banner.innerHTML = '<span class="status-dot error"></span>无法连接服务器，请确认已运行 python run.py';
    banner.className = 'status-banner error';
}

async function pollModelsStatus() {
    for (let attempt = 0; attempt < 30; attempt++) {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.loaded || (data.pytorch && data.digit)) break;
        } catch (e) {}
        await sleep(2000);
    }
    try { await loadModes(); } catch (e) {}
}

function showStatusBanner(msg, type) {
    let banner = document.getElementById('statusBanner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'statusBanner';
        banner.className = 'status-banner ' + type;
        document.body.prepend(banner);
    }
    banner.innerHTML = '<span class="status-dot ' + type + '"></span>' + msg;
    return banner;
}

async function loadModes() {
    try {
        const res = await fetchWithRetry('/api/modes');
        const data = await res.json();
        renderModeCards(data.modes);
    } catch (e) {
        showToast('加载识别模式失败，请刷新页面');
    }
}

function renderModeCards(modes) {
    const grid = document.getElementById('modeGrid');
    grid.innerHTML = '';

    modes.forEach(mode => {
        const card = document.createElement('div');
        card.className = 'mode-card' + (mode.id === state.selectedMode ? ' active' : '') + (!mode.available ? ' disabled' : '');
        card.dataset.mode = mode.id;

        card.innerHTML =
            '<div class="mode-icon ' + iconClasses[mode.id] + '">' + modeIcons[mode.id] + '</div>' +
            '<h3>' + mode.name + '</h3>' +
            '<p>' + mode.description + '</p>' +
            (mode.id === state.selectedMode ? '<div class="mode-check"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg></div>' : '') +
            (!mode.available ? '<span class="disabled-badge">需训练</span>' : '');

        if (mode.available) {
            card.addEventListener('click', () => selectMode(mode.id));
        }
        grid.appendChild(card);
    });
}

function selectMode(modeId) {
    state.selectedMode = modeId;
    document.querySelectorAll('.mode-card').forEach(card => {
        card.classList.toggle('active', card.dataset.mode === modeId);
    });
    document.querySelectorAll('.mode-check').forEach(el => el.remove());
    const activeCard = document.querySelector('.mode-card[data-mode="' + modeId + '"]');
    if (activeCard) {
        const check = document.createElement('div');
        check.className = 'mode-check';
        check.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>';
        activeCard.appendChild(check);
    }
}

function setupUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    document.getElementById('removeImage').addEventListener('click', resetUpload);
    document.getElementById('resetBtn').addEventListener('click', resetAll);
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('请选择图片文件');
        return;
    }
    state.selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('uploadArea').classList.add('hidden');
        document.getElementById('previewArea').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    state.selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('previewImage').src = '';
    document.getElementById('uploadArea').classList.remove('hidden');
    document.getElementById('previewArea').classList.add('hidden');
}

function resetAll() {
    resetUpload();
    document.getElementById('resultSection').classList.add('hidden');
    document.getElementById('topResult').innerHTML = '';
    document.getElementById('resultList').innerHTML = '';
    document.getElementById('resultSpinner').style.display = '';
}

function setupButtons() {
    document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);
}

async function runAnalysis() {
    if (!state.selectedFile) {
        showToast('请先上传图片');
        return;
    }

    const resultSection = document.getElementById('resultSection');
    const resultSpinner = document.getElementById('resultSpinner');

    resultSection.classList.remove('hidden');
    resultSpinner.style.display = '';
    document.getElementById('topResult').innerHTML = '';
    document.getElementById('resultList').innerHTML = '';

    const formData = new FormData();
    formData.append('image', state.selectedFile);
    formData.append('mode', state.selectedMode);

    try {
        const res = await fetchWithRetry('/api/predict', { method: 'POST', body: formData }, 3);
        const data = await res.json();
        resultSpinner.style.display = 'none';

        if (data.error) {
            showToast(data.error);
            return;
        }
        renderResults(data);
    } catch (e) {
        resultSpinner.style.display = 'none';
        showToast('识别请求失败: ' + (e.message || '网络异常，请重试'));
    }
}

function renderResults(data) {
    const topResult = document.getElementById('topResult');
    const resultList = document.getElementById('resultList');

    if (!topResult || !resultList) {
        showToast('界面元素丢失，请刷新页面');
        return;
    }

    if (data.annotated_image) {
        topResult.innerHTML =
            '<div class="annotated-wrapper">' +
            '<img class="annotated-img" src="data:image/jpeg;base64,' + data.annotated_image + '" alt="检测结果">' +
            '</div>';

        if (data.multi_object) {
            topResult.innerHTML +=
                '<div class="multi-badge">检测到 ' + data.objects.length + ' 个目标</div>';
        } else {
            topResult.innerHTML +=
                '<div class="confidence" style="margin-top:8px;">MobileNetV2 细分: <strong>' + escapeHtml(data.top_label) + '</strong> (' + (data.top_confidence * 100).toFixed(1) + '%)</div>';
        }
    } else if (data.digit_string && data.digit_details) {
        const rows = {};
        data.digit_details.forEach(d => {
            const row = d.row || 1;
            if (!rows[row]) rows[row] = [];
            rows[row].push(d);
        });
        const rowCount = Object.keys(rows).length;

        const displayStr = data.digit_string.replace(/\n/g, '<br>');
        topResult.innerHTML =
            '<div class="digit-string">' + displayStr + '</div>' +
            '<div class="digit-string-label">' + (rowCount > 1 ? '识别到的数字串 (' + rowCount + ' 行)' : '识别到的数字串') + '</div>';

        resultList.innerHTML = '';
        Object.keys(rows).sort().forEach(rowNum => {
            if (rowCount > 1) {
                resultList.innerHTML += '<div class="row-separator">第 ' + rowNum + ' 行</div>';
            }
            rows[rowNum].forEach((d, i) => {
                const pct = (d.confidence * 100).toFixed(1);
                resultList.innerHTML += '<div class="result-item">' +
                    '<div class="rank digit-rank">' + (i + 1) + '</div>' +
                    '<div class="item-label digit-item-label">数字 <strong>' + d.digit + '</strong></div>' +
                    '<div class="item-bar-wrapper"><div class="item-bar" style="width: ' + Math.max(d.confidence * 100, 2) + '%"></div></div>' +
                    '<div class="item-confidence">' + pct + '%</div>' +
                    '</div>';
            });
        });
        return;
    } else {
        topResult.innerHTML =
            '<div class="label">' + escapeHtml(data.top_label) + '</div>' +
            '<div class="confidence">置信度 <strong>' + (data.top_confidence * 100).toFixed(1) + '%</strong></div>';
    }

    resultList.innerHTML = data.results.map((r, i) => {
        const pct = (r.confidence * 100).toFixed(1);
        return '<div class="result-item">' +
            '<div class="rank">' + (i + 1) + '</div>' +
            '<div class="item-label" title="' + escapeHtml(r.label) + '">' + escapeHtml(r.label) + '</div>' +
            '<div class="item-bar-wrapper"><div class="item-bar" style="width: ' + Math.max(r.confidence * 100, 2) + '%"></div></div>' +
            '<div class="item-confidence">' + pct + '%</div>' +
            '</div>';
    }).join('');

    document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showToast(msg) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

document.addEventListener('DOMContentLoaded', init);
