const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const pdfUpload = document.getElementById('pdf-upload');
const uploadBtn = document.getElementById('upload-btn');
const uploadArea = document.getElementById('upload-area');
const uploadStatus = document.getElementById('upload-status');
const docList = document.getElementById('doc-list');

// ─── Document List ───────────────────────────────────────
async function loadDocuments() {
    try {
        const res = await fetch('/documents');
        const data = await res.json();
        docList.innerHTML = '';
        if (data.documents.length === 0) {
            docList.innerHTML = '<p class="no-docs">No documents loaded.</p>';
        } else {
            data.documents.forEach(name => {
                const item = document.createElement('div');
                item.className = 'doc-item';
                item.innerHTML = `
                    <span class="doc-icon">📄</span> 
                    <span class="doc-name" title="${name}">${name}</span>
                    <button class="delete-btn" onclick="deleteDocument('${name}')" title="Remove PDF">✖</button>
                `;
                docList.appendChild(item);
            });
        }
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

loadDocuments(); // Load on startup

// ─── PDF Upload Logic ────────────────────────────────────
uploadBtn.addEventListener('click', () => pdfUpload.click());
uploadArea.addEventListener('click', (e) => { if (e.target !== uploadBtn) pdfUpload.click(); });

// Drag & Drop
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
});

pdfUpload.addEventListener('change', () => {
    if (pdfUpload.files[0]) uploadFile(pdfUpload.files[0]);
});

async function uploadFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'docx', 'doc'].includes(ext)) {
        showStatus('❌ Only PDF and DOCX files are supported.', 'error');
        return;
    }

    showStatus(`<span class="spinner"></span> Uploading "${file.name}"...`, 'loading');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) {
            showStatus(`❌ ${data.error}`, 'error');
        } else {
            loadDocuments(); // Refresh the doc list
            startPolling(); // Wait for indexing to complete
        }
    } catch (e) {
        showStatus('❌ Upload failed. Is the server running?', 'error');
    }
}

function showStatus(msg, type) {
    uploadStatus.innerHTML = msg;
    uploadStatus.className = `upload-status ${type}`;
}

async function deleteDocument(filename) {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) return;
    showStatus(`<span class="spinner"></span> Deleting "${filename}"...`, 'loading');
    
    try {
        const res = await fetch(`/documents/${filename}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.error) showStatus(`❌ ${data.error}`, 'error');
        else {
            loadDocuments();
            startPolling(); // Poll until re-indexing is done
        }
    } catch (e) {
        showStatus('❌ Delete failed.', 'error');
    }
}

// ─── Status Polling Logic ─────────────────────────────────
let pollingInterval = null;

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    userInput.disabled = true;
    sendBtn.disabled = true;
    pdfUpload.disabled = true;
    uploadBtn.disabled = true;
    showStatus(`<span class="spinner"></span> 🟡 Indexing document in background...`, 'loading');

    pollingInterval = setInterval(async () => {
        try {
            const res = await fetch('/status');
            const data = await res.json();
            
            if (data.indexing === false) {
                // Done indexing
                clearInterval(pollingInterval);
                pollingInterval = null;
                userInput.disabled = false;
                sendBtn.disabled = false;
                pdfUpload.disabled = false;
                uploadBtn.disabled = false;
                showStatus(`🟢 Ready for questions.`, 'success');
            }
        } catch (e) {
            console.error('Polling failed:', e);
        }
    }, 2000);
}

// Check initially on load
startPolling();

// ─── Chat Logic ───────────────────────────────────────────
function appendMessage(text, role) {
    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-msg`;
    msgDiv.innerText = text;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return msgDiv;
}

async function handleSend() {
    const question = userInput.value.trim();
    if (!question) return;

    appendMessage(question, 'user');
    userInput.value = '';
    sendBtn.disabled = true;

    // Animated thinking dots
    const aiMsgDiv = appendMessage('', 'ai');
    let dots = 0;
    const thinkingInterval = setInterval(() => {
        dots = (dots + 1) % 4;
        aiMsgDiv.innerText = '⏳ Thinking' + '.'.repeat(dots);
    }, 400);

    try {
        // Use sync mode — reliable, tested, works 100%
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, stream: false })
        });

        const data = await response.json();
        clearInterval(thinkingInterval);

        if (data.error) {
            aiMsgDiv.innerText = '❌ ' + data.error;
            return;
        }

        // Typewriter effect for the answer
        aiMsgDiv.textContent = '';
        const answer = data.answer || 'No answer returned.';
        let i = 0;
        const typewriter = setInterval(() => {
            aiMsgDiv.textContent = answer.substring(0, i + 1);
            i++;
            chatContainer.scrollTop = chatContainer.scrollHeight;
            if (i >= answer.length) {
                clearInterval(typewriter);
                // Add metadata
                const metaDiv = document.createElement('div');
                metaDiv.className = 'metadata';
                metaDiv.innerHTML = `
                    <span>⚡ ${data.latency ? data.latency.toFixed(1) : '?'}s</span>
                    <span>📄 Docs: ${data.retrieved_docs || 0}</span>
                    <span>✅ Citations: ${data.citations_valid ? data.citations_valid.length : 0}</span>
                `;
                aiMsgDiv.appendChild(metaDiv);
            }
        }, 10);

    } catch (err) {
        clearInterval(thinkingInterval);
        aiMsgDiv.innerText = '❌ Could not connect to server. Make sure uvicorn is running.';
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter' && !sendBtn.disabled) handleSend(); });

function setQuery(text) {
    userInput.value = text;
    handleSend();
}
