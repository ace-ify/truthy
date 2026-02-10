// ==========================================================================
// UnicornStudio Background Initialization
// ==========================================================================
(function initUnicornStudio() {
  if (!window.UnicornStudio) {
    window.UnicornStudio = { isInitialized: false };
    var script = document.createElement('script');
    script.src =
      'https://cdn.jsdelivr.net/gh/hiunicornstudio/unicornstudio.js@v1.4.29/dist/unicornStudio.umd.js';
    script.onload = function () {
      if (!window.UnicornStudio.isInitialized) {
        UnicornStudio.init();
        window.UnicornStudio.isInitialized = true;
      }
    };
    (document.head || document.body).appendChild(script);
  }
})();

// ==========================================================================
// Lucide Icons Initialization
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }
});

// ==========================================================================
// Voice Detection API Client
// ==========================================================================
const VoiceDetectionAPI = {
  // API configuration
  baseUrl: window.location.origin,
  apiKey: '', // Key not required for same-origin requests

  // Supported languages
  languages: ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu'],

  // Convert file to Base64
  async fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        // Remove data:audio/mp3;base64, prefix
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = (error) => reject(error);
    });
  },

  // Analyze voice sample
  async analyzeVoice(file, language) {
    try {
      const audioBase64 = await this.fileToBase64(file);

      const response = await fetch(`${this.baseUrl}/api/voice-detection`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { 'x-api-key': this.apiKey } : {})
        },
        body: JSON.stringify({
          language: language,
          audioFormat: file.type?.includes('wav') ? 'wav' : 'mp3',
          audioBase64: audioBase64
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail?.message || data.message || 'Analysis failed');
      }

      return data;
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  },

  // Check API health
  async checkHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`);
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      return { status: 'error', models_loaded: false };
    }
  }
};

// ==========================================================================
// Voice Recorder Controller
// ==========================================================================
const VoiceRecorder = {
  mediaRecorder: null,
  audioChunks: [],
  audioBlob: null,
  audioUrl: null,
  stream: null,
  audioContext: null,
  analyserNode: null,
  animationId: null,
  timerInterval: null,
  recordingSeconds: 0,
  isRecording: false,

  // Get all DOM elements
  getElements() {
    return {
      micBtn: document.getElementById('record-mic-btn'),
      micIcon: document.getElementById('mic-icon'),
      stopIcon: document.getElementById('stop-icon'),
      ring1: document.getElementById('record-ring-1'),
      ring2: document.getElementById('record-ring-2'),
      timer: document.getElementById('record-timer'),
      status: document.getElementById('record-status'),
      waveformContainer: document.getElementById('waveform-container'),
      canvas: document.getElementById('waveform-canvas'),
      idleState: document.getElementById('record-idle-state'),
      playbackState: document.getElementById('record-playback-state'),
      playbackAudio: document.getElementById('recording-playback'),
      durationLabel: document.getElementById('recording-duration'),
      reRecordBtn: document.getElementById('re-record-btn'),
    };
  },

  init() {
    const els = this.getElements();
    if (!els.micBtn) return;

    els.micBtn.addEventListener('click', () => this.toggleRecording());
    els.reRecordBtn?.addEventListener('click', () => this.resetToIdle());
  },

  async toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      await this.startRecording();
    }
  },

  async startRecording() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      UploadModal.showError('Microphone access denied. Please allow microphone access and try again.');
      return;
    }

    this.isRecording = true;
    this.audioChunks = [];
    this.recordingSeconds = 0;

    // Set up MediaRecorder
    this.mediaRecorder = new MediaRecorder(this.stream);
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.audioChunks.push(e.data);
    };
    this.mediaRecorder.onstop = () => this.onRecordingComplete();
    this.mediaRecorder.start();

    // Set up audio context for waveform
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = this.audioContext.createMediaStreamSource(this.stream);
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 256;
    source.connect(this.analyserNode);

    // Update UI to recording state
    const els = this.getElements();
    els.micIcon.classList.add('hidden');
    els.stopIcon.classList.remove('hidden');
    els.micBtn.classList.add('bg-red-500/20', 'border-red-500/50', 'shadow-[0_0_20px_rgba(239,68,68,0.3)]');
    els.ring1.classList.remove('hidden');
    els.ring2.classList.remove('hidden');
    els.status.textContent = 'Recording...';
    els.status.classList.remove('text-gray-500');
    els.status.classList.add('text-red-400');
    els.waveformContainer.classList.remove('hidden');

    // Start timer
    this.timerInterval = setInterval(() => {
      this.recordingSeconds++;
      const mins = Math.floor(this.recordingSeconds / 60);
      const secs = this.recordingSeconds % 60;
      els.timer.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    }, 1000);

    // Start waveform drawing
    this.drawWaveform();
  },

  stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
    this.isRecording = false;

    // Stop all tracks
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    // Stop timer & animation
    clearInterval(this.timerInterval);
    cancelAnimationFrame(this.animationId);

    // Close audio context
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    // Reset recording UI elements
    const els = this.getElements();
    els.micIcon.classList.remove('hidden');
    els.stopIcon.classList.add('hidden');
    els.micBtn.classList.remove('bg-red-500/20', 'border-red-500/50', 'shadow-[0_0_20px_rgba(239,68,68,0.3)]');
    els.ring1.classList.add('hidden');
    els.ring2.classList.add('hidden');
    els.waveformContainer.classList.add('hidden');
  },

  async onRecordingComplete() {
    this.audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
    this.audioUrl = URL.createObjectURL(this.audioBlob);

    const els = this.getElements();
    const mins = Math.floor(this.recordingSeconds / 60);
    const secs = this.recordingSeconds % 60;
    const durationStr = `${mins}:${secs.toString().padStart(2, '0')}`;

    // Switch to playback state
    els.idleState.classList.add('hidden');
    els.playbackState.classList.remove('hidden');
    els.durationLabel.textContent = `Recording — ${durationStr}`;
    els.playbackAudio.src = this.audioUrl;

    // Convert WebM to WAV so the backend (librosa) can process it
    try {
      const wavBlob = await this.convertToWav(this.audioBlob);
      const file = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
      UploadModal.selectedFile = file;
    } catch (err) {
      console.error('WAV conversion failed, using WebM:', err);
      const file = new File([this.audioBlob], `recording_${Date.now()}.webm`, { type: 'audio/webm' });
      UploadModal.selectedFile = file;
    }

    // Show the enabled analyze button
    if (UploadModal.analyzeBtnDisabled) UploadModal.analyzeBtnDisabled.classList.add('hidden');
    if (UploadModal.analyzeBtn) UploadModal.analyzeBtn.classList.remove('hidden');

    UploadModal.hideError();
  },

  async convertToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    audioCtx.close();

    // Encode AudioBuffer to 16-bit PCM WAV
    const numChannels = 1; // mono
    const sampleRate = audioBuffer.sampleRate;
    const samples = audioBuffer.getChannelData(0);
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);           // chunk size
    view.setUint16(20, 1, true);            // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true); // byte rate
    view.setUint16(32, numChannels * 2, true); // block align
    view.setUint16(34, 16, true);           // bits per sample
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Write PCM samples
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
  },

  drawWaveform() {
    const els = this.getElements();
    const canvas = els.canvas;
    if (!canvas || !this.analyserNode) return;

    const ctx = canvas.getContext('2d');
    const bufferLength = this.analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    // Set canvas internal resolution
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;

    const draw = () => {
      this.animationId = requestAnimationFrame(draw);
      this.analyserNode.getByteFrequencyData(dataArray);

      ctx.fillStyle = 'rgba(10, 10, 12, 0.85)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height * 0.8;
        const hue = 220 + (dataArray[i] / 255) * 30; // blue range
        const opacity = 0.4 + (dataArray[i] / 255) * 0.6;
        ctx.fillStyle = `hsla(${hue}, 80%, 60%, ${opacity})`;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    };
    draw();
  },

  resetToIdle() {
    const els = this.getElements();

    // Clean up audio URL
    if (this.audioUrl) {
      URL.revokeObjectURL(this.audioUrl);
      this.audioUrl = null;
    }
    this.audioBlob = null;
    this.audioChunks = [];
    this.recordingSeconds = 0;

    // Reset UI
    els.playbackState.classList.add('hidden');
    els.idleState.classList.remove('hidden');
    els.timer.textContent = '0:00';
    els.status.textContent = 'Click to start recording';
    els.status.classList.remove('text-red-400');
    els.status.classList.add('text-gray-500');
    if (els.playbackAudio) els.playbackAudio.src = '';

    // Reset selected file & button
    UploadModal.selectedFile = null;
    if (UploadModal.analyzeBtn) UploadModal.analyzeBtn.classList.add('hidden');
    if (UploadModal.analyzeBtnDisabled) UploadModal.analyzeBtnDisabled.classList.remove('hidden');
  },

  destroy() {
    // Full cleanup when modal closes
    if (this.isRecording) this.stopRecording();
    this.resetToIdle();
  }
};

// ==========================================================================
// Upload Modal Controller
// ==========================================================================
const UploadModal = {
  modal: null,
  dropZone: null,
  fileInput: null,
  languageSelect: null,
  analyzeBtn: null,
  analyzeBtnDisabled: null,
  resultsSection: null,
  selectedFile: null,
  activeTab: 'upload',

  init() {
    this.modal = document.getElementById('upload-modal');
    this.dropZone = document.getElementById('drop-zone');
    this.fileInput = document.getElementById('audio-file-input');
    this.languageSelect = document.getElementById('language-select');
    this.analyzeBtn = document.getElementById('analyze-btn');
    this.analyzeBtnDisabled = document.getElementById('analyze-btn-disabled');
    this.resultsSection = document.getElementById('results-section');

    if (!this.modal) return;

    this.setupEventListeners();
    VoiceRecorder.init();
  },

  setupEventListeners() {
    // Open modal triggers
    document.querySelectorAll('[data-open-modal]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        this.open();
      });
    });

    // Close modal triggers
    document.querySelectorAll('[data-close-modal]').forEach(btn => {
      btn.addEventListener('click', () => this.close());
    });

    // Close on backdrop click
    this.modal?.addEventListener('click', (e) => {
      if (e.target === this.modal) this.close();
    });

    // Close on escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.modal?.classList.contains('active')) {
        this.close();
      }
    });

    // Tab switching
    const tabUpload = document.getElementById('tab-upload');
    const tabRecord = document.getElementById('tab-record');
    tabUpload?.addEventListener('click', () => this.switchTab('upload'));
    tabRecord?.addEventListener('click', () => this.switchTab('record'));

    // Drag and drop handling
    if (this.dropZone) {
      ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        this.dropZone.addEventListener(eventName, (e) => {
          e.preventDefault();
          e.stopPropagation();
        });
      });

      ['dragenter', 'dragover'].forEach(eventName => {
        this.dropZone.addEventListener(eventName, () => {
          this.dropZone.classList.add('drag-over');
        });
      });

      ['dragleave', 'drop'].forEach(eventName => {
        this.dropZone.addEventListener(eventName, () => {
          this.dropZone.classList.remove('drag-over');
        });
      });

      this.dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
          this.handleFileSelect(files[0]);
        }
      });

      this.dropZone.addEventListener('click', () => {
        this.fileInput?.click();
      });
    }

    // File input change
    this.fileInput?.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        this.handleFileSelect(e.target.files[0]);
      }
    });

    // Analyze button
    this.analyzeBtn?.addEventListener('click', () => this.analyze());
  },

  switchTab(tab) {
    this.activeTab = tab;
    const tabUpload = document.getElementById('tab-upload');
    const tabRecord = document.getElementById('tab-record');
    const uploadContent = document.getElementById('upload-tab-content');
    const recordContent = document.getElementById('record-tab-content');

    if (tab === 'upload') {
      tabUpload.classList.add('bg-white/10', 'text-white');
      tabUpload.classList.remove('text-gray-400');
      tabRecord.classList.remove('bg-white/10', 'text-white');
      tabRecord.classList.add('text-gray-400');
      uploadContent?.classList.remove('hidden');
      recordContent?.classList.add('hidden');

      // Clean up recorder when switching away
      VoiceRecorder.destroy();
    } else {
      tabRecord.classList.add('bg-white/10', 'text-white');
      tabRecord.classList.remove('text-gray-400');
      tabUpload.classList.remove('bg-white/10', 'text-white');
      tabUpload.classList.add('text-gray-400');
      recordContent?.classList.remove('hidden');
      uploadContent?.classList.add('hidden');

      // Reset upload state when switching away
      this.resetUploadState();
    }

    // Reset analyze button when switching tabs
    this.selectedFile = null;
    if (this.analyzeBtn) this.analyzeBtn.classList.add('hidden');
    if (this.analyzeBtnDisabled) this.analyzeBtnDisabled.classList.remove('hidden');
    this.hideError();
    this.hideResults();
  },

  resetUploadState() {
    if (this.fileInput) this.fileInput.value = '';
    const fileInfoEl = document.getElementById('selected-file-info');
    const dropText = document.getElementById('drop-zone-text');
    if (fileInfoEl) fileInfoEl.classList.add('hidden');
    if (dropText) dropText.classList.remove('hidden');
  },

  handleFileSelect(file) {
    // Validate file type
    const validTypes = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/m4a', 'audio/x-m4a'];
    const validExtensions = ['.mp3', '.wav', '.m4a'];
    const ext = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

    if (!validTypes.includes(file.type) && !validExtensions.includes(ext)) {
      this.showError('Please select a valid audio file (MP3, WAV, or M4A)');
      return;
    }

    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
      this.showError('File size must be less than 50MB');
      return;
    }

    this.selectedFile = file;
    this.updateFileDisplay(file);
    this.hideError();
    this.hideResults();
  },

  updateFileDisplay(file) {
    const fileNameEl = document.getElementById('selected-file-name');
    const fileSizeEl = document.getElementById('selected-file-size');
    const fileInfoEl = document.getElementById('selected-file-info');
    const dropText = document.getElementById('drop-zone-text');

    if (fileNameEl) fileNameEl.textContent = file.name;
    if (fileSizeEl) fileSizeEl.textContent = this.formatFileSize(file.size);
    if (fileInfoEl) fileInfoEl.classList.remove('hidden');
    if (dropText) dropText.classList.add('hidden');

    // Show enabled button (CTA style), hide disabled button
    if (this.analyzeBtnDisabled) this.analyzeBtnDisabled.classList.add('hidden');
    if (this.analyzeBtn) this.analyzeBtn.classList.remove('hidden');
  },

  formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  },

  async analyze() {
    if (!this.selectedFile) {
      this.showError('Please select or record an audio file first');
      return;
    }

    const language = this.languageSelect?.value || 'English';

    // Show loading state
    this.setLoading(true);
    this.hideError();
    this.hideResults();

    try {
      const result = await VoiceDetectionAPI.analyzeVoice(this.selectedFile, language);
      this.showResults(result);
    } catch (error) {
      this.showError(error.message || 'Analysis failed. Please try again.');
    } finally {
      this.setLoading(false);
    }
  },

  showResults(result) {
    const resultsEl = document.getElementById('results-section');
    if (!resultsEl) return;

    const isAI = result.classification === 'AI_GENERATED';
    const confidencePercent = Math.round(result.confidenceScore * 100);

    document.getElementById('result-classification').textContent =
      isAI ? 'AI Generated' : 'Human Voice';
    document.getElementById('result-classification').className =
      `text-2xl font-bold ${isAI ? 'text-red-400' : 'text-green-400'}`;

    document.getElementById('result-confidence').textContent = `${confidencePercent}%`;
    document.getElementById('result-language').textContent = result.language;
    document.getElementById('result-explanation').textContent = result.explanation;

    // Update confidence bar
    const bar = document.getElementById('confidence-bar');
    if (bar) {
      bar.style.width = `${confidencePercent}%`;
      bar.className = `h-full rounded-full transition-all duration-500 ${isAI ? 'bg-gradient-to-r from-red-500 to-orange-500' : 'bg-gradient-to-r from-green-500 to-emerald-500'
        }`;
    }

    // Update icon
    const iconEl = document.getElementById('result-icon');
    if (iconEl) {
      iconEl.innerHTML = isAI
        ? '<svg class="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>'
        : '<svg class="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
    }

    resultsEl.classList.remove('hidden');
  },

  hideResults() {
    const resultsEl = document.getElementById('results-section');
    if (resultsEl) resultsEl.classList.add('hidden');
  },

  showError(message) {
    const errorEl = document.getElementById('upload-error');
    if (errorEl) {
      errorEl.textContent = message;
      errorEl.classList.remove('hidden');
    }
  },

  hideError() {
    const errorEl = document.getElementById('upload-error');
    if (errorEl) errorEl.classList.add('hidden');
  },

  setLoading(loading) {
    if (this.analyzeBtn) {
      if (loading) {
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.innerHTML = `
          <svg class="animate-spin -ml-1 mr-2 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Analyzing...
        `;
      } else {
        this.analyzeBtn.disabled = false;
        this.analyzeBtn.innerHTML = `
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
          </svg>
          Analyze Voice
        `;
      }
    }
  },

  open() {
    if (this.modal) {
      this.modal.classList.add('active');
      document.body.style.overflow = 'hidden';
    }
  },

  close() {
    if (this.modal) {
      this.modal.classList.remove('active');
      document.body.style.overflow = '';
      this.reset();
    }
  },

  reset() {
    this.selectedFile = null;
    if (this.fileInput) this.fileInput.value = '';

    const fileInfoEl = document.getElementById('selected-file-info');
    const dropText = document.getElementById('drop-zone-text');

    if (fileInfoEl) fileInfoEl.classList.add('hidden');
    if (dropText) dropText.classList.remove('hidden');
    
    // Show disabled button (How It Works style), hide enabled button
    if (this.analyzeBtn) this.analyzeBtn.classList.add('hidden');
    if (this.analyzeBtnDisabled) this.analyzeBtnDisabled.classList.remove('hidden');

    // Clean up recorder
    VoiceRecorder.destroy();

    // Reset to upload tab
    this.switchTab('upload');

    this.hideError();
    this.hideResults();
  }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  UploadModal.init();
});

// ==========================================================================
// Scroll Animation Observer
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.animationPlayState = 'running';
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
  );
  document.querySelectorAll('.scroll-item').forEach((el) => observer.observe(el));
});

// ==========================================================================
// Dashboard Grid - Counter & In-View Animations
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('dashboard-grid');
  if (!container) return;

  const counters = document.querySelectorAll('[data-counter-target]');
  const LOOP_DURATION = 6000; // Matches CSS animation duration
  let counterInterval;

  const runCounterAnimation = () => {
    counters.forEach((counter) => {
      const target = +counter.getAttribute('data-counter-target');
      const prefix = counter.getAttribute('data-counter-prefix') || '';
      const suffix = counter.getAttribute('data-counter-suffix') || '';

      let count = 0;
      const duration = 1500; // Counter duration
      const increment = target / (duration / 20);

      counter.innerText = prefix + '0' + suffix;

      const timer = setInterval(() => {
        count += increment;
        if (count >= target) {
          count = target;
          clearInterval(timer);
        }
        counter.innerText = prefix + Math.ceil(count) + suffix;
      }, 20);
    });
  };

  const gridObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          container.classList.add('in-view');
          runCounterAnimation();

          if (!counterInterval) {
            counterInterval = setInterval(() => {
              if (container.classList.contains('in-view')) {
                runCounterAnimation();
              }
            }, LOOP_DURATION);
          }
        } else {
          container.classList.remove('in-view');
        }
      });
    },
    { threshold: 0.3 }
  );

  gridObserver.observe(container);
});

// ==========================================================================
// Testimonial Slider
// ==========================================================================
(function () {
  const testimonials = [
    {
      quote:
        'Truthy detected a deepfake scam call targeting our customer support. The multi-language detection is phenomenal—works perfectly for our Tamil and Hindi speaking users!',
      name: 'Priya Sharma',
      role: 'Security Lead, TechVault India',
      img: 'https://images.unsplash.com/photo-1522529599102-193c0d76b5b6?q=80&w=1000&auto=format&fit=crop',
    },
    {
      quote:
        'We integrated Truthy API into our verification pipeline. It catches AI-generated voice clones with 98% accuracy. The confidence scores are incredibly reliable.',
      name: 'Rajesh Kumar',
      role: 'CTO, SecureVoice Systems',
      img: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?q=80&w=1000&auto=format&fit=crop',
    },
    {
      quote:
        "As a content verification platform, detecting synthetic speech is critical. Truthy's multi-language support for Telugu and Malayalam sets it apart from competitors.",
      name: 'Arun Krishnan',
      role: 'Product Manager, MediaGuard',
      img: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=1000&auto=format&fit=crop',
    },
  ];

  let currentIndex = 0;

  function update(index) {
    const imgEl = document.getElementById('testimonial-img');
    const quoteEl = document.getElementById('testimonial-quote');
    const nameEl = document.getElementById('testimonial-name');
    const roleEl = document.getElementById('testimonial-role');

    const t = testimonials[index];
    if (imgEl && quoteEl && nameEl && roleEl) {
      imgEl.style.opacity = '0';
      quoteEl.style.opacity = '0';

      setTimeout(() => {
        imgEl.src = t.img;
        quoteEl.innerText = '"' + t.quote + '"';
        nameEl.innerText = t.name;
        roleEl.innerText = t.role;

        imgEl.style.opacity = '1';
        quoteEl.style.opacity = '1';
      }, 300);
    }
  }

  window.nextTestimonial = function () {
    currentIndex = (currentIndex + 1) % testimonials.length;
    update(currentIndex);
  };

  window.prevTestimonial = function () {
    currentIndex = (currentIndex - 1 + testimonials.length) % testimonials.length;
    update(currentIndex);
  };
})();
