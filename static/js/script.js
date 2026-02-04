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
          audioFormat: 'mp3',
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
      this.showError('Please select an audio file first');
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
