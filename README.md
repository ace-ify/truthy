<p align="center">
  <img src="https://see.fontimg.com/api/rf5/rv0aB/ZTRlNjgyYThiNTRjNDE1ZWFiYzliZmY5OGI5MDhhM2Yub3Rm/dHJ1dGh5/teknaf-regular.png?r=fs&h=81&w=1250&fg=3b82f6&bg=000000&s=65" alt="Truthy Logo" height="60">
</p>

<h1 align="center">Truthy - AI Voice Detection</h1>

<p align="center">
  <strong>Detect AI-generated voices and deepfake audio in real-time</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#installation">Installation</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#tech-stack">Tech Stack</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

---

## 🎯 Problem Statement

With the rise of AI voice cloning and deepfake audio, detecting synthetic voices has become critical for:
- **Financial Security**: Preventing voice-based fraud and scam calls
- **Media Verification**: Authenticating audio content in journalism
- **Identity Protection**: Safeguarding against voice spoofing attacks

**Truthy** solves this by providing a fast, accurate, and multilingual AI voice detection API.

---

## ✨ Features

- 🎙️ **Real-time Detection** - Analyze audio samples in under 5 seconds
- 🌍 **Multilingual Support** - English, Hindi, Tamil, Telugu, Malayalam
- 📊 **Confidence Scoring** - Get probability scores with explanations
- 🔒 **Secure API** - API key authentication with rate limiting
- 📱 **Modern UI** - Beautiful, responsive web interface
- ⚡ **Fast Inference** - Optimized for CPU and GPU

---

## 🖥️ Demo

### API Response Example
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Strong synthetic voice patterns detected. Audio shows clear signs of AI generation with unnatural pitch consistency."
}
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip
- 4GB+ RAM (for model loading)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/ace-ify/truthy.git
cd truthy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

---

## 📚 API Reference

### Authentication
All API requests require an API key in the header:
```
x-api-key: your_api_key_here
```

### Endpoints

#### `POST /api/voice-detection`
Analyze Base64-encoded audio for AI voice detection.

**Request Body:**
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "base64_encoded_audio_string"
}
```

**Response:**
```json
{
  "status": "success",
  "language": "English",
  "classification": "HUMAN | AI_GENERATED",
  "confidenceScore": 0.95,
  "explanation": "Audio exhibits natural human speech characteristics..."
}
```

#### `GET /api/health`
Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda"
}
```

#### `GET /api/info`
Get API information and supported languages.

### Error Responses
```json
{
  "status": "error",
  "message": "Error description"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid/missing API key |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance API framework |
| **PyTorch** | Deep learning inference |
| **WavLM** | Pre-trained audio model for detection |
| **Silero VAD** | Voice Activity Detection |
| **librosa** | Audio processing |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5/CSS3** | Structure and styling |
| **TailwindCSS** | Utility-first CSS |
| **JavaScript** | Client-side logic |
| **Lucide Icons** | Icon library |

### Infrastructure
| Service | Purpose |
|---------|---------|
| **Render** | Cloud deployment |
| **Hugging Face** | Model hosting |
| **GitHub** | Version control |

---

## 📁 Project Structure

```
truthy/
├── api/
│   ├── main.py          # FastAPI application
│   └── models.py        # Pydantic schemas
├── core/
│   ├── detector.py      # Deepfake detection model
│   ├── vad.py           # Voice Activity Detection
│   ├── audio_processor.py  # Audio preprocessing
│   └── aggregator.py    # Result aggregation
├── static/
│   ├── index.html       # Frontend UI
│   ├── css/style.css    # Custom styles
│   └── js/script.js     # Client-side JS
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── Procfile             # Deployment config
└── README.md            # This file
```

---

## 🌐 Deployment

### Deploy to Render

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Create Render Web Service**
   - Go to [render.com](https://render.com)
   - New → Web Service
   - Connect your GitHub repo
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables**
   | Variable | Value |
   |----------|-------|
   | `API_KEYS` | `demo_key_12345` |
   | `PYTHONUNBUFFERED` | `1` |

4. **Deploy** - Click "Create Web Service"

### Live URL
Your API will be available at: `https://your-app.onrender.com`

---

## 🔐 Security

- ✅ API Key authentication
- ✅ Rate limiting (100 requests/minute)
- ✅ HTTPS encryption (via Render)
- ✅ Input validation
- ✅ Error handling without sensitive data

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Average Response Time | < 3 seconds |
| Model Accuracy | ~95% |
| Supported Audio Length | Up to 5 minutes |
| Max File Size | 50 MB |

---

## 🗺️ Roadmap

- [ ] Add more Indian languages (Kannada, Bengali, Gujarati)
- [ ] Real-time streaming detection
- [ ] Browser extension for audio verification
- [ ] Mobile SDK (iOS/Android)
- [ ] Enhanced API analytics dashboard

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

Built with ❤️ for hackathon by the Truthy Team

---

## 📞 Support

- 📧 Email: 21naimish21@gmail.com
- 📖 Documentation: `/docs` endpoint (FastAPI auto-generated)
- 🐛 Issues: [GitHub Issues](https://github.com/ace-ify/truthy/issues)

---

<p align="center">
  <strong>Truthy</strong> - Detect. Verify. Trust.
</p>
