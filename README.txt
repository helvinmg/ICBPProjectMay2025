NST + Upscaling Project By Helvin

This project performs Neural Style Transfer (NST) on images and then upscales the stylized images using Real-ESRGAN for enhanced resolution.

Running Locally
---
Prerequisites:
- Python 3.7 or higher
- GPU with CUDA support (recommended for Real-ESRGAN upscaling)
- Git

Setup:
1. Download project folder and extract
2. open the folder in VSCode
3. Install required Python packages using: pip install -r requirements.txt
4. Download Real-ESRGAN pretrained weights and place them in Real-ESRGAN/weights/ directory.

Running the Application:
Run the Streamlit app with: streamlit run app.py

Running on Google Colab
---
Setup:
1. Open a new Colab notebook.
2. Mount the drive folder which contains project code
3. Set path accordingly eg: %cd /content/drive/MyDrive/ICBP/ICBP Project
3. Install dependencies with: 
!pip install pyngrok --quiet
!pip install torch torchvision pillow matplotlib streamlit pyngrok realesrgan

Running the Application:
#Run the Streamlit app in the background and expose it using ngrok or localtunnel.
import subprocess
import threading
from pyngrok import ngrok

# 5. Setup ngrok auth token (replace with your actual token)
ngrok.set_auth_token("token")

# 🔧 Fix torchvision import issue in basicsr
!sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py

# 6. Function to print Streamlit logs live
def stream_logs(stream):
    for line in iter(stream.readline, b''):
        print(line.decode(), end='')

# 7. Start Streamlit in background with safe flags
port = 8501
proc = subprocess.Popen(
    [
        "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.fileWatcherType", "none",
        "--server.headless", "true",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# 8. Start threads to print logs live
threading.Thread(target=stream_logs, args=(proc.stdout,), daemon=True).start()
threading.Thread(target=stream_logs, args=(proc.stderr,), daemon=True).start()

# 9. Open ngrok tunnel
public_url = ngrok.connect(port).public_url
print(f"Streamlit app is live at: {public_url}")
---

Notes:
- Enable GPU runtime in Colab for faster upscaling.

