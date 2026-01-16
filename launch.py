from pyngrok import ngrok
import subprocess
import threading
import time

# Insert your ngrok authtoken here after signup (replace the string below)
NGROK_AUTHTOKEN = "32jZ5AHFYk7fmDFvRRz4SusVXHP_5JniHbhn9FZYxTQxBPyLr"
ngrok.set_auth_token(NGROK_AUTHTOKEN)

def run_app():
    subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
    )

threading.Thread(target=run_app, daemon=True).start()
time.sleep(5)
public_url = ngrok.connect(8501)
print("Open this URL in your browser to use the app:")
print(public_url)
