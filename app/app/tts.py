import threading, queue, pyttsx3

class Speaker:
    """Speaks in a background thread so the video never freezes."""
    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        engine = pyttsx3.init()
        while True:
            text = self.q.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("TTS error:", e)

    def say(self, text):
        self.q.put(text)