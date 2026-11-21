const pages = [...document.querySelectorAll(".page")];
const indicators = [...document.querySelectorAll("[data-step-indicator]")];
const cameraVideo = document.querySelector("#camera-video");
const cameraFrame = document.querySelector("#camera-frame");
const cameraPlaceholder = document.querySelector("#camera-placeholder");
const cameraError = document.querySelector("#camera-error");
const startCameraButton = document.querySelector("#start-camera");
const captureButton = document.querySelector("#capture-button");
const stopCameraButton = document.querySelector("#stop-camera");
const imageUpload = document.querySelector("#image-upload");
const messageInput = document.querySelector("#message-input");
const lightingState = document.querySelector("#lighting-state");
const lightingDescription = document.querySelector("#lighting-description");
const lightingMeter = document.querySelector("#lighting-meter-fill");
const handState = document.querySelector("#hand-state");
const backgroundState = document.querySelector("#background-state");
const backgroundDescription = document.querySelector("#background-description");

let direction = "";
let cameraStream = null;
let capturedImage = "";
let lightingTimer = null;
let analysisTimer = null;
let analysisInFlight = false;
let backendAvailable = false;
let cameraSession = 0;

function showPage(step) {
  pages.forEach((page, index) => {
    const active = index + 1 === step;
    page.hidden = !active;
    page.classList.toggle("is-visible", active);
  });
  indicators.forEach((indicator) => {
    const number = Number(indicator.dataset.stepIndicator);
    indicator.classList.toggle("is-active", number === step);
    indicator.classList.toggle("is-done", number < step);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function stopCamera() {
  cameraSession += 1;
  if (lightingTimer) {
    window.clearInterval(lightingTimer);
    lightingTimer = null;
  }
  if (analysisTimer) {
    window.clearInterval(analysisTimer);
    analysisTimer = null;
  }
  backendAvailable = false;
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraFrame.classList.remove("is-live");
  document.querySelector("#camera-live-badge").hidden = true;
  document.querySelector("#camera-bottom-label").hidden = true;
  cameraPlaceholder.hidden = false;
  startCameraButton.disabled = false;
  captureButton.disabled = true;
  stopCameraButton.hidden = true;
  updateLightingState(null);
  handState.className = "check-state state-pending";
  handState.innerHTML = "<span></span> Not connected";
  backgroundState.className = "check-state state-pending";
  backgroundState.innerHTML = "<span></span> Not connected";
  backgroundDescription.textContent = "Needs preprocessing service";
}

function updateLightingState(brightness) {
  if (brightness === null) {
    lightingState.className = "check-state state-pending";
    lightingState.innerHTML = "<span></span> Waiting";
    lightingDescription.textContent = "We’ll check as you go";
    lightingMeter.style.width = "0";
    return;
  }

  const percent = Math.max(0, Math.min(100, Math.round((brightness / 180) * 100)));
  lightingMeter.style.width = `${percent}%`;
  if (brightness < 65) {
    lightingState.className = "check-state state-alert";
    lightingState.innerHTML = "<span></span> A little dim";
    lightingDescription.textContent = `Low light · ${Math.round(brightness)} / 255`;
    lightingMeter.style.background = "#dfa46e";
  } else {
    lightingState.className = "check-state state-good";
    lightingState.innerHTML = "<span></span> Looking good";
    lightingDescription.textContent = `Good light · ${Math.round(brightness)} / 255`;
    lightingMeter.style.background = "#7caa9a";
  }
}

function checkLighting() {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) return;
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 24;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return;
  context.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  let totalBrightness = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    totalBrightness += 0.299 * pixels[index] + 0.587 * pixels[index + 1] + 0.114 * pixels[index + 2];
  }
  updateLightingState(totalBrightness / (pixels.length / 4));
}

function setCheckState(element, state, label) {
  element.className = `check-state ${state}`;
  element.innerHTML = `<span></span> ${label}`;
}

async function requestPreprocessing(image, removeBackground = false) {
  let response;
  try {
    response = await fetch("/api/process-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image, remove_background: removeBackground }),
    });
  } catch {
    throw new Error("Can’t reach the local preprocessing service. Start the app with .venv\\Scripts\\python.exe app\\server.py.");
  }

  const result = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(result?.error || "The preprocessing service couldn’t process this frame.");
  }
  return result;
}

async function analyzeLiveFrame() {
  if (!backendAvailable || analysisInFlight || !cameraVideo.videoWidth) return;
  const session = cameraSession;
  analysisInFlight = true;
  try {
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = Math.round((cameraVideo.videoHeight / cameraVideo.videoWidth) * canvas.width);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Unable to read a camera frame.");
    context.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
    const result = await requestPreprocessing(canvas.toDataURL("image/jpeg", 0.65));
    if (session !== cameraSession) return;
    updateLightingState(result.brightness);
    setCheckState(
      handState,
      result.hand_detected ? "state-good" : "state-pending",
      result.hand_detected ? "Hand detected" : "Show your hand",
    );
  } catch (error) {
    if (session !== cameraSession) return;
    backendAvailable = false;
    if (analysisTimer) {
      window.clearInterval(analysisTimer);
      analysisTimer = null;
    }
    setCheckState(handState, "state-alert", "Service error");
    cameraError.textContent = error.message;
  } finally {
    analysisInFlight = false;
  }
}

async function connectPreprocessingService(session) {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    if (!response.ok || !status.ok || status.hand_detection !== "ready") {
      throw new Error(status.error || "The hand landmark model is not ready.");
    }
    if (session !== cameraSession || !cameraStream) return;
    backendAvailable = true;
    setCheckState(handState, "state-pending", "Looking for hand");
    setCheckState(backgroundState, "state-good", "Available");
    backgroundDescription.textContent = "Ready to isolate your hand";
    analysisTimer = window.setInterval(analyzeLiveFrame, 1000);
    analyzeLiveFrame();
  } catch (error) {
    if (session !== cameraSession || !cameraStream) return;
    backendAvailable = false;
    setCheckState(handState, "state-alert", "Service offline");
    setCheckState(backgroundState, "state-alert", "Unavailable");
    cameraError.textContent = `${error.message || "Preprocessing service is unavailable."} Start it with .venv\\Scripts\\python.exe app\\server.py.`;
  }
}

function setDirection(nextDirection) {
  direction = nextDirection;
  cameraError.textContent = "";
  document.querySelector("#text-error").textContent = "";
  document.querySelector("#setup-mode-label").textContent =
    direction === "sign-to-text" ? "SIGN TO TEXT" : "TEXT TO SIGN";
  const isCamera = direction === "sign-to-text";
  document.querySelector("#setup-title-accent").textContent = isCamera ? "camera" : "message";
  document.querySelector("#setup-title").firstChild.textContent = isCamera
    ? "Let’s get your "
    : "Let’s prepare your ";
  document.querySelector(".heading-suffix").textContent = isCamera ? " ready." : ".";
  document.querySelector("#setup-intro").textContent = isCamera
    ? "Find a comfortable spot in frame. We’ll help you check the lighting."
    : "Write a message and we’ll get it ready for your sign sequence.";
  document.querySelector("#camera-workspace").hidden = !isCamera;
  document.querySelector("#camera-footer").hidden = !isCamera;
  document.querySelector("#text-workspace").hidden = isCamera;
  showPage(2);
}

async function startCamera() {
  cameraError.textContent = "";
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraError.textContent = "Camera access isn’t available here. Open this page on localhost or use image upload.";
    return;
  }
  startCameraButton.disabled = true;
  const session = ++cameraSession;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    });
    if (session !== cameraSession) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    cameraStream = stream;
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    cameraPlaceholder.hidden = true;
    cameraFrame.classList.add("is-live");
    document.querySelector("#camera-live-badge").hidden = false;
    document.querySelector("#camera-bottom-label").hidden = false;
    captureButton.disabled = false;
    stopCameraButton.hidden = false;
    handState.className = "check-state state-pending";
    handState.innerHTML = "<span></span> Connecting";
    lightingTimer = window.setInterval(checkLighting, 1200);
    checkLighting();
    connectPreprocessingService(session);
  } catch (error) {
    if (session !== cameraSession) return;
    stopCamera();
    startCameraButton.disabled = false;
    const denied = error && (error.name === "NotAllowedError" || error.name === "PermissionDeniedError");
    cameraError.textContent = denied
      ? "Camera permission was denied. Allow access in your browser settings, or upload an image instead."
      : "We couldn’t start the camera. Check that it’s connected and not being used by another app.";
  }
}

function showResult(image, processingResult = null) {
  capturedImage = image || capturedImage;
  document.querySelector("#result-type-label").textContent =
    direction === "sign-to-text" ? "SIGN TO TEXT" : "TEXT TO SIGN";
  document.querySelector("#sign-result").hidden = direction !== "sign-to-text";
  document.querySelector("#text-result").hidden = direction !== "text-to-sign";
  document.querySelector("#speak-result").hidden = direction !== "sign-to-text";
  document.querySelector("#result-intro").textContent =
    direction === "sign-to-text"
      ? "Your sign input is ready. Here’s the current translation status."
      : "Your message is ready to be turned into a sign sequence.";

  const imageWrap = document.querySelector("#result-image-wrap");
  if (direction === "sign-to-text" && capturedImage) {
    document.querySelector("#result-image").src =
      processingResult?.processed_image || capturedImage;
    imageWrap.hidden = false;
  } else {
    imageWrap.hidden = true;
  }
  if (direction === "sign-to-text" && processingResult) {
    const handDetected = processingResult.hand_detected;
    document.querySelector("#sign-result-title").textContent = handDetected
      ? "Hand detected and processed."
      : "No hand detected yet.";
    document.querySelector("#sign-result-message").textContent = handDetected
      ? `${processingResult.low_light ? "Low-light enhancement" : "Lighting check"} and background removal are complete. Sign-to-text recognition still needs its trained model.`
      : "We checked the frame but couldn’t find a hand. Try centering your hand in the camera and capture again.";
    document.querySelector("#sign-result-status-text").textContent = handDetected
      ? processingResult.background_removed
        ? "Hand detected · background removed · translation model not connected"
        : "Hand detected · translation model not connected"
      : "No hand detected · try another frame";
  }
  if (direction === "text-to-sign") {
    const message = messageInput.value.trim();
    document.querySelector("#result-message-text").textContent = message;
    const words = message.match(/[\p{L}\p{N}’'-]+/gu) || [];
    document.querySelector("#word-sequence").replaceChildren(
      ...words.map((word) => {
        const chip = document.createElement("span");
        chip.className = "word-chip";
        chip.textContent = word;
        return chip;
      }),
    );
  }
  stopCamera();
  showPage(3);
}

document.querySelectorAll("[data-direction]").forEach((button) => {
  button.addEventListener("click", () => setDirection(button.dataset.direction));
});

document.querySelectorAll("[data-back]").forEach((button) => {
  button.addEventListener("click", () => {
    stopCamera();
    showPage(Number(button.dataset.back));
  });
});

document.querySelectorAll("[data-home]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    stopCamera();
    showPage(1);
  });
});

document.querySelector("#start-camera").addEventListener("click", startCamera);
stopCameraButton.addEventListener("click", () => {
  stopCamera();
  cameraError.textContent = "Camera stopped.";
});

captureButton.addEventListener("click", async () => {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
    cameraError.textContent = "Wait for the camera preview before capturing.";
    return;
  }
  const canvas = document.createElement("canvas");
  canvas.width = cameraVideo.videoWidth;
  canvas.height = cameraVideo.videoHeight;
  const context = canvas.getContext("2d");
  if (!context) {
    cameraError.textContent = "We couldn’t capture this frame. Please try again.";
    return;
  }
  context.translate(canvas.width, 0);
  context.scale(-1, 1);
  context.drawImage(cameraVideo, 0, 0);
  const image = canvas.toDataURL("image/jpeg", 0.88);
  captureButton.disabled = true;
  cameraError.textContent = "Checking for a hand and preparing your image…";
  try {
    const result = await requestPreprocessing(image, true);
    showResult(image, result);
  } catch (error) {
    cameraError.textContent = error.message;
    captureButton.disabled = false;
  }
});

imageUpload.addEventListener("change", () => {
  const file = imageUpload.files && imageUpload.files[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    cameraError.textContent = "Choose an image file to continue.";
    imageUpload.value = "";
    return;
  }
  if (file.size > 5 * 1024 * 1024) {
    cameraError.textContent = "This image is over 5 MB. Choose a smaller image.";
    imageUpload.value = "";
    return;
  }
  const reader = new FileReader();
  reader.addEventListener("load", async () => {
    if (typeof reader.result !== "string") return;
    cameraError.textContent = "Checking for a hand and preparing your image…";
    try {
      const result = await requestPreprocessing(reader.result, true);
      showResult(reader.result, result);
    } catch (error) {
      cameraError.textContent = error.message;
    }
  });
  reader.addEventListener("error", () => {
    cameraError.textContent = "We couldn’t read this image. Please try another.";
  });
  reader.readAsDataURL(file);
});

messageInput.addEventListener("input", () => {
  document.querySelector("#character-count").textContent = String(messageInput.value.length);
  document.querySelector("#text-error").textContent = "";
});

document.querySelector("#continue-text").addEventListener("click", () => {
  if (!messageInput.value.trim()) {
    document.querySelector("#text-error").textContent = "Add a message first so we have something to prepare.";
    messageInput.focus();
    return;
  }
  showResult();
});

document.querySelector("#speak-result").addEventListener("click", () => {
  if (!("speechSynthesis" in window)) {
    cameraError.textContent = "Speech output isn’t supported by this browser.";
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(
    "Your sign input is captured. Connect the recognition service to turn it into text or speech.",
  );
  window.speechSynthesis.speak(utterance);
});

document.querySelector("#start-again").addEventListener("click", () => {
  stopCamera();
  direction = "";
  capturedImage = "";
  messageInput.value = "";
  document.querySelector("#character-count").textContent = "0";
  imageUpload.value = "";
  showPage(1);
});

window.addEventListener("pagehide", stopCamera);
