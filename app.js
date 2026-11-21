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
let signCatalogPromise = null;
let signSequence = [];
let activeSignIndex = 0;
let activeFrameIndex = 0;
let playbackTimer = null;
let frameAdvancePending = false;
const preloadedDatasetFrames = new Map();
let isSequencePlaying = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let lastRecognitionText = "";

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
  syncPlaybackTimer();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function syncPlaybackTimer() {
  if (playbackTimer) {
    window.clearInterval(playbackTimer);
    playbackTimer = null;
  }
  if (isSequencePlaying && signSequence.length && !document.querySelector("#page-3").hidden) {
    playbackTimer = window.setInterval(advanceDatasetFrame, 220);
  }
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
  const indicator = document.createElement("span");
  element.replaceChildren(indicator, document.createTextNode(` ${label}`));
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
      result.hand_detected && result.recognition?.recognized
        ? "state-good"
        : "state-pending",
      result.hand_detected && result.recognition?.recognized
        ? `Likely sign: ${formatSignTitle(result.recognition.label)}`
        : result.hand_detected
          ? "Hand detected"
          : "Show your hand",
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
    setCheckState(
      handState,
      "state-pending",
      status.recognition?.state === "loading" ? "Preparing dataset" : "Looking for hand",
    );
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
    ? "Find a comfortable spot in frame. We’ll check lighting, isolate your hand, and compare it with the sign dataset."
    : "Write a message and we’ll preview image sequences from the included sign dataset.";
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

function formatSignTitle(label) {
  if (typeof label !== "string" || !label) return "Unknown sign";
  return label
    .replace(/[_-]+/g, " ")
    .replace(/\b\p{L}/gu, (letter) => letter.toLocaleUpperCase());
}

function getSignCatalog() {
  if (!signCatalogPromise) {
    signCatalogPromise = fetch("/api/signs")
      .then(async (response) => {
        const result = await response.json();
        if (!response.ok || !Array.isArray(result.signs)) {
          throw new Error(result.error || "The sign dataset could not be loaded.");
        }
        return result.signs;
      })
      .catch((error) => {
        signCatalogPromise = null;
        throw error;
      });
  }
  return signCatalogPromise;
}

function showDatasetFrame() {
  const active = signSequence[activeSignIndex];
  if (!active || !Array.isArray(active.sign.frames) || !active.sign.frames.length) return;
  const frame = active.sign.frames[activeFrameIndex];
  document.querySelector("#dataset-frame").src = frame;
  document.querySelector("#dataset-frame").alt = `${formatSignTitle(active.sign.title)} dataset sign frame`;
  document.querySelector("#dataset-frame-counter").textContent =
    `Frame ${activeFrameIndex + 1} / ${active.sign.frames.length}`;
  document.querySelector("#dataset-active-word").textContent =
    `${formatSignTitle(active.sign.title)} · “${active.word}”`;
  document.querySelector("#dataset-previous").disabled = activeSignIndex === 0;
  document.querySelector("#dataset-next").disabled = activeSignIndex >= signSequence.length - 1;
  document.querySelectorAll("#word-sequence .word-chip").forEach((chip) => {
    chip.classList.toggle(
      "is-active",
      Number(chip.dataset.sequenceIndex) === activeSignIndex,
    );
  });
}

function advanceDatasetFrame() {
  const active = signSequence[activeSignIndex];
  if (!active || !active.sign.frames.length || frameAdvancePending) return;
  const sequenceIndex = activeSignIndex;
  const nextFrameIndex = (activeFrameIndex + 1) % active.sign.frames.length;
  const frameUrl = active.sign.frames[nextFrameIndex];
  frameAdvancePending = true;
  preloadDatasetFrame(frameUrl)
    .then(() => {
      if (sequenceIndex !== activeSignIndex) return;
      activeFrameIndex = nextFrameIndex;
      showDatasetFrame();
    })
    .catch(() => {
      document.querySelector("#dataset-hint").textContent =
        "A sign frame could not be loaded from the local dataset.";
    })
    .finally(() => {
      frameAdvancePending = false;
    });
}

function preloadDatasetFrame(url) {
  const cached = preloadedDatasetFrames.get(url);
  if (cached) {
    preloadedDatasetFrames.delete(url);
    preloadedDatasetFrames.set(url, cached);
    return cached.promise;
  }

  const image = new Image();
  const entry = {
    image,
    promise: new Promise((resolve, reject) => {
      image.addEventListener("load", resolve, { once: true });
      image.addEventListener(
        "error",
        () => reject(new Error("Dataset frame failed to load.")),
        { once: true },
      );
    }),
  };
  preloadedDatasetFrames.set(url, entry);
  entry.promise.catch(() => {
    if (preloadedDatasetFrames.get(url) === entry) {
      preloadedDatasetFrames.delete(url);
    }
  });
  if (preloadedDatasetFrames.size > 8) {
    preloadedDatasetFrames.delete(preloadedDatasetFrames.keys().next().value);
  }
  image.src = url;
  if (image.complete) {
    if (image.naturalWidth) {
      return Promise.resolve();
    }
    return entry.promise;
  }
  return entry.promise;
}

function renderSignSequence(message) {
  const wordSequence = document.querySelector("#word-sequence");
  const player = document.querySelector("#dataset-player");
  const hint = document.querySelector("#dataset-hint");
  signSequence = [];
  wordSequence.replaceChildren();
  player.hidden = true;
  hint.textContent = "";
  activeSignIndex = 0;
  activeFrameIndex = 0;

  const words = message.match(/[\p{L}\p{N}’'-]+/gu) || [];
  if (!words.length) {
    document.querySelector("#text-result-status-text").textContent = "No words to preview";
    syncPlaybackTimer();
    return;
  }

  getSignCatalog()
    .then((catalog) => {
      const byLabel = new Map(catalog.map((sign) => [sign.label, sign]));
      const unsupported = [];
      const sequenceIndexes = words.map((word) => {
        const normalized = word
          .normalize("NFKC")
          .toLocaleLowerCase()
          .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
        const sign = byLabel.get(normalized);
        if (!sign || !Array.isArray(sign.frames) || !sign.frames.length) {
          unsupported.push(word);
          return -1;
        }
        signSequence.push({ word, sign });
        return signSequence.length - 1;
      });

      wordSequence.replaceChildren(
        ...words.map((word, index) => {
          const sequenceIndex = sequenceIndexes[index];
          const chip = document.createElement(sequenceIndex >= 0 ? "button" : "span");
          chip.className = `word-chip ${sequenceIndex >= 0 ? "is-supported" : "is-unsupported"}`;
          chip.textContent = word;
          if (sequenceIndex >= 0) {
            chip.type = "button";
            chip.dataset.sequenceIndex = String(sequenceIndex);
            chip.addEventListener("click", () => {
              activeSignIndex = sequenceIndex;
              activeFrameIndex = 0;
              showDatasetFrame();
            });
          }
          return chip;
        }),
      );

      const supportedCount = signSequence.length;
      document.querySelector("#text-result-status-text").textContent =
        `${supportedCount} dataset sign${supportedCount === 1 ? "" : "s"} matched · ${unsupported.length} unsupported word${unsupported.length === 1 ? "" : "s"}`;
      if (unsupported.length) {
        hint.textContent = `No dataset sign for: ${[...new Set(unsupported)].join(", ")}.`;
      }
      player.hidden = supportedCount === 0;
      if (supportedCount) {
        showDatasetFrame();
        document.querySelector("#dataset-play").textContent =
          isSequencePlaying ? "Pause sequence" : "Play sequence";
      } else if (!unsupported.length) {
        hint.textContent = "No matching sign sequences are available.";
      }
      syncPlaybackTimer();
    })
    .catch((error) => {
      document.querySelector("#text-result-status-text").textContent = "Sign dataset unavailable";
      hint.textContent = error.message || "Could not load the local sign dataset.";
      syncPlaybackTimer();
    });
}

function showResult(image, processingResult = null) {
  capturedImage = image || capturedImage;
  lastRecognitionText = "";
  document.querySelector("#result-type-label").textContent =
    direction === "sign-to-text" ? "SIGN TO TEXT" : "TEXT TO SIGN";
  document.querySelector("#sign-result").hidden = direction !== "sign-to-text";
  document.querySelector("#text-result").hidden = direction !== "text-to-sign";
  document.querySelector("#speak-result").hidden = direction !== "sign-to-text";
  document.querySelector("#result-intro").textContent =
    direction === "sign-to-text"
      ? "Your sign input is ready. Here’s the dataset recognition result."
      : "Your message is ready. Matching signs from the dataset are shown below.";
  document.querySelector("#result-disclaimer").textContent =
    direction === "sign-to-text"
      ? "Recognition is experimental and limited to the eight signs in this dataset; unclear signs are left unmatched."
      : "Playback uses still-image sequences from this dataset, not continuous sign-language video.";

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
    const recognition = processingResult.recognition || {};
    const title = formatSignTitle(recognition.label);
    if (!handDetected) {
      document.querySelector("#sign-result-title").textContent = "No hand detected yet.";
      document.querySelector("#sign-result-message").textContent =
        "We checked the frame but couldn’t find a hand. Try centering your hand in the camera and capture again.";
      document.querySelector("#sign-result-status-text").textContent = "No hand detected · try another frame";
    } else if (recognition.recognized) {
      lastRecognitionText = title;
      document.querySelector("#sign-result-title").textContent = `Possible match: ${title}`;
      document.querySelector("#sign-result-message").textContent =
        `This experimental dataset match scored ${recognition.match_score}%. Only signs included in the dataset can be recognized.`;
      document.querySelector("#sign-result-status-text").textContent =
        `${processingResult.background_removed ? "Hand isolated" : "Hand detected"} · ${recognition.match_score}% dataset match`;
    } else if (recognition.state === "loading" || recognition.state === "not_started") {
      document.querySelector("#sign-result-title").textContent = "Sign dataset is still preparing.";
      document.querySelector("#sign-result-message").textContent =
        "Hand detection and image preparation are complete. Wait a moment for the dataset to finish loading, then capture again.";
      document.querySelector("#sign-result-status-text").textContent = "Dataset features are loading";
    } else if (recognition.state === "error") {
      document.querySelector("#sign-result-title").textContent = "Dataset recognition is unavailable.";
      document.querySelector("#sign-result-message").textContent =
        "The camera frame was prepared, but the local sign dataset could not be loaded.";
      document.querySelector("#sign-result-status-text").textContent = "Recognition service unavailable";
    } else {
      document.querySelector("#sign-result-title").textContent = "No clear dataset match.";
      document.querySelector("#sign-result-message").textContent =
        "A hand was detected, but the sign did not confidently match an included dataset label. Try another frame or a clearer view.";
      document.querySelector("#sign-result-status-text").textContent =
        `${processingResult.background_removed ? "Hand isolated" : "Hand detected"} · no confident match`;
    }
  }
  if (direction === "text-to-sign") {
    const message = messageInput.value.trim();
    document.querySelector("#result-message-text").textContent = message;
    renderSignSequence(message);
  } else {
    signSequence = [];
    syncPlaybackTimer();
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

document.querySelector("#dataset-previous").addEventListener("click", () => {
  if (activeSignIndex === 0) return;
  activeSignIndex -= 1;
  activeFrameIndex = 0;
  showDatasetFrame();
});

document.querySelector("#dataset-next").addEventListener("click", () => {
  if (activeSignIndex >= signSequence.length - 1) return;
  activeSignIndex += 1;
  activeFrameIndex = 0;
  showDatasetFrame();
});

document.querySelector("#dataset-play").addEventListener("click", (event) => {
  isSequencePlaying = !isSequencePlaying;
  event.currentTarget.textContent = isSequencePlaying ? "Pause sequence" : "Play sequence";
  syncPlaybackTimer();
});

document.querySelector("#speak-result").addEventListener("click", () => {
  if (!("speechSynthesis" in window)) {
    cameraError.textContent = "Speech output isn’t supported by this browser.";
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(
    lastRecognitionText || "No confident sign match was found in the included dataset.",
  );
  window.speechSynthesis.speak(utterance);
});

document.querySelector("#start-again").addEventListener("click", () => {
  stopCamera();
  signSequence = [];
  isSequencePlaying = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  syncPlaybackTimer();
  direction = "";
  capturedImage = "";
  messageInput.value = "";
  document.querySelector("#character-count").textContent = "0";
  imageUpload.value = "";
  showPage(1);
});

window.addEventListener("pagehide", () => {
  stopCamera();
  if (playbackTimer) window.clearInterval(playbackTimer);
});
