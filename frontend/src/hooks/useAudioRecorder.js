/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useState, useRef, useCallback, useEffect } from "react"
import { getMergedSamples, resample, blobToBase64 } from "../utils/audioHelpers"

// Mic capture hook: records raw Float32 PCM via Web Audio, resamples to
// 16 kHz mono, and returns it base64-encoded, the payload format expected by
// POST /api/stt (backend/server.py).
//
// Low-latency version: the microphone and audio graph are opened ONCE (on the
// first key press) and kept open. Pressing the record key only flips a flag,
// so capture starts instantly instead of waiting 1-2 s for the OS to open the
// mic. A short rolling pre-roll buffer is kept while idle and prepended to
// each recording, so words spoken right as the key goes down are not clipped.
const PRE_ROLL_SECONDS = 0.5

export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [micError, setMicError] = useState(null)

  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const sourceRef = useRef(null)
  const scriptProcessorRef = useRef(null)
  const streamRef = useRef(null)
  const recordedSamplesRef = useRef([])
  const preRollRef = useRef([])
  const preRollSamplesRef = useRef(0)
  const recordingRef = useRef(false)
  const openingRef = useRef(null)

  const teardown = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect()
      scriptProcessorRef.current.onaudioprocess = null
      scriptProcessorRef.current = null
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect()
      sourceRef.current = null
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect()
      analyserRef.current = null
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close()
    }
    audioContextRef.current = null
  }, [])

  useEffect(() => teardown, [teardown])

  // Open the mic and audio graph once; reuse on every later press.
  const ensureMicOpen = useCallback(async () => {
    const ctx = audioContextRef.current
    const live =
      ctx &&
      ctx.state !== "closed" &&
      streamRef.current &&
      streamRef.current.getAudioTracks().some((t) => t.readyState === "live")
    if (live) {
      if (ctx.state === "suspended") await ctx.resume()
      return
    }
    if (openingRef.current) return openingRef.current

    openingRef.current = (async () => {
      teardown()
      // echoCancellation is OFF on purpose: with the mic held open, Chrome on
      // macOS switches to the system voice-processing mode, which ducks or
      // mutes other audio on the Mac, including our own TTS playback.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream

      const AudioContext = window.AudioContext || window.webkitAudioContext
      const context = new AudioContext()
      audioContextRef.current = context
      if (context.state === "suspended") await context.resume()

      const source = context.createMediaStreamSource(stream)
      sourceRef.current = source

      // Small FFT: the analyser only feeds the low-res bar visualizer.
      const analyser = context.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      const maxPreRoll = Math.round(PRE_ROLL_SECONDS * context.sampleRate)
      const scriptProcessor = context.createScriptProcessor(2048, 1, 1)
      scriptProcessorRef.current = scriptProcessor
      scriptProcessor.onaudioprocess = (e) => {
        const chunk = new Float32Array(e.inputBuffer.getChannelData(0))
        if (recordingRef.current) {
          recordedSamplesRef.current.push(chunk)
          return
        }
        // Idle: keep only the most recent PRE_ROLL_SECONDS of audio.
        preRollRef.current.push(chunk)
        preRollSamplesRef.current += chunk.length
        while (
          preRollRef.current.length > 1 &&
          preRollSamplesRef.current - preRollRef.current[0].length >= maxPreRoll
        ) {
          preRollSamplesRef.current -= preRollRef.current.shift().length
        }
      }
      source.connect(scriptProcessor)
      scriptProcessor.connect(context.destination)
    })()

    try {
      await openingRef.current
    } finally {
      openingRef.current = null
    }
  }, [teardown])

  // Open the mic as soon as the page loads, so even the first press is
  // instant. Browsers keep the AudioContext suspended until a key press;
  // startRecording resumes it.
  useEffect(() => {
    ensureMicOpen().catch((err) => console.warn("Mic warm-up deferred:", err))
  }, [ensureMicOpen])

  const startRecording = useCallback(async () => {
    setMicError(null)
    try {
      await ensureMicOpen()
      // Seed the recording with the pre-roll so the first syllable is kept.
      recordedSamplesRef.current = preRollRef.current.slice()
      preRollRef.current = []
      preRollSamplesRef.current = 0
      recordingRef.current = true
      setIsRecording(true)
      return true
    } catch (err) {
      console.error("Error accessing microphone:", err)
      const msg =
        err.message || "Microphone access failed (HTTPS required for remote devices)"
      setMicError(msg)
      teardown()
      return false
    }
  }, [ensureMicOpen, teardown])

  const stopRecording = useCallback(async () => {
    if (!recordingRef.current) return

    recordingRef.current = false
    setIsRecording(false)

    // The mic stays open for the next press; only collect what was captured.
    const actualSampleRate = audioContextRef.current?.sampleRate || 16000
    const chunks = recordedSamplesRef.current
    recordedSamplesRef.current = []

    if (chunks.length === 0) {
      console.warn("No audio samples recorded")
      return null
    }

    const mergedSamples = getMergedSamples(chunks)
    const targetSampleRate = 16000 // Moonshine STT expects 16 kHz mono
    const resampledSamples = resample(mergedSamples, actualSampleRate, targetSampleRate)
    const rawBlob = new Blob([resampledSamples.buffer], {
      type: "application/octet-stream",
    })

    try {
      const base64Data = await blobToBase64(rawBlob)
      return { rawBlob, base64Data }
    } catch (err) {
      console.error("Base64 encoding failed:", err)
      return null
    }
  }, [])

  return {
    isRecording,
    startRecording,
    stopRecording,
    warmUp: ensureMicOpen,
    analyser: analyserRef.current,
    micError,
    setMicError,
  }
}
