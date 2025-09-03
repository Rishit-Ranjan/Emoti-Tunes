import React, { useState, useRef, useEffect, useCallback } from 'react';
const AudioView = ({ onCapture, onClose, onError }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const timerIntervalRef = useRef(null);
    const streamRef = useRef(null);
    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
        if (timerIntervalRef.current) {
            clearInterval(timerIntervalRef.current);
        }
        setIsRecording(false);
        setRecordingTime(0);
    }, []);
    const cleanup = useCallback(() => {
        stopRecording();
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
    }, [stopRecording]);
    useEffect(() => {
        return () => {
            cleanup();
        };
    }, [cleanup]);
    const handleStartRecording = async () => {
        cleanup(); // Clean up any previous stream
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            streamRef.current = stream;
            audioChunksRef.current = [];
            const options = { mimeType: 'audio/webm;codecs=opus' };
            let recorder;
            try {
                recorder = new MediaRecorder(stream, options);
            }
            catch (e) {
                console.warn(`'audio/webm;codecs=opus' not supported, falling back.`);
                recorder = new MediaRecorder(stream);
            }
            mediaRecorderRef.current = recorder;
            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };
            recorder.onstop = () => {
                const mimeType = recorder.mimeType;
                const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64Data = reader.result;
                    const audioData = base64Data.split(',')[1];
                    onCapture({ audioData, mimeType });
                };
                reader.onerror = () => {
                    onError("Failed to read audio data.");
                };
                cleanup();
            };
            recorder.start();
            setIsRecording(true);
            setRecordingTime(0);
            timerIntervalRef.current = window.setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);
        }
        catch (err) {
            console.error("Error accessing microphone:", err);
            let errorMessage = "Could not access the microphone. Please ensure you have granted permission.";
            if (err instanceof Error && err.name === 'NotAllowedError') {
                errorMessage = "Microphone permission was denied. Please grant permission in your browser settings to use this feature.";
            }
            onError(errorMessage);
        }
    };
    const handleStopRecording = () => {
        stopRecording();
    };
    return (<div className="w-full max-w-md mx-auto flex flex-col items-center text-center p-8 rounded-2xl border border-slate-700 bg-slate-800/50 shadow-lg">
            <h2 className="text-3xl font-bold text-slate-100 mb-2">Voice Analysis</h2>
            <p className="text-slate-400 mb-8">
                {isRecording ? "Speak clearly for a few seconds." : "Record your voice to detect your current mood."}
            </p>

            <div className="relative w-48 h-48 flex items-center justify-center mb-8">
                {isRecording && (<div className="absolute inset-0 bg-green-500/20 rounded-full animate-pulse"></div>)}
                <button onClick={isRecording ? handleStopRecording : handleStartRecording} className={`z-10 w-36 h-36 rounded-full flex flex-col items-center justify-center text-white transition-all duration-300 ${isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} focus:outline-none focus:ring-4 focus:ring-offset-2 focus:ring-offset-slate-800 ${isRecording ? 'focus:ring-red-500' : 'focus:ring-green-500'}`} aria-label={isRecording ? "Stop recording and analyze" : "Start recording"}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        {isRecording ? (<path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 017.5 5.25h9a2.25 2.25 0 012.25 2.25v9a2.25 2.25 0 01-2.25-2.25h-9a2.25 2.25 0 01-2.25-2.25v-9z"/>) : (<path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>)}
                    </svg>
                    <span className="font-semibold text-lg">
                        {isRecording ? "Stop" : "Record"}
                    </span>
                </button>
            </div>
            
             {isRecording ? (<div className="text-xl font-mono text-slate-300 mb-8">
                    0:{recordingTime.toString().padStart(2, '0')}
                </div>) : (<button onClick={onClose} className="px-6 py-3 border border-slate-600 text-base font-medium rounded-full text-slate-300 bg-slate-800 hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-slate-500">
                    Skip
                </button>)}
        </div>);
};
export default AudioView;
