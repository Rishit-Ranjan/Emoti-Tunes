import React, { useRef, useEffect, useState, useCallback } from 'react';
const CameraView = ({ onCapture, onClose, onError }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [stream, setStream] = useState(null);
    const [zoom, setZoom] = useState(1);
    const [cameras, setCameras] = useState([]);
    const [currentCameraIndex, setCurrentCameraIndex] = useState(0);
    const [cameraCapabilities, setCameraCapabilities] = useState(null);
    const stopStream = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    }, [stream]);
    useEffect(() => {
        let isMounted = true;
        const initializeCamera = async () => {
            stopStream();
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                if (!isMounted)
                    return;
                setCameras(videoDevices);
                if (videoDevices.length === 0) {
                    onError("No camera found on this device.");
                    return;
                }
                const deviceId = videoDevices[currentCameraIndex]?.deviceId;
                const constraints = {
                    video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' },
                    audio: false
                };
                const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
                if (isMounted && videoRef.current) {
                    videoRef.current.srcObject = mediaStream;
                    setStream(mediaStream);
                    const videoTrack = mediaStream.getVideoTracks()[0];
                    const capabilities = videoTrack.getCapabilities();
                    setCameraCapabilities(capabilities);
                    if (capabilities.zoom) {
                        setZoom(capabilities.zoom.min);
                    }
                }
            }
            catch (err) {
                if (!isMounted)
                    return;
                console.error("Error accessing camera:", err);
                let errorMessage = "Could not access the camera. Please ensure you have granted permission.";
                if (err instanceof Error && err.name === 'NotAllowedError') {
                    errorMessage = "Camera permission was denied. Please grant permission in your browser settings to use this feature.";
                }
                onError(errorMessage);
            }
        };
        initializeCamera();
        return () => {
            isMounted = false;
            stopStream();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [currentCameraIndex, onError]);
    useEffect(() => {
        if (stream && cameraCapabilities?.zoom) {
            const videoTrack = stream.getVideoTracks()[0];
            if (typeof videoTrack.applyConstraints === 'function') {
                videoTrack.applyConstraints({ advanced: [{ zoom }] }).catch(e => console.error("Failed to apply zoom", e));
            }
        }
    }, [zoom, stream, cameraCapabilities]);
    const handleSwitchCamera = useCallback(() => {
        if (cameras.length > 1) {
            setCurrentCameraIndex(prevIndex => (prevIndex + 1) % cameras.length);
        }
    }, [cameras.length]);
    const handleCapture = useCallback(() => {
        if (videoRef.current && canvasRef.current) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            if (context) {
                // If using the front camera, the preview is mirrored via CSS.
                // We need to flip the canvas context to capture the un-mirrored image.
                if (cameras[currentCameraIndex]?.facing === 'user') {
                    context.translate(canvas.width, 0);
                    context.scale(-1, 1);
                }
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg').split(',')[1];
                onCapture(imageData);
            }
            else {
                onError("Could not process the image.");
            }
        }
    }, [onCapture, onError, cameras, currentCameraIndex]);
    const hasZoom = cameraCapabilities?.zoom && cameraCapabilities.zoom.max > cameraCapabilities.zoom.min;
    const canSwitchCamera = cameras.length > 1;
    return (<div className="w-full max-w-2xl mx-auto flex flex-col items-center">
            <div className="relative w-full aspect-video rounded-lg overflow-hidden border-2 border-slate-700 bg-black mb-6">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" style={{ transform: cameras[currentCameraIndex]?.facing === 'user' ? 'scaleX(-1)' : 'scaleX(1)' }}/>
                <canvas ref={canvasRef} className="hidden"/>

                 {(hasZoom || canSwitchCamera) && (<div className="absolute bottom-0 left-0 right-0 p-4 bg-black/40 backdrop-blur-sm flex items-center justify-center space-x-6">
                        {hasZoom && (<div className="flex items-center gap-3 w-full max-w-xs text-white" title="Zoom control">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7"/>
                                </svg>
                                <input type="range" min={cameraCapabilities.zoom.min} max={cameraCapabilities.zoom.max} step={cameraCapabilities.zoom.step || 1} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-blue-500" aria-label="Zoom slider"/>
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3h-6"/>
                                </svg>
                            </div>)}
                        {canSwitchCamera && (<button onClick={handleSwitchCamera} aria-label="Switch Camera" title="Switch Camera" className="p-3 rounded-full bg-slate-700/50 hover:bg-slate-600/70 text-white transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-blue-500">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5m-1.5-1.5l-4-4m0 0l-4 4m4-4V19"/>
                                </svg>
                            </button>)}
                    </div>)}
            </div>
            <div className="flex items-center space-x-4">
                <button onClick={onClose} className="px-6 py-3 border border-slate-600 text-base font-medium rounded-full text-slate-300 bg-slate-800 hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-slate-500">
                    Cancel
                </button>
                <button onClick={handleCapture} className="group inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-full text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-blue-500">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"/>
                    </svg>
                    Capture Emotion
                </button>
            </div>
        </div>);
};
export default CameraView;
