import React, { useState, useCallback, useEffect } from 'react';
import { EMOTIONS } from './constants';
import { generatePlaylist, detectEmotionFromImage, detectEmotionFromAudio } from './services/Service';
import EmotionSelector from './components/EmotionSelector';
import PlaylistDisplay from './components/PlaylistDisplay';
import Loader from './components/Loader';
import CameraView from './components/CameraView';
import AudioView from './components/AudioView';
const OfflineBanner = () => (<div className="fixed top-0 left-0 right-0 bg-slate-800 border-b border-yellow-500/30 text-center p-2 text-sm text-yellow-300 z-50 flex items-center justify-center shadow-lg" role="status">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M18.364 5.636a9 9 0 010 12.728m-2.828-2.828a5 5 0 010-7.072m-2.828-2.828a1 1 0 010-1.414A1 1 0 0111.314 3a1 1 0 011.414 0l.001.001.001.001a1 1 0 010 1.414m-2.828 2.828a1 1 0 010 1.414m-2.829-1.414a5 5 0 000 7.072m-2.828 2.828a9 9 0 0012.728 0M1 1l22 22"/>
        </svg>
        You are currently offline. Some features are unavailable.
    </div>);
const App = () => {
    const [currentEmotion, setCurrentEmotion] = useState(null);
    const [playlist, setPlaylist] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');
    const [error, setError] = useState(null);
    const [isCameraOpen, setIsCameraOpen] = useState(false);
    const [isMicOpen, setIsMicOpen] = useState(false);
    const [isOffline, setIsOffline] = useState(!navigator.onLine);
    useEffect(() => {
        const handleOnline = () => {
            setIsOffline(false);
            setError(null);
        };
        const handleOffline = () => setIsOffline(true);
        window.addEventListener('online', handleOnline);
        window.addEventListener('offline', handleOffline);
        return () => {
            window.removeEventListener('online', handleOnline);
            window.removeEventListener('offline', handleOffline);
        };
    }, []);
    const handleEmotionSelect = useCallback(async (emotion) => {
        if (isOffline) {
            setError("You're offline. Please connect to the internet to generate a playlist.");
            return;
        }
        setCurrentEmotion(emotion);
        setLoadingMessage(`Finding the perfect ${emotion.name.toLowerCase()} tracks for you...`);
        setIsLoading(true);
        setError(null);
        setPlaylist([]);
        try {
            const newPlaylist = await generatePlaylist(emotion.name);
            setPlaylist(newPlaylist);
        }
        catch (err) {
            console.error("Playlist generation error:", err);
            setError(err instanceof Error ? err.message : 'An unknown error occurred while generating your playlist.');
        }
        finally {
            setIsLoading(false);
        }
    }, [isOffline]);
    const handleReset = useCallback(() => {
        setCurrentEmotion(null);
        setPlaylist([]);
        setError(null);
        setIsLoading(false);
        setIsCameraOpen(false);
        setIsMicOpen(false);
    }, []);
    const handleOpenCamera = () => {
        setError(null);
        setIsCameraOpen(true);
    };
    const handleOpenMic = () => {
        setError(null);
        setIsMicOpen(true);
    };
    const handleCapture = useCallback(async (imageData) => {
        if (isOffline) {
            setError("You're offline. Please connect to the internet to analyze images.");
            setIsCameraOpen(false);
            return;
        }
        setIsCameraOpen(false);
        setLoadingMessage('Analyzing your emotion...');
        setIsLoading(true);
        setError(null);
        try {
            const detectedEmotionName = await detectEmotionFromImage(imageData);
            const matchedEmotion = EMOTIONS.find(e => e.name.toLowerCase() === detectedEmotionName.toLowerCase());
            if (matchedEmotion) {
                await handleEmotionSelect(matchedEmotion);
            }
            else {
                throw new Error(`Could not find a matching mood for "${detectedEmotionName}". Please try again.`);
            }
        }
        catch (err) {
            console.error("Emotion detection (image) error:", err);
            setError(err instanceof Error ? err.message : 'An unknown error occurred while analyzing your image.');
            setIsLoading(false);
        }
    }, [handleEmotionSelect, isOffline]);
    const handleAudioCapture = useCallback(async ({ audioData, mimeType }) => {
        if (isOffline) {
            setError("You're offline. Please connect to the internet to analyze audio.");
            setIsMicOpen(false);
            return;
        }
        setIsMicOpen(false);
        setLoadingMessage('Analyzing your voice...');
        setIsLoading(true);
        setError(null);
        try {
            const detectedEmotionName = await detectEmotionFromAudio(audioData, mimeType);
            const matchedEmotion = EMOTIONS.find(e => e.name.toLowerCase() === detectedEmotionName.toLowerCase());
            if (matchedEmotion) {
                await handleEmotionSelect(matchedEmotion);
            }
            else {
                throw new Error(`Could not find a matching mood for "${detectedEmotionName}". Please try again.`);
            }
        }
        catch (err) {
            console.error("Emotion detection (audio) error:", err);
            setError(err instanceof Error ? err.message : 'An unknown error occurred while analyzing your voice.');
            setIsLoading(false);
        }
    }, [handleEmotionSelect, isOffline]);
    const handleCameraError = useCallback((errorMessage) => {
        setError(errorMessage);
        setIsCameraOpen(false);
    }, []);
    const handleAudioError = useCallback((errorMessage) => {
        setError(errorMessage);
        setIsMicOpen(false);
    }, []);
    const renderContent = () => {
        if (isLoading) {
            return <Loader message={loadingMessage}/>;
        }
        if (error) {
            return (<div className="text-center bg-slate-800/50 border border-red-500/30 rounded-2xl p-8 max-w-lg mx-auto shadow-lg shadow-red-900/20" role="alert">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-red-400 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
          </svg>
          <h2 className="text-2xl font-bold text-slate-100 mb-2">An Error Occurred</h2>
          <p className="text-red-400/90 text-md mb-6">{error}</p>
          <button onClick={handleReset} className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-blue-500">
            Try Again
          </button>
        </div>);
        }
        if (isCameraOpen) {
            return <CameraView onCapture={handleCapture} onClose={handleReset} onError={handleCameraError}/>;
        }
        if (isMicOpen) {
            return <AudioView onCapture={handleAudioCapture} onClose={handleReset} onError={handleAudioError}/>;
        }
        if (playlist.length > 0 && currentEmotion) {
            return <PlaylistDisplay playlist={playlist} emotion={currentEmotion} onReset={handleReset}/>;
        }
        return <EmotionSelector emotions={EMOTIONS} onSelect={handleEmotionSelect} onOpenCamera={handleOpenCamera} onOpenMic={handleOpenMic} isOffline={isOffline}/>;
    };
    return (<div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center p-4 selection:bg-blue-500/30">
        {isOffline && <OfflineBanner />}
        <main className={`w-full max-w-4xl mx-auto transition-all duration-500 ${isOffline ? 'pt-10' : ''}`}>
            {renderContent()}
        </main>
        
    </div>);
};
export default App;
