import React, { useState } from 'react';
import { SpotifyIcon } from './icons/EmotionIcons';
const PlaylistDisplay = ({ playlist, emotion, onReset }) => {
    const [currentSong, setCurrentSong] = useState(playlist.length > 0 ? playlist[0] : null);
    const [isPlaying, setIsPlaying] = useState(false);
    const handlePlayPause = () => {
        if (currentSong) {
            setIsPlaying(!isPlaying);
        }
    };
    const handleSelectSong = (song) => {
        setCurrentSong(song);
        setIsPlaying(true);
    };
    return (<div className={`w-full max-w-2xl mx-auto rounded-2xl border border-slate-700 bg-slate-800/50 p-8 shadow-2xl shadow-slate-950/50 bg-gradient-to-b ${emotion.gradient}`}>
      <div className="flex items-center mb-6">
        <emotion.icon className={`w-12 h-12 mr-4 ${emotion.color}`}/>
        <div>
          <h2 className="text-3xl font-bold text-slate-100">Your {emotion.name} Playlist</h2>
          <p className="text-slate-400">Here are some tracks to match your mood.</p>
        </div>
      </div>

      <ul className="space-y-3">
        {playlist.map((song, index) => {
            const youtubeSearchUrl = `https://www.youtube.com/results?search_query=${encodeURIComponent(`${song.title} ${song.artist}`)}`;
            const spotifySearchUrl = `https://open.spotify.com/search/${encodeURIComponent(`${song.title} ${song.artist}`)}`;
            const isCurrentlyPlaying = currentSong?.title === song.title && currentSong?.artist === song.artist;
            return (<li key={index} onClick={() => handleSelectSong(song)} className={`group flex items-center justify-between p-3 rounded-lg transition-all duration-200 cursor-pointer ${isCurrentlyPlaying ? 'bg-blue-600/20' : 'bg-slate-900/40 hover:bg-slate-900/70'}`} aria-current={isCurrentlyPlaying ? 'true' : 'false'}>
              <div className="flex items-center min-w-0">
                <span className={`flex-shrink-0 font-mono text-sm w-8 ${isCurrentlyPlaying ? 'text-blue-400' : 'text-slate-500'}`}>{index + 1}.</span>
                <div className="truncate">
                  <h3 className={`font-semibold truncate ${isCurrentlyPlaying ? 'text-blue-300' : 'text-slate-200'}`}>{song.title}</h3>
                  <p className="text-sm text-slate-400 truncate">{song.artist}</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 ml-4">
                <a href={youtubeSearchUrl} target="_blank" rel="noopener noreferrer" aria-label={`Search ${song.title} on YouTube`} className="text-slate-300 hover:text-white">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                </a>
                <a href={spotifySearchUrl} target="_blank" rel="noopener noreferrer" aria-label={`Search ${song.title} on Spotify`} className="text-slate-300 hover:text-white">
                  <SpotifyIcon className="h-6 w-6"/>
                </a>
              </div>
            </li>);
        })}
      </ul>
      
      {/* Player Controls */}
      <div className="mt-8 pt-6 border-t border-slate-700/50">
        <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 min-w-0">
                <button onClick={handlePlayPause} className="p-3 flex-shrink-0 rounded-full bg-slate-700/50 hover:bg-slate-600/70 text-white transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-blue-500" aria-label={isPlaying ? 'Pause' : 'Play'}>
                    {isPlaying ? (<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>) : (<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>)}
                </button>
                <div className="text-left truncate">
                    <p className="font-semibold text-slate-200 truncate">{currentSong?.title || 'Select a song'}</p>
                    <p className="text-sm text-slate-400 truncate">{currentSong?.artist || '...'}</p>
                </div>
            </div>
        </div>
      </div>
      
      <button onClick={onReset} className="w-full mt-8 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 000 2h6a1 1 0 100-2H7z" clipRule="evenodd"/>
        </svg>
        <span>Change My Mood</span>
      </button>
    </div>);
};
export default PlaylistDisplay;
