import React from 'react';
const EmotionSelector = ({ emotions, onSelect, onOpenCamera, onOpenMic, isOffline }) => {
    const offlineTooltip = isOffline ? 'This feature requires an internet connection.' : '';
    return (<div className="text-center">
      <h1 className="text-5xl font-extrabold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
        Welcome to EmotiTunes
      </h1>
      <p className="text-xl text-slate-300 mb-8">How are you feeling right now?</p>
      
      <div className="mb-10 flex flex-col sm:flex-row items-center justify-center gap-4">
          <button onClick={onOpenCamera} disabled={isOffline} title={offlineTooltip} className="group w-full sm:w-auto inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-full text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-purple-500 transition-all duration-300 transform hover:scale-105 disabled:bg-slate-600 disabled:opacity-70 disabled:cursor-not-allowed disabled:transform-none disabled:hover:bg-slate-600">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"/>
              </svg>
              Detect with Camera
          </button>
           <button onClick={onOpenMic} disabled={isOffline} title={offlineTooltip} className="group w-full sm:w-auto inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-full text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-green-500 transition-all duration-300 transform hover:scale-105 disabled:bg-slate-600 disabled:opacity-70 disabled:cursor-not-allowed disabled:transform-none disabled:hover:bg-slate-600">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
              </svg>
              Detect with Microphone
          </button>
      </div>

      <div className="relative mb-8">
        <div className="absolute inset-0 flex items-center" aria-hidden="true">
          <div className="w-full border-t border-slate-700"/>
        </div>
        <div className="relative flex justify-center">
          <span className="bg-slate-900 px-2 text-sm text-slate-400">OR SELECT MANUALLY</span>
        </div>
      </div>


      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {emotions.map((emotion) => (<button key={emotion.name} onClick={() => onSelect(emotion)} disabled={isOffline} title={offlineTooltip} className={`group p-6 rounded-xl border border-slate-700 bg-slate-800/50 hover:bg-slate-800/80 hover:border-slate-600 transform hover:-translate-y-1 transition-all duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-blue-500 text-left disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:hover:bg-slate-800/50`}>
            <div className="flex items-center space-x-4">
              <emotion.icon className={`w-12 h-12 ${emotion.color} transition-colors`}/>
              <div>
                <h3 className="text-xl font-bold text-slate-100">{emotion.name}</h3>
                <p className="text-slate-400">{emotion.description}</p>
              </div>
            </div>
          </button>))}
      </div>
    </div>);
};
export default EmotionSelector;
