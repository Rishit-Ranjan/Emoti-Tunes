import { HappyIcon, SadIcon, AngryIcon, JoyfulIcon, MelancholyIcon, EnergeticIcon } from './components/icons/EmotionIcons';
export const EMOTIONS = [
    {
        name: 'Joy',
        description: 'Feeling upbeat and happy.',
        icon: HappyIcon,
        color: 'text-yellow-300',
        gradient: 'from-yellow-500/10 to-slate-900'
    },
    {
        name: 'Sadness',
        description: 'For quiet, reflective moments.',
        icon: SadIcon,
        color: 'text-blue-300',
        gradient: 'from-blue-500/10 to-slate-900'
    },
    {
        name: 'Anger',
        description: 'To channel your frustration.',
        icon: AngryIcon,
        color: 'text-red-400',
        gradient: 'from-red-500/10 to-slate-900'
    },
    {
        name: 'Excitement',
        description: 'High-energy and thrilling.',
        icon: JoyfulIcon,
        color: 'text-orange-400',
        gradient: 'from-orange-500/10 to-slate-900'
    },
    {
        name: 'Melancholy',
        description: 'Bittersweet and thoughtful.',
        icon: MelancholyIcon,
        color: 'text-indigo-300',
        gradient: 'from-indigo-500/10 to-slate-900'
    },
    {
        name: 'Peaceful',
        description: 'Calm, serene, and relaxing.',
        icon: EnergeticIcon,
        color: 'text-green-300',
        gradient: 'from-green-500/10 to-slate-900'
    },
    {
        name: 'Joy-Anger',
        description: 'Triumphant, righteous fury.',
        icon: AngryIcon,
        color: 'text-orange-400',
        gradient: 'from-orange-500/10 to-slate-900'
    },
    {
        name: 'Joy-Surprise',
        description: 'Delightful astonishment.',
        icon: JoyfulIcon,
        color: 'text-pink-400',
        gradient: 'from-pink-500/10 to-slate-900'
    },
    {
        name: 'Joy-Excitement',
        description: 'Bursting with positive energy.',
        icon: HappyIcon,
        color: 'text-lime-300',
        gradient: 'from-lime-500/10 to-slate-900'
    },
    {
        name: 'Sad-Anger',
        description: 'Frustrated and feeling down.',
        icon: SadIcon,
        color: 'text-purple-400',
        gradient: 'from-purple-500/10 to-slate-900'
    }
];
