import { GoogleGenAI, Type } from "@google/genai";
if (!import.meta.env.VITE_API_KEY) {
  throw new Error("API key environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: import.meta.env.VITE_API_KEY });

const checkOnlineStatus = () => {
    if (!navigator.onLine) {
        throw new Error("You are currently offline. An internet connection is required.");
    }
};
const playlistResponseSchema = {
    type: Type.OBJECT,
    properties: {
        songs: {
            type: Type.ARRAY,
            description: "A list of songs for the playlist.",
            items: {
                type: Type.OBJECT,
                properties: {
                    title: {
                        type: Type.STRING,
                        description: "The title of the song."
                    },
                    artist: {
                        type: Type.STRING,
                        description: "The name of the artist or band."
                    }
                },
                required: ["title", "artist"]
            }
        }
    },
    required: ["songs"]
};
const emotionResponseSchema = {
    type: Type.OBJECT,
    properties: {
        emotion: {
            type: Type.STRING,
            description: "The detected emotion from the list provided.",
            enum: ['Joy', 'Sadness', 'Anger', 'Excitement', 'Melancholy', 'Peaceful', 'Joy-Anger', 'Joy-Surprise', 'Joy-Excitement', 'Sad-Anger']
        }
    },
    required: ["emotion"]
};
/**
 * Extracts a JSON string from a markdown code block if present.
 * @param responseText The raw text from the AI response.
 * @returns A clean JSON string.
 */
const cleanJsonString = (responseText) => {
    let jsonText = responseText.trim();
    const jsonMatch = jsonText.match(/```json\n([\s\S]*?)\n```/);
    if (jsonMatch && jsonMatch[1]) {
        jsonText = jsonMatch[1];
    }
    return jsonText;
};
const handleApiError = (error, context) => {
    console.error(`Error during ${context}:`, error);
    if (error instanceof Error) {
        if (error.message.toLowerCase().includes("api key")) {
            return new Error("The application's API key is invalid or missing. Please contact support.");
        }
        // Passthrough for specific, user-friendly errors thrown intentionally
        if (error.message.includes("offline") || error.message.startsWith("The AI")) {
            return error;
        }
    }
    return new Error(`Could not connect to the AI service for ${context}. Please check your internet connection and try again.`);
};
export const generatePlaylist = async (emotion) => {
    checkOnlineStatus();
    try {
        const prompt = `Generate a playlist of 10 songs that perfectly capture the feeling of '${emotion}'. For each song, provide the title and the artist's name. Focus on a diverse mix of genres and artists suitable for this mood.`;
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: playlistResponseSchema,
                temperature: 0.8,
            }
        });
        const jsonText = cleanJsonString(response.text);
        let parsed;
        try {
            parsed = JSON.parse(jsonText);
        }
        catch (parseError) {
            console.error("Failed to parse playlist JSON:", jsonText, parseError);
            throw new Error("The AI returned a response, but it was in an unexpected format.");
        }
        if (parsed && Array.isArray(parsed.songs)) {
            return parsed.songs;
        }
        else {
            console.error("Unexpected JSON structure for playlist:", parsed);
            throw new Error("The AI returned a playlist, but its structure was not what we expected.");
        }
    }
    catch (error) {
        throw handleApiError(error, "playlist generation");
    }
};
export const detectEmotionFromImage = async (base64ImageData) => {
    checkOnlineStatus();
    try {
        const imagePart = {
            inlineData: {
                mimeType: 'image/jpeg',
                data: base64ImageData,
            },
        };
        const textPart = {
            text: `Analyze the user's facial expression in this image and identify their primary emotion. Choose the most fitting emotion from the following list: Joy, Sadness, Anger, Excitement, Melancholy, Peaceful, Joy-Anger, Joy-Surprise, Joy-Excitement, Sad-Anger.`
        };
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: { parts: [imagePart, textPart] },
            config: {
                responseMimeType: "application/json",
                responseSchema: emotionResponseSchema,
            }
        });
        const jsonText = cleanJsonString(response.text);
        let parsed;
        try {
            parsed = JSON.parse(jsonText);
        }
        catch (parseError) {
            console.error("Failed to parse emotion JSON from image analysis:", jsonText, parseError);
            throw new Error("The AI analyzed the image, but its response was in an unexpected format.");
        }
        if (parsed && typeof parsed.emotion === 'string') {
            return parsed.emotion;
        }
        else {
            console.error("Unexpected JSON structure for emotion detection:", parsed);
            throw new Error("The AI's analysis of the image was inconclusive or in an unexpected format.");
        }
    }
    catch (error) {
        throw handleApiError(error, "image analysis");
    }
};
export const detectEmotionFromAudio = async (base64AudioData, mimeType) => {
    checkOnlineStatus();
    try {
        const audioPart = {
            inlineData: {
                mimeType,
                data: base64AudioData,
            },
        };
        const textPart = {
            text: `Analyze the user's tone of voice in this audio clip and identify their primary emotion. Choose the most fitting emotion from the following list: Joy, Sadness, Anger, Excitement, Melancholy, Peaceful, Joy-Anger, Joy-Surprise, Joy-Excitement, Sad-Anger.`
        };
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: { parts: [audioPart, textPart] },
            config: {
                responseMimeType: "application/json",
                responseSchema: emotionResponseSchema,
            }
        });
        const jsonText = cleanJsonString(response.text);
        let parsed;
        try {
            parsed = JSON.parse(jsonText);
        }
        catch (parseError) {
            console.error("Failed to parse emotion JSON from audio analysis:", jsonText, parseError);
            throw new Error("The AI analyzed your voice, but its response was in an unexpected format.");
        }
        if (parsed && typeof parsed.emotion === 'string') {
            return parsed.emotion;
        }
        else {
            console.error("Unexpected JSON structure for emotion detection:", parsed);
            throw new Error("The AI's analysis of your voice was inconclusive or in an unexpected format.");
        }
    }
    catch (error) {
        throw handleApiError(error, "audio analysis");
    }
};
