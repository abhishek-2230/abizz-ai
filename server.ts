import { GoogleGenAI, Type } from '@google/genai';
import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '10mb' }));

  // Initialize Gemini AI client
  const getGeminiClient = () => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn('GEMINI_API_KEY environment variable is missing.');
    }
    return new GoogleGenAI({
      apiKey: apiKey || '',
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  };

  // Health Check API
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
  });

  // Agent API: Resume & Job Description ATS Analyzer
  app.post('/api/agents/analyze-resume', async (req, res) => {
    try {
      const { resumeText, jobDescriptionText, role, company } = req.body;
      const ai = getGeminiClient();

      const prompt = `You are the Resume Analyzer Agent and ATS Analyzer Agent.
Analyze the following resume against the job description for the role of "${role}" at "${company}".

Resume Text:
"""${resumeText || 'No resume provided'}"""

Job Description:
"""${jobDescriptionText || 'Standard ' + role + ' requirements at ' + company}"""

Provide a structured JSON response with the following schema:
- atsScore: number between 0 and 100
- weakSkills: array of strings (missing or weak skills relative to JD)
- strongSkills: array of strings (key strengths found in resume)
- keywordMatchPercentage: number between 0 and 100
- missingKeywords: array of strings
- formatScore: number between 0 and 100
- executiveSummary: string summarizing resume fit and key recommendations.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.6-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              atsScore: { type: Type.INTEGER },
              weakSkills: { type: Type.ARRAY, items: { type: Type.STRING } },
              strongSkills: { type: Type.ARRAY, items: { type: Type.STRING } },
              keywordMatchPercentage: { type: Type.INTEGER },
              missingKeywords: { type: Type.ARRAY, items: { type: Type.STRING } },
              formatScore: { type: Type.INTEGER },
              executiveSummary: { type: Type.STRING },
            },
            required: ['atsScore', 'weakSkills', 'strongSkills', 'keywordMatchPercentage', 'missingKeywords', 'executiveSummary'],
          },
        },
      });

      const data = JSON.parse(response.text || '{}');
      res.json({ success: true, data });
    } catch (error: unknown) {
      console.error('Error in analyze-resume agent:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to analyze resume',
        fallbackData: {
          atsScore: 82,
          weakSkills: ['System Design (Sharding)', 'Microservices Fault Tolerance', 'STAR Method Metrics'],
          strongSkills: ['React & Web Performance', 'TypeScript & Node.js', 'API Design & WebSockets'],
          keywordMatchPercentage: 80,
          missingKeywords: ['Kafka', 'Kubernetes', 'Microservices'],
          formatScore: 90,
          executiveSummary: 'Strong technical profile with excellent modern frontend and fullstack experience. Highlight quantifiable metrics in your project outcomes.',
        },
      });
    }
  });

  // Agent API: Question Generator
  app.post('/api/agents/generate-questions', async (req, res) => {
    try {
      const { role, company, type, weakSkills, count = 4 } = req.body;
      const ai = getGeminiClient();

      const prompt = `You are the Question Generator Agent, Technical Interview Agent, and HR Interview Agent.
Generate ${count} tailored interview questions for a candidate interviewing for the role of "${role}" at "${company}".
Interview Type: ${type || 'Technical & Behavioral'}.
Candidate Weak Areas to challenge: ${(weakSkills || []).join(', ')}.

Return a JSON array of questions, where each question object has:
- id: string
- type: string (Technical, Behavioral, HR, Coding, System Design, or Company Specific)
- difficulty: string (Easy, Medium, or Hard)
- category: string
- questionText: string
- expectedKeywords: array of strings
- hints: array of strings
- codeStarter: string (optional, if Coding type)
- codeSolution: string (optional, if Coding type)`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.6-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                type: { type: Type.STRING },
                difficulty: { type: Type.STRING },
                category: { type: Type.STRING },
                questionText: { type: Type.STRING },
                expectedKeywords: { type: Type.ARRAY, items: { type: Type.STRING } },
                hints: { type: Type.ARRAY, items: { type: Type.STRING } },
                codeStarter: { type: Type.STRING },
                codeSolution: { type: Type.STRING },
              },
              required: ['id', 'type', 'difficulty', 'category', 'questionText', 'expectedKeywords', 'hints'],
            },
          },
        },
      });

      const questions = JSON.parse(response.text || '[]');
      res.json({ success: true, questions });
    } catch (error: unknown) {
      console.error('Error generating questions:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to generate questions',
      });
    }
  });

  // Agent API: Answer Evaluation Agent
  app.post('/api/agents/evaluate-answer', async (req, res) => {
    try {
      const { questionId, questionText, userAnswer, expectedKeywords, role } = req.body;
      const ai = getGeminiClient();

      const prompt = `You are the Answer Evaluation Agent & Communication Analyzer Agent.
Evaluate this interview response for a candidate applying for "${role}".

Question:
"${questionText}"

User's Answer:
"${userAnswer}"

Expected Technical Keywords:
${JSON.stringify(expectedKeywords || [])}

Evaluate and return JSON with:
- correctnessScore: integer 0 - 100
- confidenceScore: integer 0 - 100
- communicationScore: integer 0 - 100
- starScore: integer 0 - 100 (evaluate if Situation, Task, Action, Result framework was effectively applied)
- technicalAccuracy: integer 0 - 100
- keyStrengths: array of strings
- improvementSuggestions: array of strings
- idealSampleAnswer: string (a top-tier exemplary response)`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.6-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              correctnessScore: { type: Type.INTEGER },
              confidenceScore: { type: Type.INTEGER },
              communicationScore: { type: Type.INTEGER },
              starScore: { type: Type.INTEGER },
              technicalAccuracy: { type: Type.INTEGER },
              keyStrengths: { type: Type.ARRAY, items: { type: Type.STRING } },
              improvementSuggestions: { type: Type.ARRAY, items: { type: Type.STRING } },
              idealSampleAnswer: { type: Type.STRING },
            },
            required: ['correctnessScore', 'confidenceScore', 'communicationScore', 'starScore', 'technicalAccuracy', 'keyStrengths', 'improvementSuggestions', 'idealSampleAnswer'],
          },
        },
      });

      const evaluation = JSON.parse(response.text || '{}');
      evaluation.questionId = questionId;
      res.json({ success: true, evaluation });
    } catch (error: unknown) {
      console.error('Error in answer evaluation:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to evaluate answer',
      });
    }
  });

  // Agent API: AI Career Coach & Mentor Chatbot
  app.post('/api/agents/ai-coach', async (req, res) => {
    try {
      const { prompt, userContext } = req.body;
      const ai = getGeminiClient();

      const systemInstruction = `You are the AI Mentor & Career Coach Agent at InterviewAI Pro.
Provide concise, actionable, highly professional career advice, interview tips, salary predictions, LinkedIn bullet optimization, or technical explanations.
Context of user: ${JSON.stringify(userContext || {})}`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.6-flash',
        contents: prompt,
        config: {
          systemInstruction,
        },
      });

      res.json({ success: true, reply: response.text });
    } catch (error: unknown) {
      console.error('Error in AI Coach agent:', error);
      res.status(500).json({
        success: false,
        reply: 'I am here to guide your interview preparation. Focus on mastering the STAR method and system design trade-offs!',
      });
    }
  });

  // Vite Middleware integration for Development & Production serving
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`InterviewAI Pro server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
