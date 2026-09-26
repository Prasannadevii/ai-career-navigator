/**
 * AI Service Abstraction Layer
 *
 * Supported providers:
 * - Anthropic
 * - Google Gemini
 * - OpenAI
 * - Groq
 *
 * Provider is configured using:
 * AI_PROVIDER=groq
 */

const AI_PROVIDER =
  process.env.AI_PROVIDER || 'anthropic';


// ============================================================
// COMMON HELPERS
// ============================================================

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


// ============================================================
// ANTHROPIC
// ============================================================

async function callAnthropic(
  systemPrompt,
  userMessage,
  maxTokens = 1500
) {
  const apiKey =
    process.env.ANTHROPIC_API_KEY;

  if (!apiKey) {
    throw new Error(
      'ANTHROPIC_API_KEY not configured'
    );
  }

  const model =
    process.env.ANTHROPIC_MODEL ||
    'claude-haiku-4-5-20251001';

  const response = await fetch(
    'https://api.anthropic.com/v1/messages',
    {
      method: 'POST',

      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },

      body: JSON.stringify({
        model,
        max_tokens: maxTokens,
        system: systemPrompt,

        messages: [
          {
            role: 'user',
            content: userMessage
          }
        ]
      })
    }
  );

  if (!response.ok) {
    const err =
      await response.text();

    throw new Error(
      `Anthropic API error ${response.status}: ${err}`
    );
  }

  const data =
    await response.json();

  if (
    !data.content ||
    !data.content[0] ||
    !data.content[0].text
  ) {
    throw new Error(
      'Anthropic returned an empty response'
    );
  }

  return data.content[0].text;
}


// ============================================================
// GEMINI RETRY HELPER
// ============================================================

async function fetchGeminiWithRetry(
  url,
  options,
  label = 'Gemini'
) {
  const maxAttempts = 3;

  const retryDelays = [
    3000,
    7000,
    12000
  ];

  for (
    let attempt = 1;
    attempt <= maxAttempts;
    attempt++
  ) {
    try {
      const response =
        await fetch(url, options);

      if (response.ok) {
        return response;
      }

      const errText =
        await response.text();

      const retryable =
        response.status === 429 ||
        response.status === 502 ||
        response.status === 503 ||
        response.status === 504;

      if (
        !retryable ||
        attempt === maxAttempts
      ) {
        throw new Error(
          `${label} API error ${response.status}: ${errText}`
        );
      }

      const retryAfterHeader =
        response.headers.get(
          'retry-after'
        );

      const retryAfterSeconds =
        Number(retryAfterHeader);

      const delay =
        Number.isFinite(
          retryAfterSeconds
        ) &&
        retryAfterSeconds > 0
          ? Math.min(
              retryAfterSeconds * 1000,
              30000
            )
          : retryDelays[
              attempt - 1
            ];

      console.warn(
        `${label} returned ${response.status}. ` +
        `Retrying in ${Math.round(
          delay / 1000
        )}s...`
      );

      await sleep(delay);

    } catch (err) {

      if (
        attempt === maxAttempts
      ) {
        throw err;
      }

      await sleep(
        retryDelays[
          attempt - 1
        ]
      );
    }
  }

  throw new Error(
    `${label} request failed after retries`
  );
}


// ============================================================
// GEMINI
// ============================================================

async function callGemini(
  systemPrompt,
  userMessage,
  maxTokens = 1500,
  jsonResponse = false
) {
  const apiKey =
    process.env.GEMINI_API_KEY;

  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY not configured'
    );
  }

  const model =
    process.env.GEMINI_MODEL ||
    'gemini-3.8-flash';

  const generationConfig = {
    maxOutputTokens: maxTokens
  };

  if (jsonResponse) {
    generationConfig.responseMimeType =
      'application/json';
  }

  const response =
    await fetchGeminiWithRetry(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
      {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey
        },

        body: JSON.stringify({
          systemInstruction: {
            parts: [
              {
                text: systemPrompt
              }
            ]
          },

          contents: [
            {
              role: 'user',

              parts: [
                {
                  text: userMessage
                }
              ]
            }
          ],

          generationConfig
        })
      },
      'Gemini API'
    );

  const data =
    await response.json();

  if (
    !data.candidates ||
    !data.candidates[0]
  ) {
    const reason =
      data.promptFeedback ||
      data.error ||
      'No candidate returned';

    throw new Error(
      `Gemini returned no candidates: ${JSON.stringify(reason)}`
    );
  }

  const candidate =
    data.candidates[0];

  if (
    !candidate.content ||
    !candidate.content.parts
  ) {
    throw new Error(
      `Gemini returned no content. Finish reason: ${
        candidate.finishReason ||
        'unknown'
      }`
    );
  }

  const message =
    candidate.content.parts
      .map(
        part => part.text || ''
      )
      .join('');

  if (!message.trim()) {
    throw new Error(
      'Gemini returned an empty response'
    );
  }

  return message;
}


// ============================================================
// OPENAI
// ============================================================

async function callOpenAI(
  systemPrompt,
  userMessage,
  maxTokens = 1500
) {
  const apiKey =
    process.env.OPENAI_API_KEY;

  if (!apiKey) {
    throw new Error(
      'OPENAI_API_KEY not configured'
    );
  }

  const model =
    process.env.OPENAI_MODEL ||
    'gpt-4o-mini';

  const response =
    await fetch(
      'https://api.openai.com/v1/chat/completions',
      {
        method: 'POST',

        headers: {
          'Content-Type':
            'application/json',

          'Authorization':
            `Bearer ${apiKey}`
        },

        body: JSON.stringify({
          model,

          max_completion_tokens:
            maxTokens,

          messages: [
            {
              role: 'system',
              content: systemPrompt
            },

            {
              role: 'user',
              content: userMessage
            }
          ]
        })
      }
    );

  if (!response.ok) {
    const errText =
      await response.text();

    throw new Error(
      `OpenAI API error ${response.status}: ${errText}`
    );
  }

  const data =
    await response.json();

  if (
    !data.choices ||
    !data.choices[0] ||
    !data.choices[0].message
  ) {
    throw new Error(
      'OpenAI returned an empty response'
    );
  }

  return (
    data.choices[0].message.content
  );
}


// ============================================================
// GROQ
// ============================================================

async function callGroq(
  systemPrompt,
  userMessage,
  maxTokens = 1500,
  jsonResponse = false,
  conversationHistory = []
) {
  const apiKey =
    process.env.GROQ_API_KEY;

  if (!apiKey) {
    throw new Error(
      'GROQ_API_KEY not configured'
    );
  }

  const model =
    process.env.GROQ_MODEL ||
    'openai/gpt-oss-120b';

  const messages = [
    {
      role: 'system',
      content: systemPrompt
    },

    ...conversationHistory.map(
      message => ({
        role:
          message.role === 'assistant'
            ? 'assistant'
            : 'user',

        content:
          message.content
      })
    ),

    {
      role: 'user',
      content: userMessage
    }
  ];

  const requestBody = {
  model,

  messages,

  max_completion_tokens:
    maxTokens,

  temperature: 0.7,

  // Reduce hidden reasoning so more tokens
  // are available for the actual answer.
  reasoning_effort: 'low',

  // Do not return reasoning content to the user.
  include_reasoning: false
};

  if (jsonResponse) {
    requestBody.response_format = {
      type: 'json_object'
    };
  }

  const maxAttempts = 3;

  const retryDelays = [
    2000,
    5000,
    10000
  ];

  for (
    let attempt = 1;
    attempt <= maxAttempts;
    attempt++
  ) {
    try {

      const response =
        await fetch(
          'https://api.groq.com/openai/v1/chat/completions',
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json',

              'Authorization':
                `Bearer ${apiKey}`
            },

            body:
              JSON.stringify(
                requestBody
              )
          }
        );

      if (response.ok) {

        const data =
          await response.json();

        if (
          !data.choices ||
          !data.choices[0] ||
          !data.choices[0].message
        ) {
          throw new Error(
            'Groq returned an empty response'
          );
        }

        const message =
          data.choices[0].message.content;

        if (
          !message ||
          !message.trim()
        ) {
          throw new Error(
            'Groq returned an empty message'
          );
        }

        console.log(
          'GROQ MODEL:',
          model
        );

        console.log(
  'GROQ RESPONSE LENGTH:',
  message.length
);

console.log(
  'GROQ FINISH REASON:',
  data.choices[0].finish_reason
);

console.log(
  'GROQ USAGE:',
  JSON.stringify(data.usage)
);

return message;
      }

      const errText =
        await response.text();

      const retryable =
        response.status === 429 ||
        response.status === 500 ||
        response.status === 502 ||
        response.status === 503 ||
        response.status === 504;

      if (
        !retryable ||
        attempt === maxAttempts
      ) {
        throw new Error(
          `Groq API error ${response.status}: ${errText}`
        );
      }

      const retryAfterHeader =
        response.headers.get(
          'retry-after'
        );

      const retryAfterSeconds =
        Number(
          retryAfterHeader
        );

      const delay =
        Number.isFinite(
          retryAfterSeconds
        ) &&
        retryAfterSeconds > 0
          ? Math.min(
              retryAfterSeconds *
                1000,
              30000
            )
          : retryDelays[
              attempt - 1
            ];

      console.warn(
        `Groq returned ${response.status}. ` +
        `Retrying in ${Math.round(
          delay / 1000
        )}s ` +
        `(attempt ${
          attempt + 1
        }/${maxAttempts})...`
      );

      await sleep(delay);

    } catch (err) {

      if (
        attempt === maxAttempts
      ) {
        throw err;
      }

      console.warn(
        `Groq request failed: ${err.message}. ` +
        `Retrying...`
      );

      await sleep(
        retryDelays[
          attempt - 1
        ]
      );
    }
  }

  throw new Error(
    'Groq request failed after retries'
  );
}

// ============================================================
// MENTOR OUTPUT SANITIZER
// ============================================================

function sanitizeMentorOutput(text) {
  if (!text || typeof text !== 'string') {
    return '';
  }

  let output = text;

  // ------------------------------------------------------------
  // 1. Normalize line endings
  // ------------------------------------------------------------
  output = output.replace(/\r\n/g, '\n');
  output = output.replace(/\r/g, '\n');

  // ------------------------------------------------------------
  // 2. Convert unsupported deep headings
  //    Mentor frontend supports ## and ### headings.
  // ------------------------------------------------------------
  output = output.replace(
    /^\s*####+\s*/gm,
    '### '
  );

  // ------------------------------------------------------------
  // 3. Remove emoji/keycap characters from the beginning
  //    of headings so the markdown renderer can recognize them.
  // ------------------------------------------------------------
  output = output.replace(
    /^(#{2,3})\s*(?:🎯|📚|💡|🧠|🚀|🛠️|🔧|📌|✅|⭐|🔥|💻|📖|🎓|1️⃣|2️⃣|3️⃣|4️⃣|5️⃣|6️⃣|7️⃣|8️⃣|9️⃣|🔟)\s*/gm,
    '$1 '
  );

  // ------------------------------------------------------------
  // 4. Normalize markdown tables.
  //
  // The Mentor UI expects:
  //
  // | Header | Header |
  // |--------|--------|
  // | Data   | Data   |
  //
  // not blank lines between rows.
  // ------------------------------------------------------------
  const lines = output.split('\n');
  const cleanedLines = [];

  let insideTable = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    const isTableRow =
      /^\s*\|.*\|\s*$/.test(line);

    if (isTableRow) {
      insideTable = true;
      cleanedLines.push(line.trim());
      continue;
    }

    if (
      insideTable &&
      line.trim() === ''
    ) {
      const nextLine = lines[i + 1] || '';

      if (/^\s*\|.*\|\s*$/.test(nextLine)) {
        continue;
      }

      insideTable = false;
    }

    cleanedLines.push(line);
  }

  output = cleanedLines.join('\n');

  // ------------------------------------------------------------
  // 5. Convert vertical flow diagrams to a single-line flow.
  //
  // Example:
  //
  // Client
  // ↓
  // Server
  // ↓
  // Database
  //
  // becomes:
  //
  // Client → Server → Database
  // ------------------------------------------------------------
  const flowLines = output.split('\n');
  const finalLines = [];

  for (let i = 0; i < flowLines.length; i++) {
    const current = flowLines[i].trim();

    if (
      current === '↓' ||
      current === '⬇' ||
      current === '▼'
    ) {
      if (
        finalLines.length > 0 &&
        i + 1 < flowLines.length
      ) {
        const previous =
          finalLines[finalLines.length - 1].trim();

        const next =
          flowLines[i + 1].trim();

        if (
          previous &&
          next &&
          !previous.startsWith('#') &&
          !next.startsWith('#') &&
          !previous.startsWith('|') &&
          !next.startsWith('|')
        ) {
          finalLines[
            finalLines.length - 1
          ] = `${previous} → ${next}`;

          i++;
          continue;
        }
      }
    }

    finalLines.push(flowLines[i]);
  }

  output = finalLines.join('\n');

  // ------------------------------------------------------------
  // 6. Clean excessive blank lines.
  // ------------------------------------------------------------
  output = output.replace(
    /\n{4,}/g,
    '\n\n\n'
  );

  return output.trim();
}

// ============================================================
// PRIMARY AI DISPATCH
// ============================================================

async function callAI(
  systemPrompt,
  userMessage,
  maxTokens = 1500,
  jsonResponse = false
) {

  if (
    AI_PROVIDER === 'gemini'
  ) {
    return callGemini(
      systemPrompt,
      userMessage,
      maxTokens,
      jsonResponse
    );
  }

  if (
    AI_PROVIDER === 'openai'
  ) {
    return callOpenAI(
      systemPrompt,
      userMessage,
      maxTokens
    );
  }

  if (
    AI_PROVIDER === 'groq'
  ) {
    return callGroq(
      systemPrompt,
      userMessage,
      maxTokens,
      jsonResponse
    );
  }

  return callAnthropic(
    systemPrompt,
    userMessage,
    maxTokens
  );
}


// ============================================================
// ROADMAP GENERATION
// ============================================================

async function generateRoadmap(context) {
  const { career, skillScores, skillGaps, userName, selectedSkills } = context;

  const gapSummary = skillGaps
    .filter(g => g.gap > 0)
    .sort((a, b) => b.gap - a.gap)
    .map(g => `${g.skillName}: gap=${g.gap} (current=${g.currentLevel}%, required=${g.requiredLevel}%)`)
    .join('\n');

  const scoreSummary = skillScores
    .map(s => `${s.skillName}: ${s.score}% (${s.proficiency})`)
    .join('\n');

  const existingSkillsStr = selectedSkills && selectedSkills.length > 0
    ? selectedSkills.join(', ')
    : 'Not specified';

  const systemPrompt = `You are an expert curriculum designer creating personalized, comprehensive career learning roadmaps.

Output ONLY valid JSON. No markdown fences, no text outside the JSON object. The response must start with { and end with }.`;

  const userMessage = `Create a detailed, comprehensive, personalized learning roadmap for ${userName}.

Target Career: ${career}

Learner's Existing Skills: ${existingSkillsStr}

Current Skill Assessment:
${scoreSummary}

Skill Gaps to Address (highest priority first):
${gapSummary || 'No major gaps — focus on mastery and projects.'}

INSTRUCTIONS:
1. Create 5-8 phases that cover everything needed for ${career}
2. Each phase has multiple weeks (typically 2-5 weeks per phase)
3. Each week has 6-12 specific topics to learn
4. Build logically from foundations to advanced topics
5. Use the skill gaps to prioritize what needs most focus
6. Existing skills can be acknowledged but don't dwell on them — fill the GAPS
7. Every topic must have: a title, description, difficulty, estimatedHours, learningObjectives (2-4), resources (2-3 with real URLs), exercises (1-2)
8. Resources MUST use real, known URLs — only use these trusted domains:
   - docs.python.org, numpy.org/doc, pandas.pydata.org, scikit-learn.org
   - developer.mozilla.org, nodejs.org, reactjs.org, vuejs.org
   - docs.docker.com, kubernetes.io/docs, cloud.google.com/docs
   - aws.amazon.com/getting-started, learn.microsoft.com/azure
   - freecodecamp.org, kaggle.com/learn, coursera.org, fast.ai
   - github.com (only well-known repos), git-scm.com/doc
   - tensorflow.org, pytorch.org, huggingface.co/docs
   - leetcode.com, hackerrank.com, exercism.org
   - www.w3schools.com, javascript.info, learnpython.org

Return this exact JSON structure:

{
  "summary": "2-3 sentence personalized overview of this roadmap",
  "totalDuration": "X weeks",
  "phases": [
    {
      "phaseNumber": 1,
      "title": "Phase title",
      "description": "What this phase covers and why",
      "duration": "X weeks",
      "skills": ["skill1", "skill2"],
      "reason": "Why this phase is needed based on the learner's gaps",
      "weeks": [
        {
          "weekNumber": 1,
          "title": "Week title",
          "description": "What this week covers",
          "estimatedHours": 10,
          "practiceProject": "One sentence describing a mini-project for this week",
          "exercises": [
            {
              "exerciseNumber": 1,
              "title": "Exercise title (e.g. Build a calculator)",
              "description": "What to build or do, with clear deliverable",
              "difficulty": "Beginner"
            },
            {
              "exerciseNumber": 2,
              "title": "Another exercise title",
              "description": "Description of what to build or do",
              "difficulty": "Intermediate"
            }
          ],
          "topics": [
            {
              "topicNumber": 1,
              "title": "Topic title",
              "description": "What this topic is and why it matters",
              "difficulty": "Beginner",
              "estimatedHours": 1.5,
              "learningObjectives": ["objective 1", "objective 2", "objective 3"],
              "resources": [
                {
                  "title": "Resource name",
                  "url": "https://actual-real-url.com/path",
                  "type": "documentation",
                  "platform": "Platform name",
                  "isFree": true,
                  "description": "What this resource teaches"
                }
              ],
              "exercises": ["Brief exercise hint for this specific topic"]
            }
          ]
        }
      ]
    }
  ]
}

IMPORTANT RULES:
- difficulty must be exactly: "Beginner", "Intermediate", or "Advanced"
- type must be one of: "documentation", "course", "tutorial", "video", "practice", "book", "github", "other"
- All URLs must be real and functional — no placeholders like "example.com"
- Provide at least 6 topics per week (aim for 8-10 for important weeks)
- Provide 2-4 week-level exercises per week (these are the main practice projects)
- Topic-level exercises are brief hints (plain strings), week-level exercises are structured objects
- The roadmap should be comprehensive enough to genuinely prepare someone for ${career}
- Return ONLY the JSON object, nothing else`;

  try {
    const raw = await callAI(systemPrompt, userMessage, 8000, true);

    let jsonStr = raw.trim();

    // Strip markdown fences if present
    const fenceMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (fenceMatch) jsonStr = fenceMatch[1].trim();

    // Strip any leading/trailing non-JSON text
    const firstBrace = jsonStr.indexOf('{');
    const lastBrace  = jsonStr.lastIndexOf('}');
    if (firstBrace > 0 || lastBrace < jsonStr.length - 1) {
      jsonStr = jsonStr.substring(firstBrace, lastBrace + 1);
    }

    const parsed = JSON.parse(jsonStr);

    if (!parsed.phases || !Array.isArray(parsed.phases) || parsed.phases.length === 0) {
      throw new Error('Invalid roadmap structure: missing phases array');
    }

    // Validate and sanitize resources — remove any with fake/placeholder URLs
    const KNOWN_SAFE_DOMAINS = [
      'docs.python.org','numpy.org','pandas.pydata.org','scikit-learn.org',
      'developer.mozilla.org','nodejs.org','reactjs.org','vuejs.org',
      'docs.docker.com','kubernetes.io','cloud.google.com','aws.amazon.com',
      'learn.microsoft.com','freecodecamp.org','kaggle.com','coursera.org',
      'fast.ai','github.com','git-scm.com','tensorflow.org','pytorch.org',
      'huggingface.co','leetcode.com','hackerrank.com','exercism.org',
      'w3schools.com','javascript.info','learnpython.org','realpython.com',
      'flask.palletsprojects.com','fastapi.tiangolo.com','docs.djangoproject.com',
      'postgresql.org','mongodb.com','redis.io','graphql.org','swagger.io',
      'docker.com','openai.com','anthropic.com','deeplearning.ai'
    ];

    function isValidUrl(url) {
      if (!url || typeof url !== 'string') return false;
      if (!url.startsWith('http://') && !url.startsWith('https://')) return false;
      try {
        const u = new URL(url);
        const host = u.hostname.replace(/^www\./, '');
        return KNOWN_SAFE_DOMAINS.some(d => host === d || host.endsWith('.' + d));
      } catch { return false; }
    }

    function sanitizeResources(resources) {
      if (!Array.isArray(resources)) return [];
      return resources
        .filter(r => r && r.title)
        .map(r => ({
          title:       r.title       || '',
          url:         isValidUrl(r.url) ? r.url : '',
          type:        ['documentation','course','tutorial','video','practice','book','github','other'].includes(r.type) ? r.type : 'other',
          platform:    r.platform    || '',
          isFree:      r.isFree !== false,
          description: r.description || ''
        }));
    }

    // Normalize phases
    parsed.phases = parsed.phases.map((phase, pi) => {
      const weeks = Array.isArray(phase.weeks) ? phase.weeks.map((week, wi) => {
        const topics = Array.isArray(week.topics) ? week.topics.map((topic, ti) => ({
          topicNumber:        topic.topicNumber        || ti + 1,
          title:              topic.title              || `Topic ${ti + 1}`,
          description:        topic.description        || '',
          difficulty:         ['Beginner','Intermediate','Advanced'].includes(topic.difficulty) ? topic.difficulty : 'Beginner',
          estimatedHours:     topic.estimatedHours     || 1,
          learningObjectives: Array.isArray(topic.learningObjectives) ? topic.learningObjectives : [],
          resources:          sanitizeResources(topic.resources),
          exercises:          Array.isArray(topic.exercises) ? topic.exercises : [],
          completed:          false,
          completedAt:        null,
          startedAt:          null
        })) : [];

        // Normalize week exercises (structured objects)
        const weekExercises = Array.isArray(week.exercises) ? week.exercises
          .filter(e => e && (typeof e === 'object' ? e.title : e))
          .map((e, ei) => {
            if (typeof e === 'string') {
              // Legacy string format → convert to object
              return {
                exerciseNumber: ei + 1,
                title: e,
                description: '',
                difficulty: 'Beginner',
                completed: false,
                completedAt: null
              };
            }
            return {
              exerciseNumber: e.exerciseNumber || ei + 1,
              title: e.title || `Exercise ${ei + 1}`,
              description: e.description || '',
              difficulty: ['Beginner','Intermediate','Advanced'].includes(e.difficulty) ? e.difficulty : 'Beginner',
              completed: false,
              completedAt: null
            };
          }) : [];

        return {
          weekNumber:         week.weekNumber     || wi + 1,
          title:              week.title          || `Week ${wi + 1}`,
          description:        week.description    || '',
          estimatedHours:     week.estimatedHours || 8,
          practiceProject:    week.practiceProject || '',
          topics,
          exercises:          weekExercises,
          resources:          sanitizeResources(week.resources),
          status:             'not-started',
          progressPercentage: 0
        };
      }) : [];

      return {
        phaseNumber:        phase.phaseNumber   || pi + 1,
        title:              phase.title         || `Phase ${pi + 1}`,
        description:        phase.description   || '',
        duration:           phase.duration      || '',
        skills:             Array.isArray(phase.skills) ? phase.skills : [],
        reason:             phase.reason        || '',
        weeks,
        topics:             [],    // keep for legacy compat
        practicalExercises: [],
        resources:          [],
        status:             'not-started',
        progressPercentage: 0
      };
    });

    return { success: true, data: parsed };

  } catch (err) {
    console.error('AI roadmap generation failed:', err.message);
    return { success: false, error: err.message };
  }
}



async function chatWithMentor(
  context,
  userMessage,
  conversationHistory = []
) {

  const {
    career,
    skillScores,
    skillGaps,
    roadmapPhase,
    userName,
    topicContext
  } = context;

  const topicContextStr = topicContext ? `
CURRENT LEARNING CONTEXT (from roadmap):
Phase: ${topicContext.phase || ''}
Week: ${topicContext.week || ''}
Topic: ${topicContext.topic || ''}
Topic Description: ${topicContext.topicDescription || ''}
Difficulty: ${topicContext.difficulty || ''}
Learning Objectives: ${(topicContext.learningObjectives || []).join(', ')}

The user is currently studying this specific topic. Answer in the context of this topic.
If they ask "explain this", explain "${topicContext.topic}".
If they ask for exercises, give exercises specifically for "${topicContext.topic}".
If they ask to quiz them, quiz them on "${topicContext.topic}".
` : '';

  const topGaps =
    skillGaps
      .filter(
        g => g.gap > 0
      )
      .sort(
        (a, b) =>
          b.gap - a.gap
      )
      .slice(0, 3)
      .map(
        g =>
          `${g.skillName} (gap: ${g.gap})`
      )
      .join(', ');

  const skillSummary =
    skillScores
      .map(
        s =>
          `${s.skillName}: ${s.score}%`
      )
      .join(', ');

  // ------------------------------------------------------------------
  // STRICT, EXAMPLE-DRIVEN FORMAT CONTRACT
  //
  // Why so strict: the front-end (mentor.html) parses this text with a
  // hand-written markdown renderer (headings, tables, flow-diagram boxes,
  // tips, code blocks). That renderer only recognizes exact patterns.
  // Loose instructions like "use markdown" cause weaker/cheaper models
  // (esp. Groq's gpt-oss-120b) to drift — e.g. dropping "##" from
  // headings, or putting blank lines between table rows — which breaks
  // the renderer's parsing. Showing a literal example fixes this far
  // more reliably than describing the rule in prose.
  // ------------------------------------------------------------------
  const systemPrompt = `
You are an expert AI Career Mentor helping ${userName}
become a ${career}.

USER PROFILE

Target Career:
${career}

Current Skill Levels:
${skillSummary}

Top Skill Gaps:
${topGaps}

Current Roadmap Phase:
${roadmapPhase || 'Not started'}

${topicContextStr}

YOUR ROLE

Give personalized and actionable career guidance.

Use the user's actual skill levels and skill gaps when relevant.

Be encouraging but realistic.

Suggest specific:

- Topics
- Resources
- Projects
- Practice tasks
- Next steps

Use the conversation history for context continuity.

STRICT OUTPUT FORMAT — follow this EXACTLY. Do not deviate, and do not
mention these rules in your answer.

1. HEADINGS
   - Every major section MUST start with "## " on its own line,
     e.g. "## Overview", "## What You Should Learn", "## Practice",
     "## Project", "## Resources", "## Next Steps".
   - Sub-sections (like a specific week or project name) MUST use "### ".
   - There are only two heading levels: "##" and "###". NEVER use "####"
     or deeper.
   - NEVER start a heading with an emoji or a number emoji (e.g. "🎯",
     "1️⃣"). Headings are plain text after the "##"/"###" marker only,
     e.g. "## Quick Goal", not "🎯 Quick Goal".
   - Never present a section title as plain text or bold text only —
     it must use "##" or "###".

2. TABLES
   - Use standard markdown pipe tables only.
   - The separator row ("|---|---|") MUST be the line immediately after
     the header row. NEVER put a blank line between them.
   - Every data row MUST immediately follow the previous row.
     NEVER put a blank line between table rows.
   - Correct example:
     | Week | Focus | Goal |
     |------|-------|------|
     | 1 | JavaScript fundamentals | Close the biggest gap first |
     | 2 | HTML & CSS layout | Build responsive UI skills |

3. FLOW / PROCESS DIAGRAMS
   - Write exactly one line per diagram.
   - Join each step with " → " (a single arrow, spaces on both sides).
   - Never use vertical arrows, box-drawing characters, or multiple lines.
   - Correct example:
     Client (React) → Express Route → Controller → MongoDB → JSON Response

4. LISTS
   - Bullet points use "- " (dash + space).
   - Numbered steps use "1. ", "2. ", etc.
   - One idea per line. Never merge several bullet points into one
     paragraph.

5. EMPHASIS & LINKS
   - Bold key terms with **term**.
   - Only include real, working links, formatted as [label](https://example.com).
     Never invent a URL.

6. STRUCTURE
   - Keep paragraphs short: 2–3 sentences maximum.
   - Any response describing a plan or roadmap MUST end with a
     "## Next Steps" section written as a numbered checklist.
   - Never output the whole answer as one long unbroken paragraph.
   - Never return raw JSON in this chat context.

FULL WORKED EXAMPLE OF EXPECTED FORMAT (structure only — replace content
with what's actually relevant to this user):

## Overview

A short 2-3 sentence summary personalized to ${userName}'s current
skills and gaps.

## What You Should Learn

| Week | Focus | Goal |
|------|-------|------|
| 1 | Topic A | Why it matters |
| 2 | Topic B | Why it matters |

## Practice

- Daily drill 1
- Daily drill 2

## Project

### Mini Project Name

Client (React) → API Route → Database → Response

## Resources

- [MDN Web Docs](https://developer.mozilla.org)
- [freeCodeCamp](https://www.freecodecamp.org)

## Next Steps

1. First concrete action
2. Second concrete action
3. Third concrete action
`;

  const recentHistory =
    conversationHistory.slice(-20);

  try {

    // ----------------------------------------------------------
    // GEMINI MENTOR
    // ----------------------------------------------------------

    if (
      AI_PROVIDER === 'gemini'
    ) {

      const apiKey =
        process.env.GEMINI_API_KEY;

      if (!apiKey) {
        throw new Error(
          'GEMINI_API_KEY not configured'
        );
      }

      const model =
        process.env.GEMINI_MODEL ||
        'gemini-3.8-flash';

      const geminiContents = [
        ...recentHistory.map(
          msg => ({
            role:
              msg.role === 'assistant'
                ? 'model'
                : 'user',

            parts: [
              {
                text:
                  msg.content
              }
            ]
          })
        ),

        {
          role: 'user',

          parts: [
            {
              text:
                userMessage
            }
          ]
        }
      ];

      const response =
        await fetchGeminiWithRetry(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,

          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json',

              'x-goog-api-key':
                apiKey
            },

            body: JSON.stringify({
              systemInstruction: {
                parts: [
                  {
                    text:
                      systemPrompt
                  }
                ]
              },

              contents:
                geminiContents,

              generationConfig: {
                maxOutputTokens:
                  5000,

                temperature:
                  0.7
              }
            })
          },

          'Gemini mentor'
        );

      const data =
        await response.json();

      if (
        !data.candidates ||
        !data.candidates[0]
      ) {
        throw new Error(
          'Gemini mentor returned no candidate'
        );
      }

      const candidate =
        data.candidates[0];

      if (
        !candidate.content ||
        !candidate.content.parts
      ) {
        throw new Error(
          'Gemini mentor returned no content'
        );
      }

      const message =
        candidate.content.parts
          .map(
            part =>
              part.text || ''
          )
          .join('');

      if (!message.trim()) {
        throw new Error(
          'Gemini mentor returned an empty response'
        );
      }

      return {
        success: true,
        message: sanitizeMentorOutput(message)
      };
    }


    // ----------------------------------------------------------
    // GROQ MENTOR
    // ----------------------------------------------------------

    if (
      AI_PROVIDER === 'groq'
    ) {

      const message =
        await callGroq(
          systemPrompt,
          userMessage,
          10000,
          false,
          recentHistory
        );

      return {
        success: true,
        message: sanitizeMentorOutput(message)
      };
    }


    // ----------------------------------------------------------
    // OPENAI MENTOR
    // ----------------------------------------------------------

    if (
      AI_PROVIDER === 'openai'
    ) {

      const apiKey =
        process.env.OPENAI_API_KEY;

      if (!apiKey) {
        throw new Error(
          'OPENAI_API_KEY not configured'
        );
      }

      const model =
        process.env.OPENAI_MODEL ||
        'gpt-4o-mini';

      const messages = [
        {
          role: 'system',
          content:
            systemPrompt
        },

        ...recentHistory,

        {
          role: 'user',
          content:
            userMessage
        }
      ];

      const response =
        await fetch(
          'https://api.openai.com/v1/chat/completions',
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json',

              'Authorization':
                `Bearer ${apiKey}`
            },

            body: JSON.stringify({
              model,

              max_completion_tokens:
                5000,

              messages
            })
          }
        );

      if (!response.ok) {
        const errText =
          await response.text();

        throw new Error(
          `OpenAI error ${response.status}: ${errText}`
        );
      }

      const data =
        await response.json();

      if (
        !data.choices ||
        !data.choices[0] ||
        !data.choices[0].message
      ) {
        throw new Error(
          'OpenAI returned an empty response'
        );
      }

      return {
        success: true,

        message: sanitizeMentorOutput(
          data.choices[0]
            .message.content
        )
      };
    }


    // ----------------------------------------------------------
    // ANTHROPIC MENTOR
    // ----------------------------------------------------------

    const apiKey =
      process.env.ANTHROPIC_API_KEY;

    if (!apiKey) {
      throw new Error(
        'ANTHROPIC_API_KEY not configured'
      );
    }

    const model =
      process.env.ANTHROPIC_MODEL ||
      'claude-haiku-4-5-20251001';

    const anthropicMessages = [
      ...recentHistory,

      {
        role: 'user',
        content: userMessage
      }
    ];

    const response =
      await fetch(
        'https://api.anthropic.com/v1/messages',
        {
          method: 'POST',

          headers: {
            'Content-Type':
              'application/json',

            'x-api-key':
              apiKey,

            'anthropic-version':
              '2023-06-01'
          },

          body: JSON.stringify({
            model,

            max_tokens:
              5000,

            system:
              systemPrompt,

            messages:
              anthropicMessages
          })
        }
      );

    if (!response.ok) {
      const errText =
        await response.text();

      throw new Error(
        `Anthropic error ${response.status}: ${errText}`
      );
    }

    const data =
      await response.json();

    if (
      !data.content ||
      !data.content[0] ||
      !data.content[0].text
    ) {
      throw new Error(
        'Anthropic returned an empty response'
      );
    }

    return {
      success: true,

      message: sanitizeMentorOutput(
        data.content[0].text
      )
    };

  } catch (err) {

    console.error(
      'AI mentor chat failed:',
      err.message
    );

    return {
      success: false,
      error: err.message
    };
  }
}


// ============================================================
// LEARNING ADVICE
// ============================================================

async function generateLearningAdvice(
  context
) {

  const {
    skillName,
    currentLevel,
    requiredLevel,
    career
  } = context;

  const systemPrompt = `
You are a concise technical learning advisor.

Give practical and specific advice.

Use clear headings ("## " for sections), bullet points ("- "),
numbered lists ("1. "), and spacing.

Keep the answer focused and actionable.
`;

  const userMessage = `
The user wants to become a ${career}.

Their current ${skillName} level is ${currentLevel}%,
but they need ${requiredLevel}%.

What specific topics, resources,
practice tasks, and projects should they focus on
to close this gap?

Be specific and actionable.
`;

  try {

    const advice =
      await callAI(
        systemPrompt,
        userMessage,
        1000,
        false
      );

    return {
      success: true,
      advice
    };

  } catch (err) {

    console.error(
      'AI learning advice failed:',
      err.message
    );

    return {
      success: false,
      error: err.message
    };
  }
}


// ============================================================
// EXPORTS
// ============================================================

module.exports = {
  generateRoadmap,
  chatWithMentor,
  generateLearningAdvice
};