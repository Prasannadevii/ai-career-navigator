const Roadmap      = require('../models/Roadmap');
const AssessmentResult = require('../models/AssessmentResult');
const Career       = require('../models/Career');
const CareerJourney= require('../models/CareerJourney');
const aiService    = require('../services/aiService');

// ─── Sync journey progress from roadmap ──────────────────────────────────────
async function syncJourneyProgress(userId, roadmap) {
  if (!roadmap) return;
  const progress = roadmap.overallProgress || 0;
  const update = { progress, lastActivityAt: new Date() };
  if (progress >= 100) { update.status = 'COMPLETED'; update.completedAt = new Date(); }
  else if (progress > 0) update.status = 'IN_PROGRESS';
  else update.status = 'IN_PROGRESS';

  if (roadmap.journeyId) {
    await CareerJourney.findOneAndUpdate({ _id: roadmap.journeyId, userId }, { $set: update });
  } else {
    await CareerJourney.findOneAndUpdate(
      { userId, careerId: roadmap.careerId, status: { $ne: 'COMPLETED' } },
      { $set: update }
    );
  }
}

// ─── Build fallback system roadmap with week/topic structure ─────────────────
function buildSystemRoadmap(career, skillGaps, selectedSkills) {
  // Curated fallback roadmaps by career type for common careers
  const careerTitle = (career || '').toLowerCase();
  const isAIML = careerTitle.includes('ai') || careerTitle.includes('ml') || careerTitle.includes('machine learning') || careerTitle.includes('data scien');
  const isBackend = careerTitle.includes('backend') || careerTitle.includes('back-end') || careerTitle.includes('node') || careerTitle.includes('python dev');
  const isFullStack = careerTitle.includes('full stack') || careerTitle.includes('fullstack');
  const isCloud = careerTitle.includes('cloud') || careerTitle.includes('devops');
  const isFrontend = careerTitle.includes('frontend') || careerTitle.includes('front-end') || careerTitle.includes('react') || careerTitle.includes('ui');

  // Build from skill gaps if available
  if (skillGaps && skillGaps.length > 0) {
    const sorted = [...skillGaps].sort((a, b) => b.gap - a.gap).slice(0, 6);
    return sorted.map((g, i) => ({
      phaseNumber: i + 1,
      title: `${g.skillName} Mastery`,
      description: `Build proficiency in ${g.skillName}`,
      duration: g.gap > 40 ? '3 weeks' : '2 weeks',
      skills: [g.skillName],
      reason: `Your ${g.skillName} score (${g.currentLevel}%) needs to reach ${g.requiredLevel}%`,
      weeks: [{
        weekNumber: 1,
        title: `${g.skillName} Fundamentals`,
        description: `Learn the core concepts of ${g.skillName}`,
        estimatedHours: 10,
        practiceProject: `Build a small project using ${g.skillName}`,
        exercises: [
          { exerciseNumber:1, title:`${g.skillName} Starter Project`, description:`Build a simple application using ${g.skillName} fundamentals`, difficulty:'Beginner', completed:false, completedAt:null },
          { exerciseNumber:2, title:`${g.skillName} Practice Problems`, description:`Complete 5 practice problems to reinforce concepts`, difficulty:'Beginner', completed:false, completedAt:null }
        ],
        topics: [
          { topicNumber:1, title:`${g.skillName} Introduction`, description:`Overview of ${g.skillName}`, difficulty:'Beginner', estimatedHours:1, learningObjectives:[`Understand ${g.skillName}`,`Set up environment`], resources:[{ title:`${g.skillName} Documentation`, url:'', type:'documentation', platform:'Official Docs', isFree:true, description:`Official docs` }], exercises:[`Complete introductory tutorial`], completed:false, completedAt:null, startedAt:null },
          { topicNumber:2, title:`${g.skillName} Core Concepts`, description:`Core concepts and patterns`, difficulty:'Beginner', estimatedHours:2, learningObjectives:[`Apply core patterns`], resources:[], exercises:[`Build a simple example`], completed:false, completedAt:null, startedAt:null },
          { topicNumber:3, title:`${g.skillName} Practice`, description:`Practical exercises`, difficulty:'Intermediate', estimatedHours:3, learningObjectives:[`Build working examples`], resources:[], exercises:[`Complete 5 practice problems`], completed:false, completedAt:null, startedAt:null }
        ],
        resources: [],
        status: 'not-started',
        progressPercentage: 0
      }],
      topics: [], practicalExercises: [], resources: [],
      status: 'not-started', progressPercentage: 0
    }));
  }

  // Generic 3-phase fallback
  return [
    {
      phaseNumber:1, title:'Foundations', description:'Core skills for your career path',
      duration:'4 weeks', skills:[], reason:'Build the foundational knowledge',
      weeks: [{
        weekNumber:1, title:'Getting Started', description:'Set up your development environment and learn the basics',
        estimatedHours:8, practiceProject:'Set up your dev environment and complete a hello-world project',
        topics:[
          {topicNumber:1,title:'Environment Setup',description:'Install and configure your development tools',difficulty:'Beginner',estimatedHours:1,learningObjectives:['Set up dev environment'],resources:[],exercises:['Install required tools'],completed:false,completedAt:null,startedAt:null},
          {topicNumber:2,title:'Core Concepts',description:'Fundamental concepts of your target career',difficulty:'Beginner',estimatedHours:3,learningObjectives:['Understand key concepts'],resources:[],exercises:['Complete introductory exercises'],completed:false,completedAt:null,startedAt:null}
        ],
        resources:[], status:'not-started', progressPercentage:0
      }],
      topics:[], practicalExercises:[], resources:[],
      status:'not-started', progressPercentage:0
    }
  ];
}

// ─── POST /api/roadmap/generate ──────────────────────────────────────────────
exports.generateRoadmap = async (req, res, next) => {
  try {
    const userId    = req.user._id;
    const journeyId = req.body.journeyId || null;

    // journeyId is AUTHORITATIVE — if provided, always get career from the journey
    let journey   = null;
    let careerId  = null;

    if (journeyId) {
      journey = await CareerJourney.findOne({ _id: journeyId, userId });
      if (!journey) {
        return res.status(403).json({ success: false, message: 'Journey not found or access denied.' });
      }
      // career MUST come from the journey — never trust req.body.careerId when journeyId exists
      careerId = journey.careerId;
    } else {
      // No journeyId: fall back to body careerId or user's profile goal
      careerId = req.body.careerId || req.user.careerGoal;
    }

    if (!careerId) {
      return res.status(400).json({ success: false, message: 'Career goal required. Please complete your profile.' });
    }

    // Get assessment result for THIS journey
    let latestResult;
    if (journeyId) {
      latestResult = await AssessmentResult.findOne({ userId, journeyId })
        .populate('skillScores.skillId', 'name').populate('skillGaps.skillId', 'name').sort({ completedAt: -1 });
      if (!latestResult) {
        latestResult = await AssessmentResult.findOne({ userId, careerId, journeyId: null })
          .populate('skillScores.skillId', 'name').populate('skillGaps.skillId', 'name').sort({ completedAt: -1 });
      }
    } else {
      latestResult = await AssessmentResult.findOne({ userId, careerId })
        .populate('skillScores.skillId', 'name').populate('skillGaps.skillId', 'name').sort({ completedAt: -1 });
    }

    if (!latestResult) {
      return res.status(400).json({ success: false, message: 'Complete an assessment first before generating a roadmap.' });
    }

    const career = await Career.findById(careerId);
    if (!career) return res.status(404).json({ success: false, message: 'Career not found.' });

    // Selected skills from journey
    const selectedSkills = journey?.selectedSkills?.map(s => s.skillName).filter(Boolean) || [];

    const context = {
      career:   career.title,
      userName: req.user.name,
      selectedSkills,
      skillScores: latestResult.skillScores.map(s => ({
        skillName: s.skillName || s.skillId?.name || 'Unknown',
        score: s.score, proficiency: s.proficiency
      })),
      skillGaps: latestResult.skillGaps.filter(g => g.gap > 0).map(g => ({
        skillName: g.skillName || g.skillId?.name || 'Unknown',
        currentLevel: g.currentLevel, requiredLevel: g.requiredLevel, gap: g.gap, severity: g.severity
      }))
    };

    const aiResult = await aiService.generateRoadmap(context);

    let phases, aiSummary, generatedBy;

    if (aiResult.success && aiResult.data?.phases?.length > 0) {
      generatedBy = 'AI';
      aiSummary   = aiResult.data.summary || '';
      phases      = aiResult.data.phases;
    } else {
      generatedBy = 'system';
      aiSummary   = `System-generated roadmap for ${career.title}. Configure AI provider for a personalized plan.`;
      phases      = buildSystemRoadmap(career.title, context.skillGaps, selectedSkills);
    }

    // Deactivate previous roadmaps for this journey only
    if (journeyId) {
      await Roadmap.updateMany({ userId, journeyId, isActive: true }, { $set: { isActive: false } });
    } else {
      await Roadmap.updateMany({ userId, careerId, journeyId: null, isActive: true }, { $set: { isActive: false } });
    }

    // Version
    const versionQuery = journeyId ? { userId, journeyId } : { userId, careerId };
    const lastRoadmap  = await Roadmap.findOne(versionQuery).sort({ version: -1 });
    const version      = lastRoadmap ? lastRoadmap.version + 1 : 1;

    // Create
    const roadmap = new Roadmap({
      userId, journeyId: journeyId || null, careerId,
      assessmentResultId: latestResult._id, generatedBy,
      phases, aiSummary, version, overallProgress: 0, isActive: true
    });
    roadmap.recalculateProgress();
    await roadmap.save();

    // Update journey
    const journeyUpdate = { lastActivityAt: new Date(), status: 'IN_PROGRESS' };
    if (journeyId) {
      await CareerJourney.findOneAndUpdate({ _id: journeyId, userId }, { $set: journeyUpdate });
    } else {
      await CareerJourney.findOneAndUpdate({ userId, careerId, status: { $in: ['NOT_STARTED','IN_PROGRESS'] } }, { $set: journeyUpdate });
    }

    const populated = await Roadmap.findById(roadmap._id).populate('careerId', 'title icon');
    res.status(201).json({
      success: true,
      message: aiResult.success ? 'AI roadmap generated successfully.' : 'System roadmap generated (AI unavailable — configure AI_PROVIDER).',
      roadmap: populated,
      aiAvailable: aiResult.success
    });
  } catch (err) { next(err); }
};

// ─── GET /api/roadmap ────────────────────────────────────────────────────────
exports.getRoadmap = async (req, res, next) => {
  try {
    const userId = req.user._id;
    const { careerId, journeyId } = req.query;
    let roadmap;

    if (journeyId) {
      const journey = await CareerJourney.findOne({ _id: journeyId, userId });
      if (!journey) return res.status(403).json({ success: false, message: 'Journey not found or access denied.' });

      roadmap = await Roadmap.findOne({ userId, journeyId, isActive: true })
        .populate('careerId', 'title icon').sort({ version: -1 });

      if (!roadmap && journey.careerId) {
        roadmap = await Roadmap.findOne({ userId, careerId: journey.careerId, journeyId: null, isActive: true })
          .populate('careerId', 'title icon').sort({ version: -1 });
      }
    } else if (careerId) {
      roadmap = await Roadmap.findOne({ userId, careerId, isActive: true })
        .populate('careerId', 'title icon').sort({ version: -1 });
    } else {
      roadmap = await Roadmap.findOne({ userId, isActive: true })
        .populate('careerId', 'title icon').sort({ createdAt: -1 });
    }

    res.json({ success: true, roadmap: roadmap || null });
  } catch (err) { next(err); }
};

// ─── Helper: build full progress summary to return to frontend ────────────────
function buildProgressSummary(roadmap) {
  return {
    overallProgress:    roadmap.overallProgress,
    totalTopics:        roadmap.totalTopics,
    completedTopics:    roadmap.completedTopics,
    totalExercises:     roadmap.totalExercises     || 0,
    completedExercises: roadmap.completedExercises || 0,
    phases: roadmap.phases.map(p => ({
      _id:                p._id,
      progressPercentage: p.progressPercentage,
      status:             p.status,
      weeks: (p.weeks || []).map(w => ({
        _id:                w._id,
        progressPercentage: w.progressPercentage,
        status:             w.status,
        completedTopics:    (w.topics || []).filter(t => t.completed).length,
        totalTopics:        (w.topics || []).length,
        completedExercises: (w.exercises || []).filter(e => e.completed).length,
        totalExercises:     (w.exercises || []).length
      }))
    }))
  };
}

// ─── PATCH /api/roadmap/:roadmapId/topic/:topicId — complete/uncomplete topic ─
exports.updateTopicCompletion = async (req, res, next) => {
  try {
    const userId   = req.user._id;
    const { roadmapId, topicId } = req.params;
    const { completed } = req.body;

    if (typeof completed !== 'boolean') {
      return res.status(400).json({ success: false, message: 'completed must be a boolean.' });
    }

    // SECURITY: roadmap must belong to this user
    const roadmap = await Roadmap.findOne({ _id: roadmapId, userId });
    if (!roadmap) return res.status(404).json({ success: false, message: 'Roadmap not found or access denied.' });

    // Find topic across all phases → weeks
    let found = false;
    outer: for (const phase of roadmap.phases) {
      for (const week of phase.weeks || []) {
        for (const topic of week.topics || []) {
          if (topic._id.toString() === topicId) {
            topic.completed  = completed;
            topic.completedAt= completed ? new Date() : null;
            if (completed && !topic.startedAt) topic.startedAt = new Date();
            found = true;
            break outer;
          }
        }
      }
    }

    if (!found) return res.status(404).json({ success: false, message: 'Topic not found in roadmap.' });

    roadmap.recalculateProgress();
    await roadmap.save();
    await syncJourneyProgress(userId, roadmap);

    res.json({
      success: true,
      message: `Topic marked as ${completed ? 'completed' : 'incomplete'}.`,
      progress: buildProgressSummary(roadmap)
    });
  } catch (err) { next(err); }
};

// ─── PATCH /api/roadmap/:roadmapId/exercise/:exerciseId — complete/uncomplete exercise ─
exports.updateExerciseCompletion = async (req, res, next) => {
  try {
    const userId = req.user._id;
    const { roadmapId, exerciseId } = req.params;
    const { completed } = req.body;

    if (typeof completed !== 'boolean') {
      return res.status(400).json({ success: false, message: 'completed must be a boolean.' });
    }

    // SECURITY: roadmap must belong to this user
    const roadmap = await Roadmap.findOne({ _id: roadmapId, userId });
    if (!roadmap) return res.status(404).json({ success: false, message: 'Roadmap not found or access denied.' });

    // Find exercise across all phases → weeks
    let found = false;
    outer: for (const phase of roadmap.phases) {
      for (const week of phase.weeks || []) {
        for (const exercise of week.exercises || []) {
          if (exercise._id.toString() === exerciseId) {
            exercise.completed  = completed;
            exercise.completedAt= completed ? new Date() : null;
            found = true;
            break outer;
          }
        }
      }
    }

    if (!found) return res.status(404).json({ success: false, message: 'Exercise not found in roadmap.' });

    roadmap.recalculateProgress();
    await roadmap.save();
    await syncJourneyProgress(userId, roadmap);

    res.json({
      success: true,
      message: `Exercise marked as ${completed ? 'completed' : 'incomplete'}.`,
      progress: buildProgressSummary(roadmap)
    });
  } catch (err) { next(err); }
};

// ─── PUT /api/roadmap/progress (legacy phase-level progress) ────────────────
exports.updateProgress = async (req, res, next) => {
  try {
    const { roadmapId, phaseIndex, status, progressPercentage } = req.body;
    const roadmap = await Roadmap.findOne({ _id: roadmapId, userId: req.user._id });
    if (!roadmap) return res.status(404).json({ success: false, message: 'Roadmap not found or access denied.' });
    if (phaseIndex < 0 || phaseIndex >= roadmap.phases.length)
      return res.status(400).json({ success: false, message: 'Invalid phase index.' });

    if (status) roadmap.phases[phaseIndex].status = status;
    if (progressPercentage !== undefined)
      roadmap.phases[phaseIndex].progressPercentage = Math.min(100, Math.max(0, progressPercentage));

    const pct = roadmap.phases[phaseIndex].progressPercentage;
    if (pct === 100) roadmap.phases[phaseIndex].status = 'completed';
    else if (pct > 0) roadmap.phases[phaseIndex].status = 'in-progress';
    else if (!status) roadmap.phases[phaseIndex].status = 'not-started';

    // If this is a new-style roadmap with weeks, also recalculate from topics
    roadmap.recalculateProgress();
    await roadmap.save();
    await syncJourneyProgress(req.user._id, roadmap);

    res.json({ success: true, message: 'Progress updated.', roadmap });
  } catch (err) { next(err); }
};

// ─── POST /api/roadmap/recalculate ──────────────────────────────────────────
exports.recalculateRoadmap = async (req, res, next) => {
  return exports.generateRoadmap(req, res, next);
};

// ─── GET /api/roadmap/all ────────────────────────────────────────────────────
exports.getAllRoadmaps = async (req, res, next) => {
  try {
    const roadmaps = await Roadmap.find({ userId: req.user._id })
      .populate('careerId', 'title icon').sort({ createdAt: -1 });
    res.json({ success: true, roadmaps });
  } catch (err) { next(err); }
};
