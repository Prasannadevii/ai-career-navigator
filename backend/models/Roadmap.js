const mongoose = require('mongoose');

// ── Resource (inside a topic or week) ────────────────────────────────────────
const resourceSchema = new mongoose.Schema({
  title:       { type: String, required: true },
  url:         { type: String, default: '' },
  type:        { type: String, enum: ['documentation','course','tutorial','video','practice','book','github','other'], default: 'other' },
  platform:    { type: String, default: '' },
  isFree:      { type: Boolean, default: true },
  description: { type: String, default: '' }
}, { _id: true });

// ── Exercise (inside a week — structured, trackable) ─────────────────────────
const exerciseSchema = new mongoose.Schema({
  exerciseNumber:  { type: Number },
  title:           { type: String, required: true },
  description:     { type: String, default: '' },
  difficulty:      { type: String, enum: ['Beginner','Intermediate','Advanced'], default: 'Beginner' },
  completed:       { type: Boolean, default: false },
  completedAt:     { type: Date,    default: null }
}, { _id: true });

// ── Topic (inside a week) ─────────────────────────────────────────────────────
const topicSchema = new mongoose.Schema({
  topicNumber:        { type: Number },
  title:              { type: String, required: true },
  description:        { type: String, default: '' },
  difficulty:         { type: String, enum: ['Beginner','Intermediate','Advanced'], default: 'Beginner' },
  estimatedHours:     { type: Number, default: 1 },
  learningObjectives: [String],
  resources:          [resourceSchema],
  // Legacy: exercises as plain strings (kept for backward compat)
  exercises:          [String],
  completed:          { type: Boolean, default: false },
  completedAt:        { type: Date,    default: null },
  startedAt:          { type: Date,    default: null }
}, { _id: true });

// ── Week (inside a phase) ─────────────────────────────────────────────────────
const weekSchema = new mongoose.Schema({
  weekNumber:         { type: Number },
  title:              { type: String, required: true },
  description:        { type: String, default: '' },
  estimatedHours:     { type: Number, default: 8 },
  topics:             [topicSchema],
  exercises:          [exerciseSchema],   // ← structured, trackable exercises
  practiceProject:    { type: String, default: '' },
  resources:          [resourceSchema],
  status: {
    type: String,
    enum: ['not-started','in-progress','completed'],
    default: 'not-started'
  },
  progressPercentage: { type: Number, default: 0, min: 0, max: 100 }
}, { _id: true });

// ── Phase (top-level bucket) ──────────────────────────────────────────────────
const phaseSchema = new mongoose.Schema({
  phaseNumber:        { type: Number },
  title:              { type: String, required: true },
  description:        { type: String, default: '' },
  duration:           { type: String, default: '' },
  skills:             [String],
  reason:             { type: String, default: '' },
  weeks:              [weekSchema],
  // Legacy fields — kept for backward compat with old phase-only roadmaps
  topics:             [String],
  practicalExercises: [String],
  resources:          [String],
  status: {
    type: String,
    enum: ['not-started','in-progress','completed'],
    default: 'not-started'
  },
  progressPercentage: { type: Number, default: 0, min: 0, max: 100 }
}, { _id: true });

// ── Roadmap ───────────────────────────────────────────────────────────────────
const roadmapSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  journeyId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'CareerJourney',
    default: null
  },
  careerId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Career',
    required: true
  },
  assessmentResultId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'AssessmentResult'
  },
  generatedBy: {
    type: String,
    enum: ['AI','system'],
    default: 'AI'
  },
  phases: [phaseSchema],
  overallProgress:    { type: Number, default: 0, min: 0, max: 100 },
  totalTopics:        { type: Number, default: 0 },
  completedTopics:    { type: Number, default: 0 },
  totalExercises:     { type: Number, default: 0 },
  completedExercises: { type: Number, default: 0 },
  version:            { type: Number, default: 1 },
  isActive:           { type: Boolean, default: true },
  aiSummary:          { type: String, default: '' }
}, { timestamps: true });

roadmapSchema.index({ userId: 1, careerId: 1, isActive: 1 });
roadmapSchema.index({ userId: 1, journeyId: 1, isActive: 1 });

// ── Instance method: recalculate all progress from topic+exercise completion ──
// SINGLE SOURCE OF TRUTH — called after every completion change
roadmapSchema.methods.recalculateProgress = function() {
  let totalTopics = 0, completedTopics = 0;
  let totalExercises = 0, completedExercises = 0;

  for (const phase of this.phases) {
    if (phase.weeks && phase.weeks.length > 0) {
      let phaseTotal = 0, phaseCompleted = 0;

      for (const week of phase.weeks) {
        const wTopicsTotal      = week.topics.length;
        const wTopicsCompleted  = week.topics.filter(t => t.completed).length;
        const wExTotal          = (week.exercises || []).length;
        const wExCompleted      = (week.exercises || []).filter(e => e.completed).length;

        // Week progress = (completed topics + completed exercises) / (total topics + total exercises)
        const wTotal     = wTopicsTotal + wExTotal;
        const wCompleted = wTopicsCompleted + wExCompleted;

        week.progressPercentage = wTotal > 0 ? Math.round((wCompleted / wTotal) * 100) : 0;
        week.status = wTotal === 0         ? 'not-started'
                    : wCompleted === wTotal ? 'completed'
                    : wCompleted > 0        ? 'in-progress'
                    :                         'not-started';

        phaseTotal     += wTotal;
        phaseCompleted += wCompleted;

        totalTopics        += wTopicsTotal;
        completedTopics    += wTopicsCompleted;
        totalExercises     += wExTotal;
        completedExercises += wExCompleted;
      }

      phase.progressPercentage = phaseTotal > 0
        ? Math.round((phaseCompleted / phaseTotal) * 100) : 0;
      phase.status = phaseTotal === 0             ? 'not-started'
                   : phaseCompleted === phaseTotal ? 'completed'
                   : phaseCompleted > 0            ? 'in-progress'
                   :                                 'not-started';
    }
    // Legacy phase-only roadmap: progressPercentage stays as-is
  }

  this.totalTopics        = totalTopics;
  this.completedTopics    = completedTopics;
  this.totalExercises     = totalExercises     || 0;
  this.completedExercises = completedExercises || 0;

  const grandTotal     = totalTopics + totalExercises;
  const grandCompleted = completedTopics + completedExercises;
  this.overallProgress = grandTotal > 0
    ? Math.round((grandCompleted / grandTotal) * 100) : 0;
};

module.exports = mongoose.model('Roadmap', roadmapSchema);
