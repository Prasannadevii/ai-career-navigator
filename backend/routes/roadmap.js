const express = require('express');
const router  = express.Router();
const ctrl    = require('../controllers/roadmapController');
const { protect } = require('../middleware/auth');

// NOTE: specific named routes MUST come before parameterized routes
router.post('/generate',                                protect, ctrl.generateRoadmap);
router.post('/recalculate',                             protect, ctrl.recalculateRoadmap);
router.get('/all',                                      protect, ctrl.getAllRoadmaps);
router.get('/',                                         protect, ctrl.getRoadmap);
router.put('/progress',                                 protect, ctrl.updateProgress);

// Topic-level completion
router.patch('/:roadmapId/topic/:topicId',              protect, ctrl.updateTopicCompletion);

// Exercise-level completion (new)
router.patch('/:roadmapId/exercise/:exerciseId',        protect, ctrl.updateExerciseCompletion);

module.exports = router;
