const assert = require('node:assert/strict');
const {UNIT, CONCEPT, CAPABILITY, BENCHMARK_RESULTS, WORLD_MODEL_RESULTS, LIFNeuron, SparseEventBus, CapabilityRouter, OnlineVectorQuantizer, SemanticAdapter, OctoSimulation} = require('../octocortex/static/app.js');

const neuron = new LIFNeuron(1, 1);
assert.equal(neuron.step(.4), false);
assert.equal(neuron.step(.7), true);
assert.equal(neuron.spikes, 1);

const bus = new SparseEventBus(.5);
assert.equal(bus.observe({salience:.2}), false);
assert.equal(bus.observe({salience:.8}), true);
assert.equal(bus.drain().length, 1);

const adapter = new SemanticAdapter();
const vector = adapter.semanticVector(UNIT.PERCEPTION, CONCEPT.DANGER_SPIKE, [.8, 1, .5]);
const losses = Array.from({length:200}, () => adapter.encodeAndLearn(vector).loss);
const encoded = adapter.encodeAndLearn(vector);
assert.ok(losses.slice(-20).reduce((a,b)=>a+b)/20 < losses.slice(0,20).reduce((a,b)=>a+b)/20);
assert.ok(encoded.latent.filter(value => value !== 0).length <= 2);

const router = new CapabilityRouter();
const lease = router.resolve(CAPABILITY.PLAN_ROUTE, new Set([UNIT.MEMORY, UNIT.PERCEPTION]));
assert.equal(lease.owner, UNIT.MEMORY);
assert.equal(lease.delegated, true);

const quantizer = new OnlineVectorQuantizer();
const codes = Array.from({length:100}, () => quantizer.quantize([0,-.2,-.18,0]).code);
assert.equal(new Set(codes.slice(-20)).size, 1);
assert.ok(quantizer.quantize([0,-.2,-.18,0], false).error < .001);

const simulation = new OctoSimulation();
const firstStep = simulation.step();
assert.equal(firstStep.transition.unit_activity.length, 4);
assert.equal(firstStep.transition.all_events.length, 3);
assert.equal(firstStep.transition.pipeline.observed, 3);
assert.ok(firstStep.transition.all_events.some(event => event.published === false));
assert.ok(firstStep.transition.all_events.every(event => Number.isInteger(event.payload.semantic_code)));
let state;
for(let tick=0;tick<80;tick++){
  state = simulation.step();
  if(state.done) break;
}
assert.equal(state.done, true);
assert.deepEqual(state.agent, state.goal);
assert.ok(state.metrics.observations > state.metrics.published_events);
assert.ok(state.metrics.adapter_steps > 0);
assert.equal(BENCHMARK_RESULTS.length, 12);
assert.equal(WORLD_MODEL_RESULTS[1][2], 1);

const failedSimulation = new OctoSimulation();
failedSimulation.disabledUnits.add(UNIT.PLANNING);
failedSimulation.navigation = () => { throw new Error('primary planner was called'); };
for(let tick=0;tick<80&&!failedSimulation.done;tick++) failedSimulation.step();
assert.equal(failedSimulation.done, true);
assert.equal(failedSimulation.snapshot().metrics.planning_owner, 'memory');
assert.equal(failedSimulation.snapshot().metrics.capability_handoffs, 1);
assert.ok(failedSimulation.snapshot().metrics.backup_accuracy > .75);

console.log('OctoCortex browser simulation tests passed');
