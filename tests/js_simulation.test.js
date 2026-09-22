const assert = require('node:assert/strict');
const {UNIT, CONCEPT, LIFNeuron, SparseEventBus, SemanticAdapter, OctoSimulation} = require('../octocortex/static/app.js');

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

const simulation = new OctoSimulation();
const firstStep = simulation.step();
assert.equal(firstStep.transition.unit_activity.length, 4);
assert.equal(firstStep.transition.all_events.length, 3);
assert.equal(firstStep.transition.pipeline.observed, 3);
assert.ok(firstStep.transition.all_events.some(event => event.published === false));
let state;
for(let tick=0;tick<80;tick++){
  state = simulation.step();
  if(state.done) break;
}
assert.equal(state.done, true);
assert.deepEqual(state.agent, state.goal);
assert.ok(state.metrics.observations > state.metrics.published_events);
assert.ok(state.metrics.adapter_steps > 0);

console.log('OctoCortex browser simulation tests passed');
