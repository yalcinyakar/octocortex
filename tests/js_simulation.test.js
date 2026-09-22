const assert = require('node:assert/strict');
const {LIFNeuron, SparseEventBus, OctoSimulation} = require('../octocortex/static/app.js');

const neuron = new LIFNeuron(1, 1);
assert.equal(neuron.step(.4), false);
assert.equal(neuron.step(.7), true);
assert.equal(neuron.spikes, 1);

const bus = new SparseEventBus(.5);
assert.equal(bus.observe({salience:.2}), false);
assert.equal(bus.observe({salience:.8}), true);
assert.equal(bus.drain().length, 1);

const simulation = new OctoSimulation();
let state;
for(let tick=0;tick<80;tick++){
  state = simulation.step();
  if(state.done) break;
}
assert.equal(state.done, true);
assert.deepEqual(state.agent, state.goal);
assert.ok(state.metrics.observations > state.metrics.published_events);

console.log('OctoCortex browser simulation tests passed');
