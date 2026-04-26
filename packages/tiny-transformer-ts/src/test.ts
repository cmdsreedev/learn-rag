import { Embedding } from './embeddings';
import { Tokenizer } from './tokenizer';
import { Attention } from './attention';
import { FeedForward } from './feedforward';
import { TransformerBlock } from './transformerBlock';
import { OutputLayer } from './output';
import { trainMany } from './train';
import { generate } from './generate';

const text = `React uses hooks
React uses state
Hooks manage state
State updates UI
React renders UI
Hooks reuse logic
State changes trigger renders
React components use hooks
Hooks simplify state management`


const tokenizer = new Tokenizer();
tokenizer.fit(text);

const ids = tokenizer.encode('React');

const embedding = new Embedding(Object.keys(tokenizer.vocab).length, 3);
const vectors = embedding.forward(ids);

console.log('Token IDs:', ids);
console.log('Vectors:');
console.table(vectors);

// attention weights: 3 -> 3
const Wq = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

const Wk = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

const Wv = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

const attention = new Attention(Wq, Wk, Wv);

const attentionOutput = attention.forward(vectors);

console.log('\n--- INPUT ---');
console.table(vectors);

console.log('\n--- ATTENTION OUTPUT ---');
console.table(attentionOutput);

// feedforward: 3 -> 4 -> 3
const W1 = [
  [1, 0, 0, 1],
  [0, 1, 0, 1],
  [0, 0, 1, 1],
];

const W2 = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 1, 1],
];

const ff = new FeedForward(W1, W2);

const ffOutput = ff.forward(attentionOutput);

console.log('\n--- FEEDFORWARD OUTPUT ---');
console.table(ffOutput);

const block = new TransformerBlock(attention, ff);

const finalOutput = block.forward(vectors);

console.log('\n--- TRANSFORMER OUTPUT ---');
console.table(finalOutput);

const vocabSize = Object.keys(tokenizer.vocab).length;

// final vector dim is 3, vocab size is 5
const Wout = Array.from({ length: 3 }, () =>
  Array.from({ length: vocabSize }, () => Math.random())
);

const outputLayer = new OutputLayer(Wout);

const outputProbs = outputLayer.forward(finalOutput);

console.log('\n--- TOKEN PROBABILITIES ---');
console.table(outputProbs);


trainMany(text, tokenizer, embedding, block, Wout, 200, 0.1);

console.log('\n--- GENERATION ---');
console.log(
  generate('Hooks', 5, tokenizer, embedding, block, Wout)
);