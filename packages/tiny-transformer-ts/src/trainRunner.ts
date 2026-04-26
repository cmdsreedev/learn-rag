import { Tokenizer } from './tokenizer';
import { Embedding } from './embeddings';
import { Attention } from './attention';
import { FeedForward } from './feedforward';
import { TransformerBlock } from './transformerBlock';
import { trainMany } from './train';

const text = 'React uses hooks and React state';

// tokenizer
const tokenizer = new Tokenizer();
tokenizer.fit(text);

// embedding
const embedding = new Embedding(
  Object.keys(tokenizer.vocab).length,
  3
);

// attention weights (3 → 3)
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

// feedforward (3 → 4 → 3)
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

const block = new TransformerBlock(attention, ff);

// output weights
const vocabSize = Object.keys(tokenizer.vocab).length;

const Wout = Array.from({ length: 3 }, () =>
  Array.from({ length: vocabSize }, () => Math.random())
);

// TRAIN
console.log('\n--- TRAINING ---');

trainMany(
  text,
  tokenizer,
  embedding,
  block,
  Wout,
  300,
  0.05
);