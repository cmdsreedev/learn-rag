import { Tokenizer } from './tokenizer';
import { Embedding } from './embeddings';
import { TransformerBlock } from './transformerBlock';

function softmax(row: number[]) {
  const max = Math.max(...row);
  const exps = row.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function sample(probs: number[]) {
  let r = Math.random();
  let sum = 0;

  for (let i = 0; i < probs.length; i++) {
    sum += probs[i];
    if (r < sum) return i;
  }
  return probs.length - 1;
}

export function generate(
  startText: string,
  steps: number,
  tokenizer: Tokenizer,
  embedding: Embedding,
  block: TransformerBlock,
  Wout: number[][]
) {
  let current = startText;

  for (let s = 0; s < steps; s++) {
    const ids = tokenizer.encode(current);
    const vectors = embedding.forward(ids);
    const out = block.forward(vectors);

    const lastVector = out[out.length - 1];

    const logits = Wout[0].map((_, col) =>
      lastVector.reduce((sum, val, i) => sum + val * Wout[i][col], 0)
    );

    const probs = softmax(logits);

    const nextId = probs.indexOf(Math.max(...probs));
    const nextWord = tokenizer.reverseVocab[nextId];

    current += ' ' + nextWord;
  }

  return current;
}