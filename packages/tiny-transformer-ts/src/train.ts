import { Tokenizer } from './tokenizer';
import { Embedding } from './embeddings';
import { TransformerBlock } from './transformerBlock';

function softmax(row: number[]) {
  const max = Math.max(...row);
  const exps = row.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function crossEntropy(probs: number[], targetIndex: number) {
  return -Math.log(probs[targetIndex] + 1e-9);
}

export function trainStep(
  tokenizer: Tokenizer,
  inputText: string,
  targetWord: string,
  embedding: Embedding,
  block: TransformerBlock,
  Wout: number[][],
  lr: number
) {
  const ids = tokenizer.encode(inputText);
  const vectors = embedding.forward(ids);
  const out = block.forward(vectors);

  const lastVector = out[out.length - 1];

  const logits = Wout[0].map((_, col) =>
    lastVector.reduce((sum, val, i) => sum + val * Wout[i][col], 0)
  );

  const probs = softmax(logits);

  const targetIndex = tokenizer.vocab[targetWord];
  const loss = crossEntropy(probs, targetIndex);

  for (let i = 0; i < Wout.length; i++) {
    for (let j = 0; j < Wout[0].length; j++) {
      const target = j === targetIndex ? 1 : 0;
      const error = probs[j] - target;

      Wout[i][j] -= lr * error * lastVector[i];
    }
  }

  return { loss, probs };
}

export function trainMany(
  text: string,
  tokenizer: Tokenizer,
  embedding: Embedding,
  block: TransformerBlock,
  Wout: number[][],
  epochs = 20,
  lr = 0.1
) {
  const words = tokenizer.tokenize(text);

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    for (let i = 1; i < words.length; i++) {
      const inputText = words.slice(0, i).join(' ');
      const targetWord = words[i];

      const result = trainStep(
        tokenizer,
        inputText,
        targetWord,
        embedding,
        block,
        Wout,
        lr
      );

      totalLoss += result.loss;
    }

    console.log(
      `Epoch ${epoch + 1}: loss = ${(totalLoss / (words.length - 1)).toFixed(4)}`
    );
  }
}