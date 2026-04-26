import { Embedding } from "./embeddings";
import { Tokenizer } from "./tokenizer";
import { Attention } from "./attention";
import { FeedForward } from "./feedforward";

const text = "React uses hooks and React state";

const tokenizer = new Tokenizer();
tokenizer.fit(text);

const ids = tokenizer.encode("React uses state");

const embedding = new Embedding(Object.keys(tokenizer.vocab).length, 3);

const vectors = embedding.forward(ids);

console.log("Token IDs:", ids);
console.log("Vectors:");
console.table(vectors);


const Wq = [
  [1, 0],
  [0, 1],
  [0, 0],
];

const Wk = [
  [1, 0],
  [0, 1],
  [0, 0],
];

const Wv = [
  [1, 0],
  [0, 1],
  [0, 0],
];

const attention = new Attention(Wq, Wk, Wv);

const output = attention.forward(vectors);

console.log("\n--- INPUT ---");
console.table(vectors);

console.log("\n--- OUTPUT (context-aware vectors) ---");
console.table(output);

const W1 = [
  [1, 0, 1],
  [0, 1, 1],
]; // expand (2 → 3)

const W2 = [
  [1, 0],
  [0, 1],
  [1, 1],
]; // shrink (3 → 2)

const ff = new FeedForward(W1, W2);
const ffOutput = ff.forward(output);

console.log("\n--- FEEDFORWARD OUTPUT ---");
console.table(ffOutput);