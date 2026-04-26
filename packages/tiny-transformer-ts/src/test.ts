import { Embedding } from "./embeddings";
import { Tokenizer } from "./tokenizer";


const text = "React uses hooks and React state";

const tokenizer = new Tokenizer();
tokenizer.fit(text);

const ids = tokenizer.encode("React uses state");

const embedding = new Embedding(Object.keys(tokenizer.vocab).length, 4);

const vectors = embedding.forward(ids);

console.log("Token IDs:", ids);
console.log("Vectors:");
console.table(vectors);