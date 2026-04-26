import fetch from 'node-fetch';

export async function embedText(text:string){
    console.log('[embed] Sending to Ollama, text length:', text.length);
    const res = await fetch('http://localhost:11434/api/embeddings',{
        method: 'POST',
        body: JSON.stringify({
            model: 'nomic-embed-text',
            prompt: text
        })
    });

    console.log('[embed] Ollama response status:', res.status);
    if (!res.ok) {
        const errText = await res.text();
        console.error('[embed] Ollama error body:', errText);
        throw new Error(`Ollama embedding failed: ${res.status} ${errText}`);
    }

    const data = await res.json() as { embedding: number[] };
    console.log('[embed] Embedding vector length:', data.embedding?.length);
    return data.embedding;
}