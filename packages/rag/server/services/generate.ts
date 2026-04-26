import fetch from 'node-fetch';

export async function generateAnswer(question: string, docs: string[]) {
  const prompt = `
        Answer ONLY using contect below
    
        Context: ${docs.join('\n')}
    
        Question: ${question}
        `;
  console.log('[generate] Sending prompt to Ollama, length:', prompt.length);
  const res = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    body: JSON.stringify({
      model: 'llama3',
      prompt,
      stream: false
    })
  });

  console.log('[generate] Ollama status:', res.status);
  if (!res.ok) {
    const errText = await res.text();
    console.error('[generate] Ollama error:', errText);
    throw new Error(`Ollama generate failed: ${res.status} ${errText}`);
  }

  const data = (await res.json()) as { response: string };
  console.log('[generate] Raw response keys:', Object.keys(data));
  console.log('[generate] Response preview:', data.response?.slice(0, 200));
  return data.response;
}
