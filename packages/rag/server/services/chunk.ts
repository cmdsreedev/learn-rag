export function chunkText(text: string, size = 500){
    console.log('[chunk] Input word count:', text.split(' ').length, '| chunk size:', size);
    const words = text.split(' ');
    const chunks = []

    for(let i=0; i< words.length; i += size ){
        chunks.push(words.slice(i, i+ size).join(' '))
    }

    console.log('[chunk] Total chunks created:', chunks.length);
    return chunks
}