// Importing Ollama using Node.js style in Deno
import ollama from 'ollama';

// Custom function to get relevant context
async function getRelevantContext(query: string): Promise<string> {
    // TODO: hardcoded example of relevant context
    return `University article 5 section 3 states that if a student is caught 
    bringing illegal substances to the premises, they shall be subject to expulsion.`;
}

// Function to call Phi 3.8 Instruct using Ollama with streaming support
async function generateResponse(): Promise<void> {

    // TODO: get relevant context from chroma
    const context = `University article 5 section 3 states that if a student is caught 
    bringing illegal substances to the premises, they shall be subject to expulsion.`;

    // TODO: if there is context
    // Context: "${context}".

    // TODO: otherwise, if there is no context
    // Instruction: Make a short response. Politely tell the user that the student handbook does not have information about "${query}". Ask if they have questions related to the student handbook instead. 

    // TODO: append conversation history up to 3 exchanges only
    const prompt = `
        <|system|>
        You are a member and an AI developed by the (CS3) Computer Science Student Society 
        from (USTP) University of Science and Technology of Southern Philippines. 
        You assist in answering the question using the given context 
        of our student handbook only and do not add any other detail.
        If unrelated, you must inform the user that you cannot provide an answer.

        Context: ${context}
        <|end|>
        <|user|>
        what if i dont get caught bringing illegal substances to the premises?
        <|end|>
        <|assistant|>
         I can't provide an answer. The context only mentions that students who are caught bringing illegal substances will be subject to expulsion, but it does not address the scenario of someone not getting caught.
        <|end|>
        <|user|>
        what does expulsion mean
        <|end|>
        <|assistant|>
    `;

    // Stream the response using Ollama
    const stream = await ollama.generate({
        model: "llama3.2",
        prompt: prompt,
        stream: true,
        raw: true,
        options: {
            temperature: 0.4,     // Less creative, more focused
            top_p: 0.8,           // Conservative token selection
            // Temperature: 0.7   // Balances creativity and coherence.
            // Top_p: 1.0         // Allows the model to consider a wide range of token choices.
        },

    });

    // Collect the streamed response
    let finalResponse = '';
    for await (const part of stream) {
        const encoded = new TextEncoder().encode(part.response);
        await Deno.stdout.write(encoded);
        finalResponse += part;
    }
}

generateResponse();
