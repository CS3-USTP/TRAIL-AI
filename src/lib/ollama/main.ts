
import ollama from 'ollama';


// Retrieve context from Chroma API
type ContextType = {
    document: string,
    reference: string,
    distance: string
}

async function retrieveContext(query: string): Promise<ContextType | null> {
    
    try {
        const chroma_api = "http://localhost:8000/semantic-search";
        const response = await fetch(chroma_api, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query, n_results: 10 }),
        });

        // Check if there is an error in the response
        if ('error' in response) { 
            console.error(response.error);
            return null;
        }

        // Return the context from the response
        return await response.json();
    }

    catch (error) {
        console.error(error);
        return null;
    }

} 


// Generate a response based on the given query and context
async function generateResponse(query: string): Promise<void> {

    // get relevant context from chroma
    const context = retrieveContext(query);

    // TODO: if there is context
    // Context: "${context}".

    // TODO: otherwise, if there is no context
    // Instruction: Make a short response. Politely tell the user that the student handbook does not have information about "${query}". Ask if they have questions related to the student handbook instead. 

    // TODO: append conversation history up to 3 exchanges only
    const prompt = `
        <|system|>
        You are a member and an AI developed by the (CS3) Computer Science Student Society 
        from (USTP) University of Science and Technology of Southern Philippines. 
        You assist students and faculty in answering the question using the given context 
        of our handbook only and do not add any other detail.
        If unrelated, you must inform the user that you cannot provide an answer.

        Context: ${context}
        <|end|>
        <|user|>
        do they accept foreign students
        <|end|>
        <|assistant|>
         Yes, USTP accepts foreign students. In fact, it is explicitly stated in Chapter 2. Admission and Registration Article. 1. Admission Section 6. Foreign/International Students. that "Foreign/international student applicants must submit the following requirements to take the Admission Test..." and upon arrival in the Philippines, foreign/international students may enroll in USTP after complying with certain requirements as stated in Chapter 2. Admission and Registration Article. 2. Registration Section 6. Foreign/International Students.
         <|end|>
         <|user|>
         can you tell me more about that
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
            // temperature: 0.8,     // Less creative, more focused
            // top_p: 0.8,           // Conservative token selection
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

const response = await retrieveContext('why is the sky so blue?');
console.log(response);