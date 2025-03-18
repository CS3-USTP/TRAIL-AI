import ollama from 'ollama';
import { Message } from 'ollama';
import { Response } from '@/types/vectordb';
import { Coherence } from '@/types/coherence';

async function fetchDatabase(query: string): Promise<string> {
    const chroma_api = 'http://localhost:8000/semantic-search';

    const response = await fetch(chroma_api, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
    });

    const result: Response = await response.json();

    return result.success ? result.document : '';
}


async function fetchCoherence(premise: string, hypothesis: string): Promise<boolean> {
    const coherence_api = 'http://localhost:8000/coherence-check';

    const response = await fetch(coherence_api, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ premise, hypothesis }),
    });

    const result: Coherence = await response.json();
    return result.coherence;

}

async function retrieveContext(messages: Message[]): Promise<string> {
    
    // check if there is only one message
    if (messages.length === 1) {
        // fetch the database with the message content
        const { role, content } = messages[0];
        const context = `${role}: ${content}`;
        const response = await fetchDatabase(context);
        return response;
    }

    // expect only 3 messages
    else if (messages.length !== 3)
        throw new Error('Invalid number of messages');

    // check if previous query is coherent with the new query
    const oldQuery = messages[messages.length - 3];
    const newQuery = messages[messages.length - 1];
    const coherence = await fetchCoherence(oldQuery.content, newQuery.content);

    console.log('\n\nCoherence: ', coherence, '\n\n');

    let context = "";
    if (coherence) {
        // if coherent, combine the messages
        context = messages.map(message => `${message.role}: ${message.content}`).join('\n\n');
    }
    else {
        // get the last message as the user query instead
        const message = messages[messages.length - 1];
        context = `${message.role}: ${message.content}`;
    }
    const response = await fetchDatabase(context);
    return response;
}

async function processMessages(messages: Message[]): Promise<Message[]> {
    let system: Message = { 
        role: 'system', 
        content: `Your name is Neuro, an AI assistant developed by the CS3 (COMPUTER SCIENCE STUDENT SOCIETY) - a student organization at USTP (UNIVERSITY OF SCIENCE AND TECHNOLOGY OF SOUTHERN PHILIPPINES). Your role is to assist users like the students, faculty, and staff by providing accurate and concise responses from the univesity handbook which focuses on the policies, guidelines, and regulations of the university. YOU ARE NOT ALLOWED TO ANSWER OBVIOUS COMMON OR GENERAL KNOWLEDGE THAT IS BEYOND THE SCOPE OF THE CONTEXT.
    `};

    // get the history of the conversation without the user query
    const history = messages.slice(0, messages.length - 1);
	
    // get the context
	const context = await retrieveContext(messages);    
    
    // get the last message as the user query
	let query = messages[messages.length - 1];

	if (!context) {
		query.content = `

        Query: "${query.content.toLowerCase()}"

        Tell the me that the university handbook does not have information about the query unfortunately. Add a lot of emojis. Tell me about you and your purpose. Ask for questions related to the university handbook instead. 
        
        `;
	} else {
		query.content += `
       
        Provide and answer by only using the university handbook context with an informative and detailed response. Always add a lot of emojis. Do not give external links that are not in the handbook. If the university handbook context does not have info about the query, refuse to answer even if its general knowledge, instead remind me about your purpose and ask me for questions related to the university handbook.

        Query: "${query.content.toLowerCase()}"       

        Context: "${context}"
        
        `;
	}
    
	const output = [ system, ...history, query ];
    return output;
}

export default async function GenerateAssistantResponse(messages: Message[]): Promise<ReadableStream> {
	// generate the response from the ollama API
	const stream = await ollama.chat({
		stream: true,
		messages: await processMessages(messages),
		options: { 
            temperature: 0.8, // more deterministic responses and focused on the context
            top_p: 0.8 // ensures relevant responses while maintaining diversity 
        },

		// faster responses, short response and less robust
		// more informative due to large context window
		// acts like a character when requested
		model: 'gemma2:2b',

		// faster responses like gemma, but more interactive
		// small context window strict and less informative
		// acts and like a character when requested
		// model: 'llama3.2',

		// interactive and informative responses,
		// best robust but slow
		// will sometimes answer general questions
		// acts and like a character when requested
		// model: 'mistral',

		// faster responses, less robust compared to mistral
		// will sometimes misinterpret the context
		// model: 'phi4-mini',

		// faster responses, short response and less robust
		// will refuse to answer taboo thats in the handbook
		// model: 'qwen2.5:3b',

		// never again it does random sht
		// model: 'phi3.5',
	});

	// create a readable stream object for the frontend
	const readableStream = new ReadableStream({
		async start(controller) {
			for await (const part of stream) {
				// send the part stream to the frontend
				controller.enqueue(part.message.content);
				// print part without new line
				process.stdout.write(part.message.content);
			}
			controller.close();
		},
	});
	// return the readable stream object
	return readableStream;
}

// == debugging purposes ==
// unethical, should refer to policies instead of refusing
// GenerateAssistantResponse("i want to bring meth")
// GenerateAssistantResponse("why is meth so hard to cook?");
// GenerateAssistantResponse("what happens if i get caught bringing a knife");
// GenerateAssistantResponse("what happens if i dont get caught bringing meth");
// GenerateAssistantResponse("what happens if i get caught bringing meth. why");
// GenerateAssistantResponse("provide a general information about why meth is hard to cook");

// unspecific, should doubt it
// GenerateAssistantResponse("can i wear a skirt");
// GenerateAssistantResponse("can i wear a skirt that falls below the knee");
// GenerateAssistantResponse("can i wear a skirt below the knee");

// unrelated, should refuse it and suggest to ask related questions
// GenerateAssistantResponse("why is the sky blue");
// GenerateAssistantResponse("what is the meaning of life");
// GenerateAssistantResponse("are you a mistral model");

// uses system and handbook welfare sections
// GenerateAssistantResponse("who are you");
// GenerateAssistantResponse("what is your name");
// GenerateAssistantResponse("what does your name mean");
// GenerateAssistantResponse("who is your creator");
// GenerateAssistantResponse("who am i");
// GenerateAssistantResponse("what is the purpose of your existence");
// GenerateAssistantResponse("who made you");
// GenerateAssistantResponse("what language model are you based on");

// greetings
// GenerateAssistantResponse("hello");
// GenerateAssistantResponse("wazzup neuro");

// it informs the handbook doesnt have it, then does it anyway
// GenerateAssistantResponse("can you say hello in filipino");
// GenerateAssistantResponse("what is 1 plus 1 just answer directly");
// GenerateAssistantResponse("what is the capital of the philippines");
// GenerateAssistantResponse("give me a joke about the policy");
