import ollama from 'ollama';

type ContextType = {
	success: boolean;
	document: string;
	reference: string;
	distance: string;
};

async function retrieveContext(query: string): Promise<ContextType> {
	const chroma_api = 'http://localhost:8000/semantic-search';

	const response = await fetch(chroma_api, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ query }),
	});

	return response.json();
}

async function generateSystemContext(query: string): Promise<string> {
	let command: string = `Your name is Neuro, an AI assistant developed by the CS3 (Computer Science Student Society) - a student organization at USTP (University of Science and Technology of Southern Philippines). Your role is to assist and inform users like the students, faculty, and staff by providing accurate and concise responses from the univesity handbook. Always add a lot of emojis.
    `;

	const context = await retrieveContext(query);
	console.log(context);

	if (!context.success) {
		command += `
        Politely tell the user that the university handbook does not have information about the user query "${query}".
        Do not answer even if it is a general or a common fact. Tell them about you and your purpose. Ask for questions related to the university handbook instead.
        `;
	} else {
		command += `        
        Only if the context entirely does not have information about the user query "${query}" then do not answer even if it is a general fact. Tell them about you and your purpose then ask for questions related to the university handbook instead. 

        Always be informative by saying "According to" base on the handbook especially on sensitive topics.

        USTP (University of Science and Technology of Southern Philippines) Handbook Context: "${context.document}"
        `;
	}

	return command;
}

// Handle POST request
export default async function ChatResponse(query: string): Promise<ReadableStream> {
	const command = await generateSystemContext(query);

	const messages = [
		{ role: 'system', content: command },
		{ role: 'user', content: query },
	];

	const stream = await ollama.chat({
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

		stream: true,
		messages,
		// options: { temperature: 0.75, top_p: 0.75 },
	});

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
// generateResponse("i want to bring meth")
// generateResponse("why is meth so hard to cook?");
// generateResponse("what happens if i get caught bringing a knife");
// generateResponse("what happens if i dont get caught bringing meth");
// generateResponse("what happens if i get caught bringing meth. why");
// generateResponse("provide a general information about why meth is hard to cook");

// unspecific, should doubt it
// generateResponse("can i wear a skirt");
// generateResponse("can i wear a skirt that falls below the knee");
// generateResponse("can i wear a skirt below the knee");

// unrelated, should refuse it and suggest to ask related questions
// generateResponse("why is the sky blue");
// generateResponse("what is the meaning of life");
// generateResponse("are you a mistral model");

// uses system and handbook welfare sections
// generateResponse("who are you");
// generateResponse("what is your name");
// generateResponse("what does your name mean");
// generateResponse("who is your creator");
// generateResponse("who am i");
// generateResponse("what is the purpose of your existence");
// generateResponse("who made you");
// generateResponse("what language model are you based on");

// greetings
// generateResponse("hello");
// generateResponse("wazzup neuro");

// it informs the handbook doesnt have it, then does it anyway
// generateResponse("can you say hello in filipino");
// generateResponse("what is 1 plus 1 just answer directly");
// generateResponse("what is the capital of the philippines");
// generateResponse("give me a joke about the policy");
