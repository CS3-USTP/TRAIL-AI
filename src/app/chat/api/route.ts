import { Message } from '@/types/inference';
import { NextResponse } from 'next/server';
import GenerateAssistantResponse from '@/lib/inference/main';

const corsHeaders = {
	'Access-Control-Allow-Origin': '*',
	'Access-Control-Allow-Methods': 'POST, OPTIONS',
	'Access-Control-Allow-Headers': 'Content-Type',
};

export function OPTIONS() {
    
    // allow cors for frontend requests
    return NextResponse.json(
        {}, 
        { status: 200, headers: corsHeaders }
    );
}

export async function POST(req: Request) {
	
    try {
		const messages: Message[] = await req.json();

		if (!validateChatHistory(messages)) {
            console.warn("\n\n/* ----------------------------- INVALID REQUEST ---------------------------- */\n\n");
            console.warn(messages);
            console.warn("\n\n/* -------------------------------------------------------------------------- */\n\n");

            return NextResponse.json(
                { error: 'INVALID REQUEST' }, 
                { status: 400, headers: corsHeaders }
            );
		}

		const readableStream = await GenerateAssistantResponse(messages);

		return new Response(readableStream, {
			headers: {
				'Content-Type': 'text/event-stream',
				'Cache-Control': 'no-cache',
				Connection: 'keep-alive',
				...corsHeaders,
			},
		});
	} 
    
    catch (error) {
		console.error('\n\n/* -------------------------- INTERNAL SERVER ERROR ------------------------- */\n\n');
        console.error(error);
		console.error('\n\n/* -------------------------------------------------------------------------- */\n\n');
 
        return NextResponse.json(
            { error: 'INTERNAL SERVER ERROR' }, 
            { status: 500, headers: corsHeaders }
        );       
	}
}


export function GET() {
	return NextResponse.json(
        { error: 'Method Not Allowed' }, 
        { status: 405, headers: corsHeaders }
    );
}

function validateChatHistory(messages: Message[]): boolean {

    const MAX_USER_LENGTH      = 100;
    const MAX_ASSISTANT_LENGTH = 6000;

    const isAnArray = Array.isArray(messages);
    
    if ( !isAnArray ) 
        return false;

    const hasAtMostThreeMessages = messages.length <= 3;
    const isMessageCountOdd      = messages.length % 2 === 1; // the query is last
    
    if ( !hasAtMostThreeMessages || !isMessageCountOdd ) 
        return false;

    let index;
    for (index = 0; index < messages.length; index++) {
        
        const message             = messages[index];
        const isAnObject          = typeof message === 'object';
        const hasExactlyTwoKeys   = Object.keys(message).length === 2;
        const hasRoleKey          = 'role'    in message;
        const hasContentKey       = 'content' in message;
        const hasValidRole        = message.role === 'user' || message.role === 'assistant';
        const hasValidContent     = typeof message.content === 'string';

        // check if the role sequence is correct
        const isUserTurn          = index % 2 === 0;
        const isAssistantTurn     = index % 2 === 1;
        const followsRoleSequence = (
               (isUserTurn      && message.role === 'user') 
            || (isAssistantTurn && message.role === 'assistant')
        );

        const isContentLengthValid = (
               (isUserTurn      && message.content.length <= MAX_USER_LENGTH) 
            || (isAssistantTurn && message.content.length <= MAX_ASSISTANT_LENGTH)
        );

        if (
               !isAnObject 
            || !hasExactlyTwoKeys 
            || !hasRoleKey 
            || !hasContentKey 
            || !hasValidRole 
            || !hasValidContent 
            || !followsRoleSequence
            || !isContentLengthValid
        ) return false;
    }
    
    return true;
}