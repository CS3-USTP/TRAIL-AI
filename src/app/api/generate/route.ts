import { NextResponse } from 'next/server';
import ChatResponse from '@/lib/ollama/main';

// Allow CORS for frontend requests
export function OPTIONS() {
	return NextResponse.json({}, { status: 200, headers: corsHeaders });
}

// CORS Headers
const corsHeaders = {
	'Access-Control-Allow-Origin': '*',
	'Access-Control-Allow-Methods': 'POST, OPTIONS',
	'Access-Control-Allow-Headers': 'Content-Type',
};

// Handle POST request
export async function POST(req: Request) {
	try {
		const { query } = await req.json();
		if (!query) {
			return NextResponse.json({ error: 'Missing query parameter' }, { status: 400, headers: corsHeaders });
		}

		const readableStream = await ChatResponse(query);

		return new Response(readableStream, {
			headers: {
				'Content-Type': 'text/event-stream',
				'Cache-Control': 'no-cache',
				Connection: 'keep-alive',
				...corsHeaders,
			},
		});
	} catch (error) {
		console.error('Error processing request:', error);
		return NextResponse.json({ error: 'Internal server error' }, { status: 500, headers: corsHeaders });
	}
}

// Reject GET requests
export function GET() {
	return NextResponse.json({ error: 'Method Not Allowed' }, { status: 405, headers: corsHeaders });
}
