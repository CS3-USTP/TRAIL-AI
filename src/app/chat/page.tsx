'use client';

import { useState, useRef, useEffect } from 'react';

type MessageType = {
	role: 'user' | 'assistant';
	content: string;
};

export default function ChatPage() {
	const [query, setQuery] = useState('');
	const [messages, setMessages] = useState<MessageType[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState('');

	const messagesEndRef = useRef<HTMLDivElement>(null);
	const textareaRef = useRef<HTMLTextAreaElement>(null);

	// Auto-scroll to bottom of messages
	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	}, [messages]);

	// Auto-resize textarea based on content
	useEffect(() => {
		if (textareaRef.current) {
			textareaRef.current.style.height = 'auto';
			textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
		}
	}, [query]);

	const handleCopy = (text: string) => {
		navigator.clipboard.writeText(text);
	};

	const handleClearChat = () => {
		setMessages([]);
	};

	const handleKeyDown = (e: React.KeyboardEvent) => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSubmit(e);
		}
	};

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		if (!query.trim()) return;

		const userMessage = query.trim();
		// Add user message to the messages array
		const updatedMessages: MessageType[] = [...messages, { role: 'user', content: userMessage }];
		setMessages(updatedMessages);
		setQuery('');
		setError('');
		setLoading(true);

		try {
			// Add empty assistant message to show typing indicator
			setMessages([...updatedMessages, { role: 'assistant', content: '' }]);

			// Get the last 3 messages to send to API
			const lastMessages = updatedMessages.slice(-3);

            console.log(lastMessages);

			const res = await fetch('/api/generate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(lastMessages),
			});

			if (!res.ok) throw new Error(`Server responded with ${res.status}`);
			if (!res.body) throw new Error('No response from server');

			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let finalText = '';

			// Update the latest message as chunks come in
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				const chunk = decoder.decode(value, { stream: true });
				finalText += chunk;

				setMessages(prev => {
					const newMessages = [...prev];
					newMessages[newMessages.length - 1] = {
						role: 'assistant',
						content: finalText,
					};
					return newMessages;
				});
			}
		} catch (err) {
			setMessages(prev => prev.slice(0, -1)); // Remove the empty assistant message
			setError(err instanceof Error ? err.message : 'Failed to fetch response');
		} finally {
			setLoading(false);
			if (textareaRef.current) {
				textareaRef.current.focus();
			}
		}
	};

	return (
		<div className="flex flex-col h-screen max-h-screen bg-gray-50">
			{/* Header */}
			<header className="flex items-center justify-between px-6 py-4 bg-white border-b shadow-sm">
				<h1 className="text-2xl font-bold text-blue-600">Neuro AI Chat</h1>
				<div className="flex gap-2">
					<button onClick={handleClearChat} className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200">
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-refresh-cw">
							<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
							<path d="M21 3v5h-5"></path>
							<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
							<path d="M3 21v-5h5"></path>
						</svg>
						Clear Chat
					</button>
				</div>
			</header>

			{/* Chat messages area */}
			<div className="flex-1 overflow-y-auto p-4 space-y-4">
				{messages.length === 0 ? (
					<div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
						<div className="mb-4 p-3 bg-blue-100 rounded-full">
							<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600">
								<path d="M22 2 11 13"></path>
								<path d="m22 2-7 20-4-9-9-4 20-7z"></path>
							</svg>
						</div>
						<h2 className="text-xl font-semibold mb-2">Welcome to Neuro AI Chat</h2>
						<p className="max-w-md">Ask any question to get started. I'm here to help!</p>
					</div>
				) : (
					messages.map((message, index) => (
						<div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
							<div className={`max-w-3xl rounded-lg px-4 py-3 ${message.role === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-white border shadow-sm rounded-bl-none'}`}>
								<div className="flex justify-between items-start">
									<div className="prose prose-sm max-w-none">
										{message.content ? (
											<div className="whitespace-pre-wrap">{message.content}</div>
										) : (
											<div className="flex items-center">
												<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin mr-2">
													<path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
												</svg>
												<span>Thinking...</span>
											</div>
										)}
									</div>
									{message.role === 'assistant' && message.content && (
										<button onClick={() => handleCopy(message.content)} className="ml-2 p-1 text-gray-400 hover:text-gray-600 rounded" title="Copy to clipboard">
											<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
												<rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect>
												<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path>
											</svg>
										</button>
									)}
								</div>
							</div>
						</div>
					))
				)}
				<div ref={messagesEndRef} />
			</div>

			{/* Error message */}
			{error && (
				<div className="mx-6 mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 flex items-center">
					<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
						<circle cx="12" cy="12" r="10"></circle>
						<line x1="15" x2="9" y1="9" y2="15"></line>
						<line x1="9" x2="15" y1="9" y2="15"></line>
					</svg>
					{error}
				</div>
			)}

			{/* Input area */}
			<div className="p-4 bg-white border-t">
				<form onSubmit={handleSubmit} className="flex items-end gap-2 max-w-4xl mx-auto">
					<div className="relative flex-1">
						<textarea ref={textareaRef} value={query} onChange={e => setQuery(e.target.value)} onKeyDown={handleKeyDown} placeholder="Ask a question..." rows={1} disabled={loading} className="w-full p-3 pr-10 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 max-h-32" />
					</div>
					<button type="submit" disabled={loading || !query.trim()} className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:bg-blue-400 disabled:cursor-not-allowed">
						{loading ? (
							<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
								<path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
							</svg>
						) : (
							<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
								<path d="M22 2 11 13"></path>
								<path d="m22 2-7 20-4-9-9-4 20-7z"></path>
							</svg>
						)}
					</button>
				</form>
			</div>
		</div>
	);
}
