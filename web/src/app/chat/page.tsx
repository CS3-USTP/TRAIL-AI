'use client';

import { useState, useRef, useEffect } from 'react';

type MessageType = {
	role: 'user' | 'model';
	content: string;
};

export default function ChatPage() {
	const [query, setQuery] = useState('');
	const [messages, setMessages] = useState<MessageType[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState('');
	const MAX_CHARS = 100;

	const messagesEndRef = useRef<HTMLDivElement>(null);
	const textareaRef = useRef<HTMLTextAreaElement>(null);

	// Auto-scroll to bottom of messages
	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	}, [messages]);

	// Auto-resize textarea based on content and only show scrollbar if more than 2 lines
	useEffect(() => {
		if (textareaRef.current) {
			textareaRef.current.style.height = 'auto';
			textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
			
			// Get line height and calculate if content is more than 2 lines
			const lineHeight = parseInt(window.getComputedStyle(textareaRef.current).lineHeight);
			const lines = textareaRef.current.scrollHeight / lineHeight;
			
			if (lines <= 2) {
				textareaRef.current.style.overflowY = 'hidden';
			} else {
				textareaRef.current.style.overflowY = 'auto';
			}
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

	const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
		const value = e.target.value;
		// Limit input to MAX_CHARS characters
		if (value.length <= MAX_CHARS) {
			setQuery(value);
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
			// Add empty model message to show typing indicator
			setMessages([...updatedMessages, { role: 'model', content: '' }]);

			// Get the last 3 messages to send to API
			const lastMessages = updatedMessages.slice(-3);

            console.log(lastMessages);

			const res = await fetch('/chat/api', {
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
						role: 'model',
						content: finalText,
					};
					return newMessages;
				});
			}
		} catch (err) {
			setMessages(prev => prev.slice(0, -1)); // Remove the empty model message
			setError(err instanceof Error ? err.message : 'Failed to fetch response');
		} finally {
			setLoading(false);
			if (textareaRef.current) {
				textareaRef.current.focus();
			}
		}
	};

	// Calculate character count color based on how close to limit
	const getCharCountColor = () => {
		const percentage = query.length / MAX_CHARS;
		if (percentage < 0.7) return 'text-gray-400';
		if (percentage < 0.9) return 'text-yellow-400';
		return 'text-red-400';
	};

	return (
		<div className="flex flex-col h-screen max-h-screen bg-gray-900 text-gray-100">
			{/* Header */}
			<header className="flex items-center justify-between px-6 py-4 bg-gray-800 border-b border-gray-700">
				<div className="flex items-center gap-2">
					<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-400">
						<circle cx="12" cy="12" r="10"></circle>
						<circle cx="12" cy="12" r="4"></circle>
						<line x1="21.17" x2="12" y1="8" y2="12"></line>
					</svg>
					<h1 className="text-md font-semibold text-indigo-400">Trail AI</h1>
				</div>
				<div className="flex gap-2">
					<button 
						onClick={handleClearChat} 
						className="flex items-center gap-1 px-3 py-2 text-sm text-gray-300 border border-gray-700 rounded-md bg-gray-800 hover:bg-gray-700 transition-colors"
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
							<rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
							<line x1="12" y1="8" x2="12" y2="16"></line>
							<line x1="8" y1="12" x2="16" y2="12"></line>
						</svg>
						New Chat
					</button>
				</div>
			</header>

			{/* Chat messages area */}
			<div className="flex-1 overflow-y-auto p-4 space-y-6">
				{messages.length === 0 ? (
					<div className="flex flex-col items-center justify-center h-full text-center text-gray-400">
						<div className="mb-4 p-4 bg-gray-800 rounded-full">
							<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-400">
								<path d="M22 2 11 13"></path>
								<path d="m22 2-7 20-4-9-9-4 20-7z"></path>
							</svg>
						</div>
						<h2 className="text-2xl font-semibold mb-2 text-gray-200">How can I help you today?</h2>
						<p className="max-w-md text-gray-400">Ask me anything or start a conversation.</p>
					</div>
				) : (
					messages.map((message, index) => (
						<div key={index} className="max-w-3xl mx-auto">
							<div className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
								<div className={`max-w-[85%] rounded-2xl px-4 py-3 ${
									message.role === 'user' 
									? 'bg-indigo-600 text-white' 
									: 'bg-gray-800 text-gray-100 border border-gray-700'
								}`}>
									<div className="flex justify-between items-start">
										<div className="prose prose-invert prose-sm max-w-none">
											{message.content ? (
												<div className="whitespace-pre-wrap">{message.content}</div>
											) : (
												<div className="flex items-center justify-center py-2">
													<div className="thinking-animation">
														<div className="brain-container">
															<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-400">
																<circle cx="12" cy="12" r="10"></circle>
																<circle cx="12" cy="12" r="4"></circle>
																<line x1="21.17" x2="12" y1="8" y2="12"></line>
															</svg>
															<div className="pulse-rings"></div>
														</div>
														<div className="ml-3 thinking-text">Thinking...</div>
													</div>
												</div>
											)}
										</div>
										{message.role === 'model' && message.content && (
											<button 
												onClick={() => handleCopy(message.content)} 
												className="ml-2 p-1 text-gray-500 hover:text-gray-300 transition-colors" 
												title="Copy to clipboard"
											>
												<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
													<rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect>
													<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path>
												</svg>
											</button>
										)}
									</div>
								</div>
							</div>
						</div>
					))
				)}
				<div ref={messagesEndRef} />
			</div>

			{/* Error message */}
			{error && (
				<div className="mx-6 mb-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-400 flex items-center">
					<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
						<circle cx="12" cy="12" r="10"></circle>
						<line x1="15" x2="9" y1="9" y2="15"></line>
						<line x1="9" x2="15" y1="9" y2="15"></line>
					</svg>
					{error}
				</div>
			)}

			{/* Input area */}
			<div className="p-4 bg-gray-800 border-t border-gray-700">
				<form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative">
					<div className="relative">
						<textarea 
							ref={textareaRef} 
							value={query} 
							onChange={handleInputChange} 
							onKeyDown={handleKeyDown} 
							placeholder="Send a message..." 
							rows={1} 
							disabled={loading} 
							className="w-full p-3 pr-12 bg-gray-700 border border-gray-600 rounded-lg resize-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 text-gray-100 placeholder-gray-400 max-h-32" 
							maxLength={MAX_CHARS}
						/>
						{/* Character counter */}
						<div className={`absolute text-xs bottom-5.5 right-14 ${getCharCountColor()}`}>
							{query.length}/{MAX_CHARS}
						</div>
						<button 
							type="submit" 
							disabled={loading || !query.trim()} 
							className="absolute bottom-3 right-3 p-2 text-indigo-400 hover:text-indigo-300 disabled:text-gray-500 disabled:cursor-not-allowed rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
						>
							{loading ? (
								<div className="spinner-container">
									<div className="spinner"></div>
								</div>
							) : (
								<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
									<path d="M22 2 11 13"></path>
									<path d="m22 2-7 20-4-9-9-4 20-7z"></path>
								</svg>
							)}
						</button>
					</div>
				</form>
			</div>

			{/* CSS for loading animation */}
			<style jsx>{`
				.thinking-animation {
					display: flex;
					align-items: center;
				}
				
				.brain-container {
					position: relative;
					width: 30px;
					height: 30px;
					display: flex;
					align-items: center;
					justify-content: center;
				}
				
				.pulse-rings {
					position: absolute;
					top: 0;
					left: 0;
					right: 0;
					bottom: 0;
					border-radius: 50%;
					border: 2px solid rgba(144, 137, 252, 0.3);
					animation: pulse 2s infinite ease-out;
				}
				
				.pulse-rings:before, .pulse-rings:after {
					content: '';
					position: absolute;
					top: 0;
					left: 0;
					right: 0;
					bottom: 0;
					border-radius: 50%;
					border: 2px solid rgba(144, 137, 252, 0.3);
					animation: pulse 2s infinite ease-out;
				}
				
				.pulse-rings:before {
					animation-delay: 0.5s;
				}
				
				.pulse-rings:after {
					animation-delay: 1s;
				}
				
				.thinking-text {
					font-size: 0.9rem;
					color: #9089fc;
					opacity: 0.9;
					animation: fadeInOut 2s infinite;
				}
				
				.spinner-container {
					width: 20px;
					height: 20px;
					display: flex;
					align-items: center;
					justify-content: center;
				}
				
				.spinner {
					width: 16px;
					height: 16px;
					border: 2px solid rgba(144, 137, 252, 0.3);
					border-top: 2px solid #9089fc;
					border-radius: 50%;
					animation: spin 0.8s linear infinite;
				}
				
				@keyframes pulse {
					0% {
						transform: scale(0.8);
						opacity: 0.8;
					}
					50% {
						transform: scale(1.5);
						opacity: 0;
					}
					100% {
						transform: scale(0.8);
						opacity: 0;
					}
				}
				
				@keyframes fadeInOut {
					0%, 100% {
						opacity: 0.4;
					}
					50% {
						opacity: 1;
					}
				}
				
				@keyframes spin {
					0% {
						transform: rotate(0deg);
					}
					100% {
						transform: rotate(360deg);
					}
				}
			`}</style>
		</div>
	);
}