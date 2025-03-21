export type Response = {
    success: boolean;
    document: string;
    reference: string;
    distance: string;
    score: string;
};

export type Message = {
	role: 'user' | 'assistant';
	content: string;
};

export type Coherence = {
    coherence: boolean;
    values: {
      contradiction: number;
      entailment: number;
      neutral: number;
    }
};