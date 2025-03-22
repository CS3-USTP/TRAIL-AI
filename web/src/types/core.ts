export type Response = {
    success: boolean;
    document: string;
    reference: string;
    distance: string;
    score: string;
};

export type Message = {
	role: 'user' | 'model';
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