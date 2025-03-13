export type Coherence = {
    coherence: boolean;
    values: {
      contradiction: number;
      entailment: number;
      neutral: number;
    }
  };