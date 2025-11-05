
export interface Problem {
  title: string;
  pseudoCode: string;
}

export interface Pattern {
  id: number;
  name: string;
  whenToUse: string;
  approach: string;
  example: string;
  problems: Problem[];
}
