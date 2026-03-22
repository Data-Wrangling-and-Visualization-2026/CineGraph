export interface MyNode {
  id: string;
  name: string;
  group: number;
  val: number;
  level: number;
  type?: string;
  size?: number;

  color?: number | string;
  x?: number;
  y?: number;
}

export interface MyLink {
  source: string | MyNode;
  target: string | MyNode;
  color?: number | string;
}

export interface GraphData {
  nodes: MyNode[];
  links: MyLink[];
}