import { useEffect, type RefObject } from 'react';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import type { MyNode } from '../types/graph';

export function useGraphCamera(
  fgRef: RefObject<ForceGraphMethods<any, any> | undefined>, // Заменили MutableRefObject на RefObject
  isDetailsOpen: boolean | undefined,
  selectedNode: MyNode | null,
  nodes: MyNode[]
) {
  useEffect(() => {
    if (!fgRef.current) return;

    if (isDetailsOpen && selectedNode) {
      const graphNode = nodes.find(n => n.id === selectedNode.id) as any;
      if (graphNode && graphNode.x !== undefined && graphNode.y !== undefined) {
        try {
          fgRef.current.centerAt(graphNode.x - 84, graphNode.y, 1000); 
          fgRef.current.zoom(6, 1000); 
        } catch (e) {
          console.warn("Не удалось сфокусировать камеру на узле", e);
        }
      }
    } else if (!isDetailsOpen && !selectedNode) {
      if (nodes.length > 0 && (nodes[0] as any).x !== undefined) {
        try {
          fgRef.current.zoomToFit(800, 50);
        } catch (e) {
          console.warn("Не удалось отдалить камеру", e);
        }
      }
    }
  }, [isDetailsOpen, selectedNode, nodes, fgRef]);
}