import { useRef, useState, useMemo, useCallback, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import type { MyNode, GraphData } from '../../types/graph';
import { useGraphCamera } from '../../hooks/useGraphCamera';
import { drawNodeCanvasObject } from '../../utils/canvasHelper';

interface GraphProps {
  data: GraphData;
  selectedNode: MyNode | null;
  onNodeClick: (node: MyNode) => void;
  onNodeDoubleClick?: (node: MyNode) => void;
  onBackgroundClick: () => void;
  isDetailsOpen?: boolean;
}

export function Graph({ 
  data, selectedNode, onNodeClick, onNodeDoubleClick, onBackgroundClick, isDetailsOpen 
}: GraphProps) {
  const fgRef = useRef<ForceGraphMethods<any, any> | undefined>(undefined);
  const [hoverNode, setHoverNode] = useState<MyNode | null>(null);

  // --- ХУК КАМЕРЫ ---
  useGraphCamera(fgRef, isDetailsOpen, selectedNode, data.nodes);

  // --- ЛОГИКА ДВОЙНОГО КЛИКА ---
  const lastClickRef = useRef<{ id: string, time: number }>({ id: '', time: 0 });
  const handleNodeClick = useCallback((node: any) => {
    const now = Date.now();
    const last = lastClickRef.current;

    if (last.id === node.id && now - last.time < 300) {
      if (onNodeDoubleClick) onNodeDoubleClick(node as MyNode);
      lastClickRef.current = { id: '', time: 0 }; 
    } else {
      onNodeClick(node as MyNode);
      lastClickRef.current = { id: node.id, time: now };
    }
  }, [onNodeClick, onNodeDoubleClick]);

  // --- ВЫДЕЛЕНИЕ СВЯЗЕЙ ---
  const { highlightNodes, highlightLinks } = useMemo(() => {
    const nodes = new Set<string>();
    const links = new Set<any>();
    const activeNode = hoverNode || selectedNode;

    if (activeNode) {
      nodes.add(activeNode.id);
      data.links.forEach((link: any) => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;

        if (sourceId === activeNode.id || targetId === activeNode.id) {
          links.add(link);
          nodes.add(sourceId);
          nodes.add(targetId);
        }
      });
    }

    return { highlightNodes: nodes, highlightLinks: links };
  }, [data, hoverNode, selectedNode]);

  // Меняем курсор
  useEffect(() => {
    document.body.style.cursor = hoverNode ? 'pointer' : 'default';
  }, [hoverNode]);

  return (
    <ForceGraph2D
      ref={fgRef}
      graphData={data}
      backgroundColor="#0f0f11"
      
      // Стилизация связей
      linkColor={(link: any) => highlightLinks.has(link) ? 'rgba(255, 255, 255, 0.8)' : 'rgba(255, 255, 255, 0.1)'}
      linkWidth={(link: any) => highlightLinks.has(link) ? 3 : 1}
      linkDirectionalParticles={(link: any) => highlightLinks.has(link) ? 4 : 0}
      linkDirectionalParticleWidth={3}
      linkDirectionalParticleSpeed={0.01}

      // События
      onNodeClick={handleNodeClick}
      onBackgroundClick={onBackgroundClick}
      onNodeHover={(node) => setHoverNode((node as MyNode) || null)}

      // Отрисовка узла
      nodeCanvasObject={(node: any, ctx, globalScale) => {
        drawNodeCanvasObject(node, ctx, globalScale, {
          hoverNodeId: hoverNode?.id,
          selectedNodeId: selectedNode?.id,
          highlightNodes
        });
      }}
    />
  );
}