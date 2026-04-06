import { useRef, useState, useMemo, useCallback, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import type { MyNode, MyLink, GraphData } from '../../types/graph';

interface GraphProps {
  data: GraphData;
  selectedNode: MyNode | null;
  onNodeClick: (node: MyNode) => void;
  onNodeDoubleClick?: (node: MyNode) => void; // НОВОЕ
  onBackgroundClick: () => void;
  isDetailsOpen?: boolean; // НОВОЕ: Знает ли граф о том, что открыто большое окно
}

const GROUP_COLORS: Record<number, string> = {
  1: '#61dafb', // Blue (Root)
  2: '#bd34fe', // Purple (Category)
  3: '#ff4b4b', // Red (Movie)
};

export function Graph({ data, selectedNode, onNodeClick, onNodeDoubleClick, onBackgroundClick, isDetailsOpen }: GraphProps) {
  const fgRef = useRef<ForceGraphMethods<any, any> | undefined>(undefined);
  const [hoverNode, setHoverNode] = useState<MyNode | null>(null);

  // --- ЛОГИКА ДВОЙНОГО КЛИКА ---
  const lastClickRef = useRef<{ id: string, time: number }>({ id: '', time: 0 });
  const handleNodeClick = useCallback((node: any) => {
    const now = Date.now();
    const last = lastClickRef.current;

    if (last.id === node.id && now - last.time < 300) {
      // Это двойной клик (меньше 300мс)
      if (onNodeDoubleClick) onNodeDoubleClick(node as MyNode);
      lastClickRef.current = { id: '', time: 0 }; // сброс
    } else {
      // Это одинарный клик
      onNodeClick(node as MyNode);
      lastClickRef.current = { id: node.id, time: now };
    }
  }, [onNodeClick, onNodeDoubleClick]);

// --- ЛОГИКА ЗУМА КАМЕРЫ ---
  useEffect(() => {
    // Ждем, пока график и его методы не инициализируются
    if (!fgRef.current) return;

    if (isDetailsOpen && selectedNode) {
      // Ищем узел в актуальных данных графа
      const graphNode = data.nodes.find(n => n.id === selectedNode.id) as any;
      
      // Выполняем зум ТОЛЬКО если узел найден и физический движок уже рассчитал для него X и Y
      if (graphNode && graphNode.x !== undefined && graphNode.y !== undefined) {
        try {
          fgRef.current.centerAt(graphNode.x - 50, graphNode.y, 1000); 
          fgRef.current.zoom(6, 1000); 
        } catch (e) {
          console.warn("Не удалось сфокусировать камеру на узле", e);
        }
      }
    } else if (!isDetailsOpen && !selectedNode) {
      // Отдаляем камеру ТОЛЬКО если в графе уже есть узлы и у первого узла есть координаты
      if (data.nodes.length > 0 && (data.nodes[0] as any).x !== undefined) {
        try {
          fgRef.current.zoomToFit(800, 50);
        } catch (e) {
          console.warn("Не удалось отдалить камеру", e);
        }
      }
    }
  },[isDetailsOpen, selectedNode, data.nodes]);


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
  },[data, hoverNode, selectedNode]);

  useEffect(() => {
    document.body.style.cursor = hoverNode ? 'pointer' : 'default';
  }, [hoverNode]);

  return (
    <ForceGraph2D
      ref={fgRef}
      graphData={data}
      linkColor={(link: any) => highlightLinks.has(link) ? 'rgba(255, 255, 255, 0.8)' : 'rgba(255, 255, 255, 0.1)'}
      linkWidth={(link: any) => highlightLinks.has(link) ? 3 : 1}
      linkDirectionalParticles={(link: any) => highlightLinks.has(link) ? 4 : 0}
      linkDirectionalParticleWidth={3}
      linkDirectionalParticleSpeed={0.01}

      // Используем нашу функцию с таймером
      onNodeClick={handleNodeClick}
      onBackgroundClick={onBackgroundClick}
      onNodeHover={(node) => setHoverNode((node as MyNode) || null)}

      nodeCanvasObject={(node: any, ctx, globalScale) => {
        const isHovered = hoverNode?.id === node.id;
        const isSelected = selectedNode?.id === node.id;
        const isDimmed = (hoverNode || selectedNode) && !highlightNodes.has(node.id);
        const nodeRadius = node.val ?? 6;
        const baseColor = GROUP_COLORS[node.group] || '#999';

        if (isHovered || isSelected) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, nodeRadius + (isHovered ? 4 : 3), 0, 2 * Math.PI, false);
          ctx.fillStyle = isSelected ? 'rgba(255, 255, 255, 0.5)' : 'rgba(255, 255, 255, 0.3)';
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI, false);
        ctx.fillStyle = isDimmed ? 'rgba(80, 80, 80, 0.3)' : baseColor;
        ctx.fill();

        if (node.level === 1 && !isDimmed) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, nodeRadius * 0.9, 0, 2 * Math.PI, false);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.fill();
        }

        ctx.lineWidth = 1.5 / globalScale;
        ctx.strokeStyle = isDimmed ? 'rgba(0,0,0,0)' : '#1a1a1a';
        ctx.stroke();

        const showText = globalScale > 1.2 || isHovered || isSelected || highlightNodes.has(node.id);

        if (showText && !isDimmed) {
          const label = node.name;
          const fontSize = 12 / globalScale;
          ctx.font = `bold ${fontSize}px Sans-Serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = isHovered || isSelected ? '#ffffff' : 'rgba(255, 255, 255, 0.7)';
          ctx.fillText(label, node.x, node.y + nodeRadius + (10 / globalScale));
        }

        if (node.group === 3) {
          ctx.rect(node.x - nodeRadius, node.y - nodeRadius, nodeRadius * 2, nodeRadius * 2);
          ctx.fillStyle = baseColor;
          ctx.fill();
        } else {
          ctx.beginPath();
          ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI, false);
          ctx.fillStyle = baseColor;
          ctx.fill();
        }
      }}
      backgroundColor="#0f0f11"
    />
  );
}