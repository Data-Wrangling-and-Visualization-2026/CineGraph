import { useRef, useState, useMemo, useCallback, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import type { MyNode, GraphData } from '../../types/graph';
import { useGraphCamera } from '../../hooks/useGraphCamera';
import { drawNodeCanvasObject, getNodeColor } from '../../utils/canvasHelper';

interface GraphProps {
  data: GraphData;
  selectedNode: MyNode | null;
  onNodeClick: (node: MyNode) => void;
  onNodeDoubleClick?: (node: MyNode) => void;
  onBackgroundClick: () => void;
  isDetailsOpen?: boolean;
  expandedNodeIds?: Set<string>;
}

function getCollisionRadius(node: any): number {
  if (node.level === 1) return 60;
  if (node.group === 2) return 30 + (node.childCount || 1) * 2.5;
  return 8;
}

function createCollisionForce(radiusFn: (n: any) => number, strength = 0.8) {
  let nodes: any[] = [];
  const ITERATIONS = 3;

  function force(alpha: number) {
    for (let iter = 0; iter < ITERATIONS; iter++) {
      for (let i = 0; i < nodes.length - 1; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = (b.x ?? 0) - (a.x ?? 0) || 1e-6;
          const dy = (b.y ?? 0) - (a.y ?? 0) || 1e-6;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const minDist = radiusFn(a) + radiusFn(b);
          if (dist < minDist && dist > 0) {
            const push = ((minDist - dist) / dist) * strength * alpha * 0.5;
            a.vx = (a.vx ?? 0) - dx * push;
            a.vy = (a.vy ?? 0) - dy * push;
            b.vx = (b.vx ?? 0) + dx * push;
            b.vy = (b.vy ?? 0) + dy * push;
          }
        }
      }
    }
  }
  force.initialize = (n: any[]) => { nodes = n; };
  return force;
}

function createRadialForce(cx = 0, cy = 0, strength = 0.04) {
  let nodes: any[] = [];

  function force(alpha: number) {
    for (const node of nodes) {
      if (node.group === 3 || node.level === 1) continue;
      const targetR = 220;
      const dx = (node.x ?? 0) - cx;
      const dy = (node.y ?? 0) - cy;
      const currentR = Math.sqrt(dx * dx + dy * dy) || 1;
      const k = (targetR - currentR) / currentR * strength * alpha;
      node.vx = (node.vx ?? 0) + dx * k;
      node.vy = (node.vy ?? 0) + dy * k;
    }
  }
  force.initialize = (n: any[]) => { nodes = n; };
  return force;
}

function getLinkCurvature(link: any): number {
  const id = typeof link.target === 'object' ? link.target.id : link.target;
  const hash = String(id).charCodeAt(0);
  return hash % 2 === 0 ? 0.15 : -0.15;
}

function getLinkControlPoint(start: any, end: any, curvature: number) {
  return {
    cx: (start.x + end.x) / 2 + curvature * (end.y - start.y),
    cy: (start.y + end.y) / 2 - curvature * (end.x - start.x),
  };
}

export function Graph({
  data, selectedNode, onNodeClick, onNodeDoubleClick,
  onBackgroundClick, isDetailsOpen, expandedNodeIds,
}: GraphProps) {
  const fgRef = useRef<ForceGraphMethods<any, any> | undefined>(undefined);
  const [hoverNode, setHoverNode] = useState<MyNode | null>(null);

  const expandedRef = useRef<Set<string> | undefined>(expandedNodeIds);
  useEffect(() => {
    expandedRef.current = expandedNodeIds;
  }, [expandedNodeIds]);

  useGraphCamera(fgRef as any, isDetailsOpen, selectedNode, data.nodes);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    const linkForce = fg.d3Force('link');
    if (linkForce) {
      (linkForce as any).distance((link: any) => {
        const srcLevel = link.source?.level ?? 2;
        const tgtLevel = link.target?.level ?? 3;
        const srcId = String(link.source?.id ?? '');
        const tgtId = String(link.target?.id ?? '');
        const expanded = expandedRef.current;

        if (srcLevel === 1 || tgtLevel === 1) {
          const categoryId = srcLevel === 1 ? tgtId : srcId;
          return expanded?.has(categoryId) ? 300 : 180;
        }
        const parentId = tgtLevel === 3 ? srcId : tgtId;
        return expanded?.has(parentId) ? 120 : 50;
      });
    }

    const chargeForce = fg.d3Force('charge');
    if (chargeForce) {
      (chargeForce as any)
        .strength((node: any) => {
          if (node.level === 1) return -800;
          if (node.group === 2) return -200 - (node.childCount || 1) * 15;
          return -30;
        })
        .distanceMax(400);
    }

    fg.d3Force('collision', createCollisionForce(getCollisionRadius, 0.85));
    fg.d3Force('radial', createRadialForce(0, 0, 0.03));
    fg.d3ReheatSimulation();
  }, [data]);

  const prevExpandedRef = useRef<Set<string> | undefined>(undefined);
  useEffect(() => {
    if (prevExpandedRef.current === undefined) {
      prevExpandedRef.current = expandedNodeIds;
      return;
    }
    prevExpandedRef.current = expandedNodeIds;

    const fg = fgRef.current;
    if (!fg) return;

    (fg as any).d3AlphaTarget?.(0.08);
    const timer = setTimeout(() => {
      (fg as any).d3AlphaTarget?.(0);
    }, 1500);
    return () => clearTimeout(timer);
  }, [expandedNodeIds]);

  const handleNodeClick = useCallback((node: any) => {
    const now = Date.now();
    const last = lastClickRef.current;
    if (last.id === node.id && now - last.time < 300) {
      onNodeDoubleClick?.(node as MyNode);
      lastClickRef.current = { id: '', time: 0 };
    } else {
      onNodeClick(node as MyNode);
      lastClickRef.current = { id: node.id, time: now };
    }
  }, [onNodeClick, onNodeDoubleClick]);

  const lastClickRef = useRef<{ id: string; time: number }>({ id: '', time: 0 });

  const highlightLinks = useMemo(() => {
    const links = new Set<any>();
    const activeNode = hoverNode || selectedNode;
    if (activeNode) {
      data.links.forEach((link: any) => {
        const srcId = typeof link.source === 'object' ? link.source.id : link.source;
        const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
        if (srcId === activeNode.id || tgtId === activeNode.id) links.add(link);
      });
    }
    return links;
  }, [data, hoverNode, selectedNode]);

  useEffect(() => {
    document.body.style.cursor = hoverNode ? 'pointer' : 'default';
  }, [hoverNode]);

  return (
    <ForceGraph2D
      ref={fgRef as any}
      graphData={data}
      backgroundColor="#0D0D0F"
      nodeLabel=""
      linkCurvature={getLinkCurvature}
      linkCanvasObject={(link: any, ctx) => {
        const start = link.source;
        const end = link.target;
        if (!start || !end || typeof start.x !== 'number') return;

        const isHighlighted = highlightLinks.has(link);
        const srcColor = getNodeColor(start);
        const tgtColor = getNodeColor(end);

        const gradient = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
        if (isHighlighted) {
          gradient.addColorStop(0, `${srcColor}CC`);
          gradient.addColorStop(1, `${tgtColor}CC`);
        } else {
          gradient.addColorStop(0, `${srcColor}22`);
          gradient.addColorStop(1, `${tgtColor}18`);
        }

        const curvature = getLinkCurvature(link);
        const { cx, cy } = getLinkControlPoint(start, end, curvature);

        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.quadraticCurveTo(cx, cy, end.x, end.y);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = isHighlighted ? 1.8 : 0.6;
        ctx.stroke();
      }}
      linkCanvasObjectMode={() => 'replace'}
      linkDirectionalParticles={(link: any) => {
        if (!selectedNode) return 0;
        const srcId = typeof link.source === 'object' ? link.source.id : link.source;
        const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
        return (srcId === selectedNode.id || tgtId === selectedNode.id) ? 3 : 0;
      }}
      linkDirectionalParticleWidth={2}
      linkDirectionalParticleColor={(link: any) => {
        const src = typeof link.source === 'object' ? link.source : null;
        return src ? getNodeColor(src) : '#ffffff';
      }}
      linkDirectionalParticleSpeed={0.004}
      onNodeClick={handleNodeClick}
      onBackgroundClick={onBackgroundClick}
      onNodeHover={(node) => setHoverNode((node as MyNode) || null)}
      nodeCanvasObject={(node: any, ctx, globalScale) => {
        drawNodeCanvasObject(node, ctx, globalScale, {
          hoverNodeId: hoverNode?.id,
          selectedNodeId: selectedNode?.id,
        });
      }}
    />
  );
}