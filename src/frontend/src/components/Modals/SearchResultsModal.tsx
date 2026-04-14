import React, { useMemo, useRef, useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import styles from './Modals.module.css';
import type { MovieData } from '../../types/movie';
import type { MyNode } from '../../types/graph';
import { drawNodeCanvasObject, getNodeColor } from '../../utils/canvasHelper';

interface SearchResultsModalProps {
  movies: MovieData[];
  onClose: () => void;
  onMovieSelect: (node: MyNode) => void;
  onClearSelection: () => void;
  isDetailsOpen: boolean;
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

function getCollisionRadius(node: any): number {
  if (node.level === 1) return 60;
  return 15;
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

export function SearchResultsModal({ movies, onClose, onMovieSelect, onClearSelection, isDetailsOpen }: SearchResultsModalProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods<any, any>>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 });
  const [hoverNode, setHoverNode] = useState<MyNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<MyNode | null>(null);

  useEffect(() => {
    if (containerRef.current) {
      setDimensions({
        width: containerRef.current.offsetWidth,
        height: containerRef.current.offsetHeight
      });
    }
  }, []);

  const graphData = useMemo(() => {
    const rootNode: MyNode = { 
      id: 'query_root', 
      name: 'Search Results', 
      group: 1, 
      level: 1, 
      val: 20 
    };
    
    const nodes: MyNode[] = [rootNode];
    const links: any[] = [];

    movies.forEach(movie => {
      const movieNode: MyNode = {
        id: `movie-${movie.id}`,
        name: movie.title,
        group: 3,
        level: 3,
        val: 10
      };
      nodes.push(movieNode);
      links.push({ source: 'query_root', target: movieNode.id });
    });

    return { nodes, links };
  }, [movies]);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    const linkForce = fg.d3Force('link');
    if (linkForce) {
      (linkForce as any).distance(70);
    }

    const chargeForce = fg.d3Force('charge');
    if (chargeForce) {
      (chargeForce as any).strength(-100).distanceMax(300);
    }

    fg.d3Force('collision', createCollisionForce(getCollisionRadius, 0.85));
    fg.d3ReheatSimulation();
  }, [graphData]);

  useEffect(() => {
    if (!fgRef.current) return;

    if (selectedNode) {
      const graphNode = graphData.nodes.find(n => n.id === selectedNode.id) as any;
      
      if (graphNode && graphNode.x !== undefined && graphNode.y !== undefined) {
        try {
          fgRef.current.centerAt(graphNode.x, graphNode.y, 800); 
          fgRef.current.zoom(3, 800);
        } catch (e) {
          console.warn("Could not focus camera on node", e);
        }
      }
    } else {
      try {
        setTimeout(() => {
          fgRef.current?.zoomToFit(800, 50);
        }, 100);
      } catch (e) {
        console.warn("Could not zoom out camera", e);
      }
    }
  }, [graphData, selectedNode]);

  useEffect(() => {
    document.body.style.cursor = hoverNode ? 'pointer' : 'default';
  }, [hoverNode]);

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div 
        className={`${styles.modal} ${isDetailsOpen ? styles.modal_shifted : ''}`} 
        style={{ width: '80vw', height: '80vh', display: 'flex', flexDirection: 'column' }} 
        onClick={(e) => e.stopPropagation()}
      >
        <button className={styles.close_btn} onClick={onClose}>×</button>
        <h2 className={styles.title} style={{ marginBottom: '10px' }}>
          Movies found: {movies.length}
        </h2>
        <p style={{ color: '#888', fontSize: '13px', marginBottom: '20px' }}>
          Click on a movie to open its analysis
        </p>

        <div ref={containerRef} style={{ flex: 1, background: '#0D0D0F', borderRadius: '8px', overflow: 'hidden', border: '1px solid #333' }}>
          {dimensions.width > 0 && (
            <ForceGraph2D
              ref={fgRef as any}
              width={dimensions.width}
              height={dimensions.height}
              graphData={graphData}
              backgroundColor="#0D0D0F"
              nodeLabel=""
              d3VelocityDecay={0.2}
              linkCurvature={getLinkCurvature}
              linkCanvasObject={(link: any, ctx) => {
                const start = link.source;
                const end = link.target;
                if (!start || !end || typeof start.x !== 'number') return;

                const isHighlighted = 
                  hoverNode?.id === start.id || hoverNode?.id === end.id ||
                  selectedNode?.id === start.id || selectedNode?.id === end.id;

                const srcColor = getNodeColor(start);
                const tgtColor = getNodeColor(end);

                const gradient = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
                if (isHighlighted) {
                  gradient.addColorStop(0, `${srcColor}CC`);
                  gradient.addColorStop(1, `${tgtColor}CC`);
                } else {
                  gradient.addColorStop(0, `${srcColor}33`);
                  gradient.addColorStop(1, `${tgtColor}33`);
                }

                const curvature = getLinkCurvature(link);
                const { cx, cy } = getLinkControlPoint(start, end, curvature);

                ctx.beginPath();
                ctx.moveTo(start.x, start.y);
                ctx.quadraticCurveTo(cx, cy, end.x, end.y);
                ctx.strokeStyle = gradient;
                ctx.lineWidth = isHighlighted ? 2.5 : 1.2;
                ctx.stroke();
              }}
              linkCanvasObjectMode={() => 'replace'}
              linkDirectionalParticles={2}
              linkDirectionalParticleWidth={2.5}
              linkDirectionalParticleColor={(link: any) => {
                const src = typeof link.source === 'object' ? link.source : null;
                return src ? getNodeColor(src) : '#ffffff';
              }}
              linkDirectionalParticleSpeed={0.006}
              onNodeHover={(node) => setHoverNode((node as MyNode) || null)}
              onNodeClick={(node) => {
                if (node.id !== 'query_root') {
                  setSelectedNode(node as MyNode);
                  onMovieSelect(node as MyNode);
                }
              }}
              onBackgroundClick={() => {
                setSelectedNode(null);
                onClearSelection();
              }}
              nodeCanvasObject={(node: any, ctx, globalScale) => {
                drawNodeCanvasObject(node, ctx, globalScale, {
                  hoverNodeId: hoverNode?.id,
                  selectedNodeId: selectedNode?.id,
                });
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}