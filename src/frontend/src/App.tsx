import { useState, useMemo, useEffect } from 'react';
import { Graph } from './components/Graph/Graph';
import { Sidebar } from './components/Sidebar/Sidebar';
import { fetchGraph } from './api/graph';
import { transformGraphData } from './utils/transform';
import type { MyNode, GraphData } from './types/graph';

export default function App() {
  const [selectedNode, setSelectedNode] = useState<MyNode | null>(null);
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });

  useEffect(() => {
    loadGraph(1);
  }, []);

  const loadGraph = async (nodeId: number) => {
      try {
        const data = await fetchGraph(nodeId);
        const transformed = transformGraphData(data);

        setGraphData(prev => {
          const newNodes = [...prev.nodes];
          const newLinks = [...prev.links];

          transformed.nodes.forEach(newNode => {
            if (!newNodes.find(n => n.id === newNode.id)) newNodes.push(newNode);
          });
          transformed.links.forEach(newLink => {
            // A simple check to avoid duplicate links
            if (!newLinks.find(l => l.source === newLink.source && l.target === newLink.target)) {
              newLinks.push(newLink);
            }
          });
          return { nodes: newNodes, links: newLinks };
        });
      } catch (e) {
        console.error(e);
      }
    };

  const visibleData = useMemo(() => {
    let visibleNodes: MyNode[] = [];

    if (activeCategory === null) {
      // Show root AND its direct children
      visibleNodes = graphData.nodes.filter(node =>
          node.level === 1 || node.level === 2 || node.level === 3
      );
    } else {
      visibleNodes = graphData.nodes.filter(node => {
        if (node.id === activeCategory) return true;

        if (node.level === 2 || node.level === 3) {
          return graphData.links.some(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;

            return (
              (sourceId === activeCategory && targetId === node.id) ||
              (targetId === activeCategory && sourceId === node.id)
            );
          });
        }

        return false;
      });
    }

    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

    const visibleLinks = graphData.links.filter(link => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
      return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
    });
    console.log("Visible Nodes Count:", visibleNodes.length);
    console.log("Visible Links Count:", visibleLinks.length);
    return { nodes: visibleNodes, links: visibleLinks };
  }, [graphData, activeCategory]);

  const handleNodeClick = async (node: MyNode) => {
    // Movie → open sidebar only
    if (node.id.startsWith('movie-') || node.level === 3) {
      setSelectedNode(node);
      return;
    }

    // Level 1 category toggle
    if (node.level === 1) {
      if (activeCategory === node.id) {
        // Collapse - go back to root
        setActiveCategory(null);
        setSelectedNode(null);
        return; // Don't reload when collapsing
      } else {
        // Expand this category
        setActiveCategory(node.id);
        setSelectedNode(null);
      }
    }

    // Load children for this node
    const numericId = Number(node.id);
    if (!isNaN(numericId)) {
      await loadGraph(numericId);
    }
  };

  const handleBackgroundClick = () => {
    setSelectedNode(null);
  };

  const handleGoBack = () => {
    setActiveCategory(null);
    setSelectedNode(null);
    loadGraph(1);
  };

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', background: '#0f0f11' }}>

      {activeCategory && (
        <button
          onClick={handleGoBack}
          style={{
            position: 'absolute',
            top: '20px',
            left: '20px',
            zIndex: 10,
            padding: '10px 20px',
            fontSize: '16px',
            background: 'rgba(255, 255, 255, 0.1)',
            color: '#fff',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '8px',
            cursor: 'pointer',
            backdropFilter: 'blur(5px)'
          }}
        >
          ← Назад к категориям
        </button>
      )}

      <Graph
        data={visibleData}
        selectedNode={selectedNode}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}
      />

      <Sidebar
        selectedNode={selectedNode}
        onClose={() => setSelectedNode(null)}
      />
    </div>
  );
}